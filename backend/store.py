"""
Catalogue storage on PostgreSQL + pgvector.

This replaces Qdrant. Not because Qdrant is bad — the hybrid retrieval it gave us
was correct — but because a free-tier managed vector cluster is reaped after a
period of inactivity, and a portfolio site that sits idle between visits is
precisely the profile that gets reaped. It happened twice. The durable fix is to
put the vectors in the database that already holds the users, wishlists and
history, and which does not delete anything.

What Postgres gives us that the Qdrant setup did not:

  * ONE datastore instead of two, so one fewer thing to provision, secure,
    keep awake, and have deleted overnight.
  * Real hybrid retrieval in a single query: HNSW over pgvector for the dense
    channel, `tsvector` + `ts_rank_cd` for the lexical channel. Postgres full
    text search brings stemming and phrase matching, which raw BM25 did not.
  * Honest relevance. Qdrant's server-side RRF returns only a fused rank and
    discards the cosine; here both channels' own scores come back, so the
    %MATCH badge can be a real content signal (see backend/ranking.py).
  * Facet filtering and COUNTS over the new tag system, via GIN indexes on
    array columns — awkward in Qdrant, native here.

Layout: a lean `media` table carrying everything search touches, and a separate
`media_extra` holding the heavy detail blob (cast, crew, providers, alt titles).
Splitting them keeps the hot table small — the detail payload is roughly as big
as everything else combined and is only read when someone opens a title.
"""

from __future__ import annotations

import json
import os

from sqlalchemy import text

from backend.database import engine

VECTOR_SIZE = 384

# Postgres text-search configuration. 'english' gives stemming and stopword
# removal, so "haunting" matches "haunted".
TS_CONFIG = os.getenv("TS_CONFIG", "english")

# float16 vectors, when the server supports them.
#
# Measured on 5,000 realistic rows, the HNSW index was the single largest object
# at ~2.0 KB/title, with the vector column itself adding ~1.5 KB more. halfvec
# halves both. The precision loss is irrelevant here: float16 carries ~3 decimal
# digits and we only ever compare cosine distances for ranking, never accumulate.
#
# Requires pgvector >= 0.7. Detected rather than assumed, because a managed
# provider on an older extension would otherwise fail at schema creation with a
# confusing type error.
_VECTOR_TYPE: str | None = None


def vector_type() -> str:
    """'halfvec' where available, else 'vector'. Cached after the first check."""
    global _VECTOR_TYPE
    if _VECTOR_TYPE is None:
        if os.getenv("FORCE_VECTOR_TYPE"):
            _VECTOR_TYPE = os.environ["FORCE_VECTOR_TYPE"]
            return _VECTOR_TYPE
        try:
            with engine.connect() as cx:
                cx.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                cx.commit()
                v = cx.execute(text(
                    "SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar()
            major, minor = (int(x) for x in str(v).split(".")[:2])
            _VECTOR_TYPE = "halfvec" if (major, minor) >= (0, 7) else "vector"
        except Exception:  # noqa: BLE001 - fall back to the universally supported type
            _VECTOR_TYPE = "vector"
    return _VECTOR_TYPE


# How wide to walk the HNSW graph when a filter is present. The default (40) is
# tuned for unfiltered search and is far too narrow once a WHERE clause throws
# most candidates away.
FILTERED_EF_SEARCH = int(os.getenv("FILTERED_EF_SEARCH", "200"))

_ITERATIVE_SCAN = None


def _supports_iterative_scan() -> bool:
    """pgvector >= 0.8 can keep scanning until the LIMIT is satisfied."""
    global _ITERATIVE_SCAN
    if _ITERATIVE_SCAN is None:
        try:
            with engine.connect() as cx:
                v = cx.execute(text(
                    "SELECT extversion FROM pg_extension WHERE extname='vector'")).scalar()
            major, minor = (int(x) for x in str(v).split(".")[:2])
            _ITERATIVE_SCAN = (major, minor) >= (0, 8)
        except Exception:  # noqa: BLE001 - assume the conservative path
            _ITERATIVE_SCAN = False
    return _ITERATIVE_SCAN


def _tune_filtered_scan(cx) -> None:
    """Make an HNSW scan survive a selective WHERE clause.

    The index walks a fixed-size candidate list and the filter is applied to
    whatever that walk happens to find — so a selective filter returns fewer
    rows than LIMIT, or none at all, while thousands of matching rows sit in the
    table. Searching "something mellow chill" with category=ANIME returned zero
    for exactly this reason: every candidate in the default window was
    live-action. SET LOCAL, so it lasts one statement and never leaks into
    another request on the same pooled connection.
    """
    stmts = [f"SET LOCAL hnsw.ef_search = {FILTERED_EF_SEARCH}"]
    if _supports_iterative_scan():
        stmts.append("SET LOCAL hnsw.iterative_scan = relaxed_order")
    for stmt in stmts:
        # A savepoint, so an unsupported knob on some other Postgres build
        # cannot poison the transaction and take search down with it. Worst
        # case the scan runs with server defaults, exactly as it did before.
        try:
            with cx.begin_nested():
                cx.execute(text(stmt))
        except Exception:  # noqa: BLE001 - tuning is best-effort
            pass


class StoreUnavailable(RuntimeError):
    """The catalogue tables are missing or unreachable."""


# --- Schema -------------------------------------------------------------------

def _schema_sql() -> str:
    return f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS media (
    id                TEXT PRIMARY KEY,
    tmdb_id           INTEGER NOT NULL,
    tmdb_kind         TEXT    NOT NULL,
    imdb_id           TEXT,
    title             TEXT    NOT NULL,
    original_title    TEXT,
    description       TEXT,
    tagline           TEXT,
    image             TEXT,
    image_sm          TEXT,
    backdrop          TEXT,
    category          TEXT,
    types             TEXT[]  NOT NULL DEFAULT '{{}}',
    forms             TEXT[]  NOT NULL DEFAULT '{{}}',
    genres            TEXT[]  NOT NULL DEFAULT '{{}}',
    tags              TEXT[]  NOT NULL DEFAULT '{{}}',
    rating            REAL    NOT NULL DEFAULT 0,
    votes             INTEGER NOT NULL DEFAULT 0,
    year              INTEGER NOT NULL DEFAULT 0,
    runtime           INTEGER,
    seasons           INTEGER,
    episodes          INTEGER,
    status            TEXT,
    certification     TEXT,
    original_language TEXT,
    release_date      TEXT,
    trailer_key       TEXT,
    embedding         {vector_type()}({VECTOR_SIZE}),
    tsv               tsvector
);

-- The heavy half, read only when a detail page is opened.
CREATE TABLE IF NOT EXISTS media_extra (
    id      TEXT PRIMARY KEY REFERENCES media(id) ON DELETE CASCADE,
    payload JSONB NOT NULL
);
"""

# Built AFTER the bulk load: creating HNSW on an empty table and inserting into
# it is far slower than loading first and indexing once.
def _index_sql() -> list:
    vt = vector_type()
    return [
    # m/ef_construction above the defaults: the catalogue is written once and
    # read forever, so spend the build time on recall.
    "CREATE INDEX IF NOT EXISTS media_embedding_idx ON media "
    f"USING hnsw (embedding {vt}_cosine_ops) WITH (m = 16, ef_construction = 96)",
    "CREATE INDEX IF NOT EXISTS media_tsv_idx      ON media USING gin (tsv)",
    "CREATE INDEX IF NOT EXISTS media_tags_idx     ON media USING gin (tags)",
    "CREATE INDEX IF NOT EXISTS media_genres_idx   ON media USING gin (genres)",
    "CREATE INDEX IF NOT EXISTS media_forms_idx    ON media USING gin (forms)",
    "CREATE INDEX IF NOT EXISTS media_types_idx    ON media USING gin (types)",
    # Trigram index on the title powers fuzzy title lookup — the old exact/prefix
    # match found neither typos nor partial titles.
    "CREATE INDEX IF NOT EXISTS media_title_trgm   ON media USING gin (title gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS media_orig_trgm    ON media USING gin (original_title gin_trgm_ops)",
    "CREATE INDEX IF NOT EXISTS media_category_idx ON media (category)",
    "CREATE INDEX IF NOT EXISTS media_votes_idx    ON media (votes DESC)",
    "CREATE INDEX IF NOT EXISTS media_rating_idx   ON media (rating DESC)",
    "CREATE INDEX IF NOT EXISTS media_year_idx     ON media (year)",
]


def create_schema() -> None:
    with engine.begin() as cx:
        for stmt in _schema_sql().strip().split(";\n\n"):
            if stmt.strip():
                cx.execute(text(stmt))


def create_indexes() -> None:
    """Built AFTER the bulk load — inserting into an existing HNSW index is far
    slower than loading first and indexing once."""
    with engine.begin() as cx:
        for stmt in _index_sql():
            cx.execute(text(stmt))


def analyze() -> None:
    """Refresh planner statistics; without this the first queries after a bulk
    load can pick sequential scans over the new indexes."""
    with engine.connect() as cx:
        cx.execute(text("COMMIT"))
        cx.execute(text("ANALYZE media"))


# --- Health -------------------------------------------------------------------

def health() -> dict:
    """Real state of the catalogue. An empty table is a FAILURE, not an empty
    result set — a vanished index used to look exactly like 'no matches'."""
    if engine.dialect.name != "postgresql":
        # Without DATABASE_URL the app falls back to SQLite for local dev, and
        # the first catalogue query then fails with an opaque "no such function"
        # error. Say the actual cause instead.
        return {"ok": False, "titles": 0,
                "error": "DATABASE_URL is not set to Postgres "
                         f"(running on {engine.dialect.name}); the catalogue "
                         "needs Postgres with pgvector"}
    try:
        with engine.connect() as cx:
            exists = cx.execute(text(
                "SELECT to_regclass('public.media') IS NOT NULL")).scalar()
            if not exists:
                return {"ok": False, "titles": 0, "error": "media table missing"}
            n = cx.execute(text("SELECT count(*) FROM media")).scalar() or 0
            return {"ok": n > 0, "titles": int(n),
                    "error": None if n else "catalogue empty"}
    except Exception as e:  # noqa: BLE001 - health must never raise
        return {"ok": False, "titles": 0, "error": str(e)}


def storage_report() -> dict:
    """Bytes actually used, so catalogue size can be chosen from measurement
    rather than from an estimate."""
    with engine.connect() as cx:
        rows = cx.execute(text("""
            SELECT pg_total_relation_size('media')       AS media_bytes,
                   COALESCE(pg_total_relation_size('media_extra'), 0) AS extra_bytes,
                   (SELECT count(*) FROM media)          AS titles
        """)).mappings().first()
    n = max(int(rows["titles"]), 1)
    total = int(rows["media_bytes"]) + int(rows["extra_bytes"])
    return {
        "titles": int(rows["titles"]),
        "media_mb": round(int(rows["media_bytes"]) / 1e6, 1),
        "extra_mb": round(int(rows["extra_bytes"]) / 1e6, 1),
        "total_mb": round(total / 1e6, 1),
        "bytes_per_title": int(total / n),
        "projected_100k_mb": round(total / n * 100_000 / 1e6, 1),
        # Below a few thousand rows the fixed costs (8KB pages, minimum index
        # sizes, TOAST tables) dominate and bytes-per-title is wildly inflated,
        # so the projection is noise. Say so rather than letting it mislead.
        "reliable": int(rows["titles"]) >= 2000,
    }


# --- Filters ------------------------------------------------------------------

def build_filter(category=None, min_rating=None, year_min=None, year_max=None,
                 tags=None, genres=None) -> tuple[str, dict]:
    """Compose a WHERE fragment plus its bound parameters.

    Returns SQL text and params separately so every value stays a bound
    parameter — string-formatting user input into SQL is how injection happens.
    """
    clauses, params = [], {}
    if category:
        clauses.append("category = :category")
        params["category"] = str(category).upper()
    if min_rating:
        clauses.append("rating >= :min_rating")
        params["min_rating"] = float(min_rating)
    if year_min:
        clauses.append("year >= :year_min")
        params["year_min"] = int(year_min)
    if year_max:
        clauses.append("year <= :year_max")
        params["year_max"] = int(year_max)
    if tags:
        # @> is an array-containment test: must carry ALL the requested tags,
        # and it uses the GIN index.
        clauses.append("tags @> :tags")
        params["tags"] = list(tags)
    if genres:
        clauses.append("genres && :genres")   # overlap: ANY of these genres
        params["genres"] = list(genres)
    return (" AND ".join(clauses), params)


def _where(fragment: str, extra: str = "") -> str:
    parts = [p for p in (fragment, extra) if p]
    return ("WHERE " + " AND ".join(parts)) if parts else ""


# --- Retrieval ----------------------------------------------------------------

CARD_COLUMNS = """
    id, tmdb_id, tmdb_kind, imdb_id, title, original_title, description, tagline,
    image, image_sm, backdrop, category, types, forms, genres, tags,
    rating, votes, year, runtime, seasons, episodes, status, certification,
    original_language, release_date, trailer_key
"""


def _card(row) -> dict:
    card = dict(row)
    card["rating_f"] = card.get("rating")
    card["year_i"] = card.get("year")
    card["type"] = (card.get("category") or "MOVIE").title()
    card["genre"] = ", ".join(card.get("genres") or [])
    return card


def hybrid_candidates(query: str, embedding: list, limit: int = 60,
                      filter_sql: str = "", params: dict = None) -> dict:
    """Dense + lexical retrieval in ONE round trip.

    Each channel returns its own ranking AND its own score. Keeping the cosine
    is the point: a fused rank alone cannot say whether the top hit is actually
    any good, which is what made the old %MATCH badge meaningless.

    Returns {"cards": {id: card}, "dense": [ids], "lexical": [ids],
             "cosines": {id: float}}.
    """
    params = dict(params or {})
    params["q"] = query
    params["emb"] = _vector_literal(embedding)
    params["lim"] = int(limit)

    sql = f"""
    WITH q AS (SELECT websearch_to_tsquery(:ts_config, :q) AS tsq),
    dense AS (
        SELECT id, 1 - (embedding <=> CAST(:emb AS {vector_type()}({VECTOR_SIZE}))) AS cosine,
               ROW_NUMBER() OVER (ORDER BY embedding <=> CAST(:emb AS {vector_type()}({VECTOR_SIZE}))) AS rnk
        FROM media
        {_where(filter_sql, "embedding IS NOT NULL")}
        ORDER BY embedding <=> CAST(:emb AS {vector_type()}({VECTOR_SIZE}))
        LIMIT :lim
    ),
    lexical AS (
        SELECT m.id, ts_rank_cd(m.tsv, q.tsq) AS lex,
               ROW_NUMBER() OVER (ORDER BY ts_rank_cd(m.tsv, q.tsq) DESC) AS rnk
        FROM media m, q
        {_where(filter_sql, "m.tsv @@ q.tsq")}
        ORDER BY ts_rank_cd(m.tsv, q.tsq) DESC
        LIMIT :lim
    ),
    hits AS (
        SELECT id FROM dense UNION SELECT id FROM lexical
    ),
    -- How many documents match this query at ALL. A tiny number means the query
    -- is distinctive ("ghibli"), which is evidence a cosine cannot express.
    -- One aggregate over the GIN index, same value repeated on every row.
    lexcount AS (
        SELECT count(*) AS n FROM media m, q
        {_where(filter_sql, "m.tsv @@ q.tsq")}
    )
    SELECT {CARD_COLUMNS},
           d.cosine AS _cosine, d.rnk AS _dense_rank, l.rnk AS _lex_rank,
           (SELECT n FROM lexcount) AS _lex_total
    FROM media m
    JOIN hits USING (id)
    LEFT JOIN dense d USING (id)
    LEFT JOIN lexical l USING (id)
    """
    params["ts_config"] = TS_CONFIG
    with engine.connect() as cx:
        if filter_sql:
            _tune_filtered_scan(cx)
        rows = cx.execute(text(sql), params).mappings().all()

    cards, cosines, dense, lexical = {}, {}, [], []
    lex_total = 0
    for r in rows:
        lex_total = int(r["_lex_total"] or 0)
        cid = r["id"]
        cards[cid] = _card({k: v for k, v in r.items() if not k.startswith("_")})
        if r["_cosine"] is not None:
            cosines[cid] = float(r["_cosine"])
        if r["_dense_rank"] is not None:
            dense.append((int(r["_dense_rank"]), cid))
        if r["_lex_rank"] is not None:
            lexical.append((int(r["_lex_rank"]), cid))
    dense.sort()
    lexical.sort()
    return {"cards": cards, "cosines": cosines, "lex_total": lex_total,
            "dense": [c for _, c in dense], "lexical": [c for _, c in lexical]}


def title_candidates(query: str, limit: int = 12, filter_sql: str = "",
                     params: dict = None) -> list[dict]:
    """Fuzzy title lookup via trigram similarity.

    Postgres `similarity()` catches typos and partial titles that the previous
    exact/prefix string comparison missed entirely.

    Matches the ORIGINAL title too. The dense model is English-only and the
    tsvector uses the english config, so neither can find a film typed in its
    own language — searching 千と千尋の神隠し returned three unrelated Japanese
    titles instead of Spirited Away. Trigram over original_title is the only
    channel that can answer that query.
    """
    params = dict(params or {})
    params["q"] = query
    params["lim"] = int(limit)
    # Containment match, as a bound parameter so no literal % ever enters the
    # SQL text (see the escaping note below).
    #
    # Only for queries carrying non-ASCII characters. pg_trgm keeps only what
    # the database locale calls alphanumeric, so CJK can yield NO trigrams at
    # all and similarity() is then permanently 0 — searching a film by its
    # Japanese title found nothing at all. LIKE does not tokenise, so it works
    # where trigram is blind. Restricted to exactly that case on purpose: an
    # ASCII "%war%" would match hundreds of titles and pin them all, and
    # trigram already handles ASCII correctly.
    q_clean = query.strip().lower()
    needs_like = len(q_clean) >= 2 and any(ord(c) > 127 for c in q_clean)
    params["like"] = f"%{q_clean}%" if needs_like else ""
    # NOTE the single `%`. SQLAlchemy doubles literal percent signs when the
    # driver uses pyformat paramstyle (psycopg2), and psycopg2 then halves them
    # again — so writing `%%` here reaches Postgres as `%%`, which is not an
    # operator. A single `%` is what survives the round trip.
    # `%` is the pg_trgm similarity operator and is what uses the GIN index;
    # similarity() in the ORDER BY only re-scores the rows it already found.
    sql = f"""
    SELECT {CARD_COLUMNS},
           GREATEST(similarity(title, :q),
                    similarity(COALESCE(original_title, ''), :q),
                    CASE WHEN :like <> '' AND (
                             lower(COALESCE(original_title, '')) LIKE :like
                             OR lower(title) LIKE :like)
                         THEN 0.95 ELSE 0 END) AS _sim
    FROM media m
    {_where(filter_sql, "(title % :q OR original_title % :q OR (:like <> '' AND "
                        "(lower(COALESCE(original_title, '')) LIKE :like "
                        "OR lower(title) LIKE :like)))")}
    ORDER BY _sim DESC, votes DESC
    LIMIT :lim
    """
    with engine.connect() as cx:
        rows = cx.execute(text(sql), params).mappings().all()
    out = []
    for r in rows:
        card = _card({k: v for k, v in r.items() if not k.startswith("_")})
        # Keep the trigram score: it is the only signal that survives an
        # intra-word typo ("spirted away"), which token comparison cannot see.
        card["_sim"] = float(r["_sim"] or 0)
        out.append(card)
    return out


def by_ids(ids: list) -> list[dict]:
    """Cards for specific ids, in the order given."""
    ids = [str(i) for i in ids if i and not str(i).startswith("ai-")]
    if not ids:
        return []
    with engine.connect() as cx:
        rows = cx.execute(text(
            f"SELECT {CARD_COLUMNS} FROM media WHERE id = ANY(:ids)"),
            {"ids": ids}).mappings().all()
    by_id = {r["id"]: _card(r) for r in rows}
    return [by_id[i] for i in ids if i in by_id]


def detail(media_id: str) -> dict | None:
    """Full record including the heavy blob (cast, crew, providers, alt titles)."""
    with engine.connect() as cx:
        row = cx.execute(text(
            f"SELECT {CARD_COLUMNS}, e.payload FROM media m "
            "LEFT JOIN media_extra e USING (id) WHERE m.id = :id"),
            {"id": str(media_id)}).mappings().first()
    if not row:
        return None
    card = _card({k: v for k, v in row.items() if k != "payload"})
    extra = row["payload"]
    if extra:
        card.update(extra if isinstance(extra, dict) else json.loads(extra))
    return card


def neighbours(positive_ids: list, negative_ids: list = None, limit: int = 12,
               exclude_ids: set = None, filter_sql: str = "",
               params: dict = None) -> list[dict]:
    """Vector recommendation from a taste centroid.

    The centroid is computed in SQL (AVG over the positive rows' embeddings) and
    negatives are subtracted, which is Rocchio feedback: pull toward what you
    liked, push away from what you dismissed.
    """
    positive_ids = [str(i) for i in positive_ids if i and not str(i).startswith("ai-")]
    if not positive_ids:
        return []
    negative_ids = [str(i) for i in (negative_ids or []) if i]
    exclude = {str(i) for i in (exclude_ids or [])} | set(positive_ids)

    p = dict(params or {})
    p.update({"pos": positive_ids, "neg": negative_ids or [""],
              "excl": list(exclude) or [""], "lim": int(limit)})

    # AVG() is defined for `vector`, so average in that type and cast back —
    # works identically whether the column is vector or halfvec.
    vt = vector_type()
    sql = f"""
    WITH centroid AS (
        SELECT AVG(embedding::vector)::{vt} AS v FROM media WHERE id = ANY(:pos)
    ),
    away AS (
        SELECT AVG(embedding::vector)::{vt} AS v FROM media WHERE id = ANY(:neg)
    )
    SELECT {CARD_COLUMNS},
           1 - (m.embedding <=> centroid.v) AS _cosine
    FROM media m, centroid
    LEFT JOIN away ON TRUE
    {_where(filter_sql, "m.embedding IS NOT NULL AND NOT (m.id = ANY(:excl))")}
    ORDER BY (m.embedding <=> centroid.v)
             - COALESCE(0.35 * (m.embedding <=> away.v), 0) ASC
    LIMIT :lim
    """
    with engine.connect() as cx:
        rows = cx.execute(text(sql), p).mappings().all()
    out = []
    for r in rows:
        card = _card({k: v for k, v in r.items() if not k.startswith("_")})
        card["_cos"] = float(r["_cosine"]) if r["_cosine"] is not None else 0.0
        out.append(card)
    return out


def top_by_quality(limit: int = 20, filter_sql: str = "", params: dict = None,
                   min_votes: int = 200) -> list[dict]:
    """Best titles by vote-shrunk rating, ranked in SQL.

    This is the IMDb weighted-rating formula inline, which replaces the old
    hardcoded 7.5-9.2 rating window — a Bayesian prior written as a guess that
    also discarded everything above 9.2.
    """
    p = dict(params or {})
    p.update({"lim": int(limit), "min_votes": int(min_votes)})
    sql = f"""
    SELECT {CARD_COLUMNS},
           (votes::float / (votes + 500)) * rating
         + (500.0 / (votes + 500)) * 6.7 AS _wr
    FROM media m
    {_where(filter_sql, "votes >= :min_votes")}
    ORDER BY _wr DESC
    LIMIT :lim
    """
    with engine.connect() as cx:
        rows = cx.execute(text(sql), p).mappings().all()
    return [_card({k: v for k, v in r.items() if not k.startswith("_")}) for r in rows]


def facet_counts(namespace: str, limit: int = 40, filter_sql: str = "",
                 params: dict = None) -> list[dict]:
    """Most common tags in a namespace, with counts.

    Powers a real faceted filter bar. Qdrant made this awkward; here it is one
    unnest and a group-by over a GIN-indexed array.
    """
    p = dict(params or {})
    p.update({"prefix": f"{namespace}:%", "lim": int(limit)})
    sql = f"""
    SELECT t AS tag, count(*) AS n
    FROM media m, unnest(m.tags) AS t
    {_where(filter_sql, "t LIKE :prefix")}
    GROUP BY t ORDER BY n DESC LIMIT :lim
    """
    with engine.connect() as cx:
        rows = cx.execute(text(sql), p).mappings().all()
    return [{"tag": r["tag"], "label": r["tag"].split(":", 1)[1].replace("-", " "),
             "count": int(r["n"])} for r in rows]


def random_title(min_votes: int = 2000) -> dict | None:
    """One well-known title at random — the 'surprise me' feature."""
    with engine.connect() as cx:
        row = cx.execute(text(
            f"SELECT {CARD_COLUMNS} FROM media m WHERE votes >= :v "
            "ORDER BY random() LIMIT 1"), {"v": int(min_votes)}).mappings().first()
    return _card(row) if row else None


# --- Writing ------------------------------------------------------------------

def upsert(rows: list[dict], documents: list[str], embeddings: list[list]) -> int:
    """Bulk upsert. Idempotent on id, so re-running a build never duplicates."""
    payload = []
    for rec, doc, emb in zip(rows, documents, embeddings):
        payload.append({
            "id": rec["id"], "tmdb_id": rec["tmdb_id"], "tmdb_kind": rec["tmdb_kind"],
            "imdb_id": rec.get("imdb_id"), "title": rec["title"],
            "original_title": rec.get("original_title"),
            "description": rec.get("description"), "tagline": rec.get("tagline"),
            "image": rec.get("image"), "image_sm": rec.get("image_sm"),
            "backdrop": rec.get("backdrop"), "category": rec.get("category"),
            "types": rec.get("types") or [], "forms": rec.get("forms") or [],
            "genres": rec.get("genres") or [], "tags": rec.get("tags") or [],
            "rating": float(rec.get("rating") or 0), "votes": int(rec.get("votes") or 0),
            "year": int(rec.get("year_i") or 0), "runtime": rec.get("runtime"),
            "seasons": rec.get("seasons"), "episodes": rec.get("episodes"),
            "status": rec.get("status"), "certification": rec.get("certification"),
            "original_language": rec.get("original_language"),
            "release_date": rec.get("release_date"),
            "trailer_key": rec.get("trailer_key"),
            "embedding": _vector_literal(emb), "doc": doc,
        })

    vt = f"{vector_type()}({VECTOR_SIZE})"
    sql = text(f"""
        INSERT INTO media (
            id, tmdb_id, tmdb_kind, imdb_id, title, original_title, description,
            tagline, image, image_sm, backdrop, category, types, forms, genres,
            tags, rating, votes, year, runtime, seasons, episodes, status,
            certification, original_language, release_date, trailer_key,
            embedding, tsv)
        VALUES (
            :id, :tmdb_id, :tmdb_kind, :imdb_id, :title, :original_title, :description,
            :tagline, :image, :image_sm, :backdrop, :category, :types, :forms, :genres,
            :tags, :rating, :votes, :year, :runtime, :seasons, :episodes, :status,
            :certification, :original_language, :release_date, :trailer_key,
            CAST(:embedding AS {vt}), to_tsvector('{TS_CONFIG}', :doc))
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title, original_title = EXCLUDED.original_title,
            description = EXCLUDED.description, tagline = EXCLUDED.tagline,
            image = EXCLUDED.image, image_sm = EXCLUDED.image_sm,
            backdrop = EXCLUDED.backdrop, category = EXCLUDED.category,
            types = EXCLUDED.types, forms = EXCLUDED.forms, genres = EXCLUDED.genres,
            tags = EXCLUDED.tags, rating = EXCLUDED.rating, votes = EXCLUDED.votes,
            year = EXCLUDED.year, runtime = EXCLUDED.runtime,
            seasons = EXCLUDED.seasons, episodes = EXCLUDED.episodes,
            status = EXCLUDED.status, certification = EXCLUDED.certification,
            original_language = EXCLUDED.original_language,
            release_date = EXCLUDED.release_date, trailer_key = EXCLUDED.trailer_key,
            embedding = EXCLUDED.embedding, tsv = EXCLUDED.tsv
    """)
    with engine.begin() as cx:
        cx.execute(sql, payload)
    return len(payload)


def upsert_extra(rows: list[dict]) -> int:
    """The heavy detail blob, kept out of the hot search table."""
    payload = [{"id": r["id"],
                "payload": json.dumps({k: r.get(k) for k in
                                       ("cast", "crew", "providers", "alt_titles",
                                        "imdb_rating", "imdb_votes", "anilist_id")},
                                      ensure_ascii=False)}
               for r in rows]
    with engine.begin() as cx:
        cx.execute(text("""
            INSERT INTO media_extra (id, payload)
            VALUES (:id, CAST(:payload AS jsonb))
            ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload
        """), payload)
    return len(payload)


def drop_all() -> None:
    with engine.begin() as cx:
        cx.execute(text("DROP TABLE IF EXISTS media_extra"))
        cx.execute(text("DROP TABLE IF EXISTS media"))


def _vector_literal(vec) -> str:
    """pgvector accepts '[1,2,3]' text and casts it, so no extra driver
    dependency is needed just to send a vector."""
    return "[" + ",".join(f"{float(x):.6f}" for x in vec) + "]"
