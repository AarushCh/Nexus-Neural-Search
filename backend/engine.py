"""
Nexus retrieval engine.

Everything expensive (Qdrant client, embedding model, BM25 encoder, cross-encoder)
is loaded once, lazily, and reused. The public surface is small:

    embed_query / embed_docs      -> dense vectors (bge-small, local)
    sparse_query / sparse_docs    -> BM25 sparse vectors (fastembed, local)
    hybrid_search(text, ...)      -> dense + BM25 fused (RRF) + cross-encoder rerank
    recommend(pos_ids, neg_ids)   -> Qdrant recommend (used by /similar + personalization)
    ground_with_llm(query, hits)  -> grounded RAG reranking over REAL hits (Nemotron)

Design goals for the rebuild:
  * ONE embedding model for ingest AND query (kills the old MiniLM-vs-bge mismatch).
  * Hybrid retrieval so exact keyword matches AND semantic "close" matches both surface.
  * A real reranker (ms-marco cross-encoder) instead of a README promise.
  * Graceful degradation: if a heavy component can't load, fall back, never crash the API.
"""

from __future__ import annotations

import html
import os
import re
import threading
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

from backend import ranking

load_dotenv()

# --- Configuration ------------------------------------------------------------

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "freeme_collection")
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"
VECTOR_SIZE = 384

DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
BM25_MODEL = os.getenv("BM25_MODEL", "Qdrant/bm25")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# bge models want an instruction prefix on the QUERY side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Feature flags. Rerank is OFF by default: the cross-encoder needs torch (~600MB) and OOMs
# on Render's 512MB free tier. Set ENABLE_RERANK=true only on a host with spare RAM (and
# install sentence-transformers + torch there).
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"
ENABLE_SPARSE = os.getenv("ENABLE_SPARSE", "true").lower() == "true"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
if QDRANT_URL.startswith("ttps://"):  # tolerate the old typo'd .env value
    QDRANT_URL = QDRANT_URL.replace("ttps://", "https://", 1)


class EngineUnavailable(RuntimeError):
    """The vector store could not be reached or the collection is missing.

    Raised instead of returning [] so the API can answer 503 rather than a 200
    with an empty list — "the index is gone" and "no results for your query"
    used to be indistinguishable from the browser.
    """


# --- Lazy singletons ----------------------------------------------------------

_lock = threading.Lock()
_dense_model = None
_sparse_model = None
_cross_encoder = None
_cross_encoder_failed = False


def get_qdrant():
    """One reused client for the whole process."""
    return _qdrant_singleton()


@lru_cache(maxsize=1)
def _qdrant_singleton():
    from qdrant_client import QdrantClient

    if QDRANT_URL:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)
    # Local fallback for offline dev.
    return QdrantClient(path=os.getenv("QDRANT_PATH", "qdrant_storage"))


def collection_health() -> dict:
    """Real state of the vector store, for the /health endpoint.

    An empty or missing collection is a FAILURE, not an empty result set — a
    deleted Qdrant cluster previously showed up as a silently empty UI.
    """
    try:
        client = get_qdrant()
        if not client.collection_exists(COLLECTION_NAME):
            return {"ok": False, "points": 0, "collection": COLLECTION_NAME,
                    "error": "collection missing"}
        points = client.count(COLLECTION_NAME, exact=False).count
        return {"ok": points > 0, "points": points, "collection": COLLECTION_NAME,
                "error": None if points else "collection empty"}
    except Exception as e:  # noqa: BLE001 - health must never raise
        return {"ok": False, "points": 0, "collection": COLLECTION_NAME, "error": str(e)}


def _get_dense():
    global _dense_model
    if _dense_model is None:
        with _lock:
            if _dense_model is None:
                # ONNX runtime (no torch) — keeps RAM under Render's 512MB free tier.
                from fastembed import TextEmbedding

                _dense_model = TextEmbedding(DENSE_MODEL)
    return _dense_model


def _get_sparse():
    global _sparse_model
    if _sparse_model is None:
        with _lock:
            if _sparse_model is None:
                from fastembed import SparseTextEmbedding

                _sparse_model = SparseTextEmbedding(BM25_MODEL)
    return _sparse_model


def _get_cross_encoder():
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is None and not _cross_encoder_failed:
        with _lock:
            if _cross_encoder is None and not _cross_encoder_failed:
                try:
                    from sentence_transformers import CrossEncoder

                    _cross_encoder = CrossEncoder(RERANK_MODEL, max_length=512)
                except Exception as e:  # noqa: BLE001 - reranker is optional
                    print(f"⚠️  Cross-encoder unavailable, skipping rerank: {e}")
                    _cross_encoder_failed = True
    return _cross_encoder


# --- Embedding helpers --------------------------------------------------------

def embed_query(text: str) -> list[float]:
    # fastembed returns normalized vectors; Qdrant cosine is scale-invariant regardless.
    return next(_get_dense().embed([QUERY_INSTRUCTION + text])).tolist()


def embed_docs(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_dense().embed(list(texts), batch_size=64)]


def _to_sparse_vector(embedding):
    from qdrant_client import models

    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def sparse_query(text: str):
    emb = next(_get_sparse().query_embed(text))
    return _to_sparse_vector(emb)


def sparse_docs(texts: list[str]):
    return [_to_sparse_vector(e) for e in _get_sparse().embed(texts)]


# --- Search -------------------------------------------------------------------

def _normalize(s: str) -> str:
    # Decode HTML entities first (e.g. "Let&#039;s" -> "Let's") so name matching works.
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(str(s)).lower()).strip()


def _rating(card: dict) -> float:
    try:
        return float(card.get("rating") or 0)
    except (TypeError, ValueError):
        return 0.0


def build_filter(category=None, min_rating=None, year_min=None, year_max=None):
    """Assemble a Qdrant payload filter for server-side faceting (indexes come from build_catalogue.py)."""
    from qdrant_client import models

    must = []
    if category:
        must.append(models.FieldCondition(key="category", match=models.MatchValue(value=str(category).upper())))
    if min_rating:
        must.append(models.FieldCondition(key="rating_f", range=models.Range(gte=float(min_rating))))
    if year_min or year_max:
        must.append(models.FieldCondition(key="year_i", range=models.Range(
            gte=int(year_min) if year_min else None,
            lte=int(year_max) if year_max else None,
        )))
    return models.Filter(must=must) if must else None


def _title_lookup(query: str, limit: int = 8, qfilter=None) -> list[dict]:
    """
    Whole-DB exact/prefix title matches via the full-text `title` index.
    This is what guarantees "correct name matches": if you search a real title,
    the actual title is pinned to the top even if vector fusion ranked it lower.
    Only exact + prefix matches are pinned (so vibe queries aren't hijacked).
    """
    from qdrant_client import models

    nq = _normalize(query)
    if not nq:
        return []
    must = [models.FieldCondition(key="title", match=models.MatchText(text=query))]
    if qfilter is not None and getattr(qfilter, "must", None):
        must += list(qfilter.must)
    try:
        points, _ = get_qdrant().scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=must),
            limit=64,
            with_payload=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Title lookup failed: {e}")
        return []

    pinned = []
    for p in points:
        card = _hit_to_dict(p)
        # Token-set matching, not string prefix: "fellowship of the ring" and
        # "star wars a new hope" are titles people type and the old exact/prefix
        # comparison found neither.
        pin = ranking.classify_pin(query, card.get("title", ""))
        if pin:
            card["_pin"] = pin
            card["_pin_sim"] = ranking.title_similarity(query, card.get("title", ""))
            pinned.append(card)
    # Best string match first; quality breaks ties between equally-good matches
    # (so "Dune" surfaces the one people mean before the 1970s TV movie).
    pinned.sort(key=lambda c: (c["_pin"] == "exact", c["_pin_sim"],
                               ranking.quality_prior(_rating(c), c.get("votes", 0))),
                reverse=True)
    for c in pinned:
        c.pop("_pin_sim", None)
    return pinned[:limit]


def _score_cards(query: str, cards: list[dict], cosines: dict = None,
                 fused: dict = None, ceiling: float = 1.0) -> list[dict]:
    """Attach a real relevance to every card, order by it, and set the %MATCH badge.

    Relevance comes from the strongest available CONTENT signal, in order:
       1. cross-encoder logit   (best: reads the query against the text)
       2. dense cosine          (absolute similarity, always available)
       3. fused RRF score       (rank-only, last resort — see ranking.py)

    Ordering then blends in a vote-shrunk quality prior, so between two equally
    relevant titles the one people have actually watched wins, while a popular
    title that does not match cannot climb over one that does.
    """
    if not cards:
        return []
    cosines, fused = cosines or {}, fused or {}

    # One batched cross-encoder pass over the whole candidate set.
    logits = {}
    cross = _get_cross_encoder() if ENABLE_RERANK else None
    if cross is not None:
        pairs = [[query, f"{c.get('title','')}. {c.get('description','')}"[:512]] for c in cards]
        try:
            for c, s in zip(cards, cross.predict(pairs)):
                logits[str(c.get("id"))] = float(s)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Rerank failed, falling back to cosine relevance: {e}")

    for c in cards:
        cid = str(c.get("id"))
        if cid in logits:
            rel = ranking.relevance_from_logit(logits[cid])
        elif cid in cosines:
            rel = ranking.relevance_from_cosine(cosines[cid])
        else:
            rel = ranking.relevance_from_rrf(fused.get(cid, 0.0), ceiling)
        c["_rel"] = rel
        c["_order"] = ranking.blend_score(rel, _rating(c), c.get("votes", 0))

    # Title pins always lead: if you typed the name, that IS the answer.
    cards.sort(key=lambda c: (c.get("_pin") == "exact", c.get("_pin") == "prefix",
                              c["_order"]), reverse=True)
    for c in cards:
        c["score"] = ranking.match_percent(c["_rel"], c.get("_pin"))
        for k in ("_rel", "_order", "_pin"):
            c.pop(k, None)
    return cards


def hybrid_search(text: str, top_k: int = 12, prefetch: int = 60, qfilter=None) -> list[dict]:
    """
    Dense + BM25 retrieval fused with RRF, then cross-encoder reranked.
    Optional `qfilter` (from build_filter) restricts by category/rating/year.
    Returns frontend-ready card dicts.
    """
    client = get_qdrant()

    # Each channel is queried SEPARATELY rather than through Qdrant's built-in
    # RRF prefetch. Server-side fusion returns only a fused rank, which throws
    # away the dense cosine — and the cosine is the one absolute relevance signal
    # available without a reranker. Two round trips buys honest %MATCH numbers
    # and per-channel weights that Qdrant's fusion does not expose.
    channels: dict[str, list[str]] = {}
    cards: dict[str, dict] = {}
    cosines: dict[str, float] = {}

    try:
        dense_hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embed_query(text),
            using=DENSE_VECTOR,
            query_filter=qfilter,
            limit=prefetch,
            with_payload=True,
        ).points
    except Exception as e:  # noqa: BLE001
        # Do NOT degrade to [] here: an unreachable/deleted collection must look
        # different from "your query matched nothing".
        raise EngineUnavailable(f"vector search failed: {e}") from e

    for p in dense_hits:
        cid = str(p.id)
        cards[cid] = _hit_to_dict(p)
        cosines[cid] = float(p.score)
    channels["dense"] = [str(p.id) for p in dense_hits]

    if ENABLE_SPARSE:
        try:
            sparse_hits = client.query_points(
                collection_name=COLLECTION_NAME,
                query=sparse_query(text),
                using=SPARSE_VECTOR,
                query_filter=qfilter,
                limit=prefetch,
                with_payload=True,
            ).points
            for p in sparse_hits:
                cards.setdefault(str(p.id), _hit_to_dict(p))
            channels["sparse"] = [str(p.id) for p in sparse_hits]
        except Exception as e:  # noqa: BLE001 - sparse is best-effort
            print(f"⚠️  Sparse query failed, dense-only: {e}")

    fused = ranking.weighted_rrf(channels)
    ceiling = ranking.rrf_ceiling(channels.keys())

    # Whole-DB title matches, pinned regardless of where fusion placed them.
    for card in _title_lookup(text, limit=top_k, qfilter=qfilter):
        cid = str(card.get("id"))
        if cid in cards:
            cards[cid]["_pin"] = card["_pin"]
        else:
            cards[cid] = card

    # Trim to a working set by fused rank before the (expensive) rerank pass.
    ordered = sorted(cards.values(),
                     key=lambda c: (c.get("_pin") is not None,
                                    fused.get(str(c.get("id")), 0.0)),
                     reverse=True)[:max(top_k * 3, 30)]

    return _score_cards(text, ordered, cosines, fused, ceiling)[:top_k]


def recommend(
    positive_ids: list,
    negative_ids: list | None = None,
    top_k: int = 12,
    exclude_ids: set | None = None,
) -> list[dict]:
    """Qdrant recommend over the dense space. Powers /similar and personalization."""
    from qdrant_client import models

    if not positive_ids:
        return []
    client = get_qdrant()
    exclude = set(str(i) for i in (exclude_ids or [])) | set(str(i) for i in positive_ids)

    try:
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=positive_ids,
                    negative=negative_ids or [],
                )
            ),
            using=DENSE_VECTOR,
            limit=top_k + len(exclude) + 5,
            with_payload=True,
        )
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"vector recommend failed: {e}") from e

    results, seen_titles = [], set()
    for p in response.points:
        if str(p.id) in exclude:
            continue
        card = _hit_to_dict(p)
        key = _normalize(card.get("title", ""))
        if key in seen_titles:  # de-dup near-identical titles
            continue
        seen_titles.add(key)
        # Qdrant returns the cosine against the combined taste vector — a real
        # similarity, so keep it rather than overwriting it with list position.
        card["_cos"] = float(p.score)
        card["score"] = ranking.match_percent(ranking.relevance_from_cosine(p.score))
        results.append(card)
        if len(results) >= top_k:
            break
    return results


def similar_items(item_id, top_k: int = 12) -> list[dict]:
    """
    "Explore similar" done right: recommend a wide candidate pool from the source
    item, cross-encoder rerank it against the source's own title+description, and
    softly prefer the same medium (Anime->Anime). Gives tight, on-theme neighbours
    instead of a raw top-K vector dump.
    """
    if str(item_id).startswith("ai-"):
        return []

    client = get_qdrant()
    try:
        got = client.retrieve(COLLECTION_NAME, ids=[item_id], with_payload=True)
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"retrieve failed: {e}") from e
    if not got:
        return []

    src = _hit_to_dict(got[0])
    src_type = str(src.get("type", "")).upper()
    src_text = f"{src.get('title','')}. {src.get('description','')}"[:512]

    cands = recommend(positive_ids=[item_id], top_k=top_k * 3, exclude_ids={item_id})
    if not cands:
        return []

    # Score against the SOURCE title's own text, so "similar" means thematically
    # close rather than merely near in vector space. The per-candidate cosine
    # from Qdrant recommend carries through as the relevance signal when no
    # cross-encoder is loaded.
    cosines = {str(c.get("id")): c.pop("_cos") for c in cands if "_cos" in c}
    scored = _score_cards(src_text, cands, cosines)

    # Prefer the same medium (Anime -> Anime), but only as a tiebreak: a much
    # better match from another medium should still win.
    scored.sort(key=lambda c: (c.get("score", 0)
                               + (6 if str(c.get("type", "")).upper() == src_type else 0)),
                reverse=True)
    return scored[:top_k]


def _hit_to_dict(point) -> dict:
    payload = dict(point.payload or {})
    # Clean HTML entities so every consumer (frontend, RAG, non-browser) sees real text.
    if payload.get("title"):
        payload["title"] = html.unescape(str(payload["title"]))
    if payload.get("description"):
        payload["description"] = html.unescape(str(payload["description"]))
    payload["id"] = point.id
    return payload


# --- Grounded RAG (Nemotron via OpenRouter) -----------------------------------

def ground_with_llm(query: str, hits: list[dict], top_k: int = 12) -> list[dict]:
    """
    Grounded RAG: the LLM may only SELECT and ORDER from `hits` (real DB rows) and
    add a one-line reason. It cannot invent titles or posters. If anything goes
    wrong we return the hybrid `hits` unchanged (still real, still useful).
    """
    if not hits:
        return []
    if not OPENROUTER_API_KEY:
        return hits[:top_k]

    import json

    # Include a recognizability signal (vote count) so the model can prefer mainstream.
    catalogue = [
        {"i": i, "title": h.get("title", ""), "type": h.get("type", ""),
         "votes": int(h.get("votes") or 0),
         "desc": (h.get("description", "") or "")[:200]}
        for i, h in enumerate(hits)
    ]
    prompt = (
        "You are a recommendation reranker. From the CANDIDATES below, pick the best "
        f"matches for the user request and return them most-relevant first.\n\n"
        f'User request: "{query}"\n\n'
        f"CANDIDATES (JSON):\n{json.dumps(catalogue, ensure_ascii=False)}\n\n"
        "Rules:\n"
        "- Only use items from CANDIDATES. Never invent titles.\n"
        "- Strongly prefer popular, mainstream, widely-recognized titles (higher 'votes'); "
        "only pick an obscure title (low/zero votes) when it is a clearly better match.\n"
        f"- Return up to {top_k} items.\n"
        'Respond with ONLY a JSON array of objects: {"i": <index>, "reason": "<one short sentence>"}.'
    )

    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
        completion = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"HTTP-Referer": "https://nexus-neural-search"},
            timeout=30,
        )
        content = completion.choices[0].message.content or ""
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if not match:
            return hits[:top_k]

        order = json.loads(match.group())
        ranked: list[dict] = []
        used = set()
        for entry in order:
            idx = entry.get("i")
            if not isinstance(idx, int) or idx < 0 or idx >= len(hits) or idx in used:
                continue
            used.add(idx)
            card = dict(hits[idx])
            reason = str(entry.get("reason", "")).strip()
            if reason:
                card["description"] = f"💡 {reason}\n\n{card.get('description', '')}".strip()
            # Keep the retrieval-derived %MATCH. The LLM reorders, but it has no
            # relevance measurement of its own — overwriting the real score with
            # the LLM's list position would throw away the only honest number.
            ranked.append(card)
            if len(ranked) >= top_k:
                break
        return ranked or hits[:top_k]
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Grounded RAG failed, returning hybrid hits: {e}")
        return hits[:top_k]


# --- Feed / personalization / detail helpers ---------------------------------

def get_by_ids(ids: list) -> list[dict]:
    """Fetch payload cards for specific point ids (order preserved as given)."""
    ids = [i for i in ids if i and not str(i).startswith("ai-")]
    if not ids:
        return []
    try:
        pts = get_qdrant().retrieve(COLLECTION_NAME, ids=list(ids), with_payload=True)
    except Exception as e:  # noqa: BLE001
        print(f"❌ retrieve failed: {e}")
        return []
    by_id = {str(p.id): _hit_to_dict(p) for p in pts}
    return [by_id[str(i)] for i in ids if str(i) in by_id]


def top_rated(top_k: int = 20, qfilter=None, min_rating: float = 6.5) -> list[dict]:
    """
    Best titles by vote-shrunk rating (optionally filtered by category).

    The previous version hid obscure perfect scores behind a hardcoded 7.5-9.2
    rating window, which is a Bayesian prior written as a guess — and it threw
    away every genuinely great title above 9.2. Now it pulls a generous
    candidate pool and ranks it by `bayesian_rating`, so a 10/10 with four votes
    sinks on its own merits and a 9.4 with 200k votes is allowed to win.
    """
    from qdrant_client import models

    must = [models.FieldCondition(key="rating_f", range=models.Range(gte=min_rating))]
    if qfilter is not None and getattr(qfilter, "must", None):
        must += list(qfilter.must)
    try:
        pts, _ = get_qdrant().scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=must),
            order_by=models.OrderBy(key="rating_f", direction=models.Direction.DESC),
            limit=max(top_k * 8, 200),  # wide pool; the prior does the real ranking
            with_payload=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  top_rated failed (did you run build_catalogue.py?): {e}")
        return []
    out = [_hit_to_dict(p) for p in pts]
    out.sort(key=lambda h: ranking.bayesian_rating(_rating(h), h.get("votes", 0)), reverse=True)
    out = out[:top_k]
    for h in out:
        h["score"] = ranking.match_percent(ranking.quality_prior(_rating(h), h.get("votes", 0)))
    return out


def top_popular(top_k: int = 20, qfilter=None) -> list[dict]:
    """
    Most mainstream titles = highest `votes` (vote_count). Only items harvested
    with a vote count qualify, so this naturally returns recognizable titles with
    working TMDB posters (see build_catalogue.py). Falls back to top_rated if the
    votes index isn't populated yet.
    """
    from qdrant_client import models

    must = [models.FieldCondition(key="votes", range=models.Range(gte=200))]
    if qfilter is not None and getattr(qfilter, "must", None):
        must += list(qfilter.must)
    try:
        pts, _ = get_qdrant().scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=must),
            order_by=models.OrderBy(key="votes", direction=models.Direction.DESC),
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  top_popular failed, falling back to top_rated: {e}")
        return top_rated(top_k=top_k, qfilter=qfilter)
    if not pts:
        return top_rated(top_k=top_k, qfilter=qfilter)
    out = [_hit_to_dict(p) for p in pts]
    for h in out:
        h["score"] = ranking.match_percent(ranking.quality_prior(_rating(h), h.get("votes", 0)))
    return out


def for_you(positive_ids: list, negative_ids: list | None = None,
            top_k: int = 20, exclude_ids: set | None = None) -> list[dict]:
    """History-driven recommendations: wishlist/likes/views as positives, dismisses negative."""
    positive_ids = [i for i in positive_ids if i and not str(i).startswith("ai-")]
    if not positive_ids:
        return []
    # Cap positives so the profile stays coherent (most-recent first, deduped).
    seen, pos = set(), []
    for i in positive_ids:
        if str(i) not in seen:
            seen.add(str(i))
            pos.append(i)
    return recommend(
        positive_ids=pos[:40],
        negative_ids=[i for i in (negative_ids or []) if i][:20],
        top_k=top_k,
        exclude_ids=exclude_ids,
    )


def enrich_detail(media_id: str, payload: dict) -> dict:
    """
    Look the title up on TMDB (search by title+year) and return trailer / cast /
    streaming providers. Returns full image URLs. Best-effort: empties on miss.
    Callers should cache the result (see MediaDetail table).
    """
    import requests

    key = os.getenv("TMDB_API_KEY")
    title = (payload.get("title") or "").strip()
    year = str(payload.get("year") or "")[:4]
    category = str(payload.get("category") or payload.get("type") or "").upper()
    prefer_tv = "TV" in category or "ANIME" in category

    out = {"tmdb_id": None, "trailer_key": None, "cast": [], "providers": [],
           "backdrop": None, "runtime": None, "poster": None}
    if not key or not title:
        return out

    IMG = "https://image.tmdb.org/t/p"
    try:
        for kind in (["tv", "movie"] if prefer_tv else ["movie", "tv"]):
            params = {"api_key": key, "query": title}
            if year:
                params["year" if kind == "movie" else "first_air_date_year"] = year
            r = requests.get(f"https://api.themoviedb.org/3/search/{kind}", params=params, timeout=8)
            results = r.json().get("results", []) if r.status_code == 200 else []
            if not results:
                continue

            tid = results[0]["id"]
            out["tmdb_id"] = tid
            bd = results[0].get("backdrop_path")
            out["backdrop"] = f"{IMG}/w780{bd}" if bd else None
            ps = results[0].get("poster_path")
            out["poster"] = f"{IMG}/w500{ps}" if ps else None

            d = requests.get(
                f"https://api.themoviedb.org/3/{kind}/{tid}",
                params={"api_key": key, "append_to_response": "videos,credits,watch/providers"},
                timeout=8,
            ).json()

            for v in d.get("videos", {}).get("results", []):
                if v.get("site") == "YouTube" and v.get("type") in ("Trailer", "Teaser"):
                    out["trailer_key"] = v.get("key")
                    break
            for c in d.get("credits", {}).get("cast", [])[:10]:
                p = c.get("profile_path")
                out["cast"].append({
                    "name": c.get("name"),
                    "character": c.get("character"),
                    "profile": f"{IMG}/w185{p}" if p else None,
                })
            prov = d.get("watch/providers", {}).get("results", {}).get("US", {})
            seen_p = set()
            for bucket in ("flatrate", "free", "ads", "rent", "buy"):
                for p in prov.get(bucket, []):
                    n = p.get("provider_name")
                    if n and n not in seen_p:
                        seen_p.add(n)
                        logo = p.get("logo_path")
                        out["providers"].append({"name": n, "logo": f"{IMG}/w92{logo}" if logo else None})
            rt = d.get("runtime") or (d.get("episode_run_time") or [None])[0]
            out["runtime"] = rt
            break
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  enrich_detail error: {e}")
    return out
