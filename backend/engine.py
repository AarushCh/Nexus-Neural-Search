"""
Nexus retrieval engine.

Thin layer over two collaborators:

    backend/store.py    — where the catalogue lives (Postgres + pgvector)
    backend/ranking.py  — how results are scored and ordered

The public surface:

    embed_query / embed_docs      -> dense vectors (bge-small, local ONNX)
    hybrid_search(text, ...)      -> dense + lexical, RRF-fused, honestly scored
    recommend / similar_items     -> taste-vector neighbours
    ground_with_llm(query, hits)  -> grounded RAG reranking over REAL hits

Two things that used to live here are gone:

  * The Qdrant client. The catalogue moved to Postgres because free-tier vector
    clusters get reaped for inactivity, which is what took the site down twice.
  * The BM25 sparse embedding model. Postgres `tsvector` does the lexical
    channel natively and brings stemming with it, so a whole embedding model and
    its ONNX runtime dropped out of the request path.

Invariant worth protecting: ONE embedding model for ingest AND query. The
original build used MiniLM for one and bge for the other, which silently
destroyed retrieval quality.
"""

from __future__ import annotations

import html
import os
import re
import threading

from dotenv import load_dotenv

from backend import ranking, store

load_dotenv()

# --- Configuration ------------------------------------------------------------

DENSE_MODEL = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
VECTOR_SIZE = store.VECTOR_SIZE

# bge models want an instruction prefix on the QUERY side only.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# The cross-encoder now runs on fastembed's ONNX runtime — the same one already
# loaded for the dense model — instead of torch + sentence-transformers. That
# was the only reason reranking had to be disabled in production, so it is now
# ON by default and the README's claim about it is finally true.
# Off by default: the cross-encoder costs a measured 106 MB on top of the
# dense model's 186 MB, which does not fit a 512 MB instance alongside the
# app. Turn it on where there is 1 GB or more — it is a rank channel only,
# so its absence costs ordering quality, not correctness.
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "false").lower() == "true"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-nano-12b-v2-vl:free")

# How many candidates each channel contributes before fusion and reranking.
PREFETCH = int(os.getenv("PREFETCH", 60))

# Postgres trigram similarity above which a result counts as "you typed this
# title". 0.45 accepts a one- or two-character misspelling of a medium-length
# title while staying well clear of vibe queries, which score far lower against
# any single title.
TRIGRAM_PIN = float(os.getenv("TRIGRAM_PIN", 0.45))


class EngineUnavailable(RuntimeError):
    """The catalogue could not be reached, or it is empty.

    Raised instead of returning [] so the API can answer 503 rather than a 200
    with an empty list — "the index is gone" and "no results for your query"
    used to be indistinguishable from the browser, which is how a vanished
    catalogue went unnoticed.
    """


# ONNX Runtime defaults to one intra-op thread per reported core, and each one
# takes its own slice of a CPU arena that grows but never shrinks. A small
# container reports the host's core count, so the default quietly multiplies
# resident memory and the worker gets OOM-killed mid-request. Measured on one
# core: dense 186 MB, cross-encoder a further 106 MB.
ONNX_THREADS = int(os.getenv("ONNX_THREADS", "1"))


# --- Lazy singletons ----------------------------------------------------------

_lock = threading.Lock()
_dense_model = None
_cross_encoder = None
_cross_encoder_failed = False


def _get_dense():
    global _dense_model
    if _dense_model is None:
        with _lock:
            if _dense_model is None:
                from fastembed import TextEmbedding

                _dense_model = TextEmbedding(DENSE_MODEL, threads=ONNX_THREADS)
    return _dense_model


def _get_cross_encoder():
    """ONNX cross-encoder. No torch, so it fits a small instance."""
    global _cross_encoder, _cross_encoder_failed
    if _cross_encoder is None and not _cross_encoder_failed:
        with _lock:
            if _cross_encoder is None and not _cross_encoder_failed:
                try:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    _cross_encoder = TextCrossEncoder(model_name=RERANK_MODEL,
                                                      threads=ONNX_THREADS)
                except Exception as e:  # noqa: BLE001 - reranker stays optional
                    print(f"⚠️  Cross-encoder unavailable, using cosine relevance: {e}")
                    _cross_encoder_failed = True
    return _cross_encoder


def collection_health() -> dict:
    """Real catalogue state, for the /health endpoint."""
    h = store.health()
    return {"ok": h["ok"], "points": h["titles"], "collection": "media",
            "error": h["error"]}


# --- Embedding helpers --------------------------------------------------------

def embed_query(text: str) -> list[float]:
    return next(_get_dense().embed([QUERY_INSTRUCTION + text])).tolist()


def embed_docs(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in _get_dense().embed(list(texts), batch_size=64)]


# --- Helpers ------------------------------------------------------------------

def _normalize(s: str) -> str:
    # Decode HTML entities first (e.g. "Let&#039;s" -> "Let's") so name matching works.
    return re.sub(r"[^a-z0-9]+", " ", html.unescape(str(s)).lower()).strip()


def _rating(card: dict) -> float:
    try:
        return float(card.get("rating") or 0)
    except (TypeError, ValueError):
        return 0.0


def build_filter(category=None, min_rating=None, year_min=None, year_max=None,
                 tags=None, genres=None):
    """Compose a filter for the store. Returns (sql_fragment, params)."""
    return store.build_filter(category, min_rating, year_min, year_max, tags, genres)


def _split_filter(qfilter):
    if not qfilter:
        return "", {}
    return qfilter[0], qfilter[1]


# --- Scoring ------------------------------------------------------------------

def rerank_order(query: str, cards: list[dict]) -> list[str] | None:
    """Cross-encoder ranking of `cards`, best first. None when unavailable.

    Returns an ORDER, not scores. The logits themselves are not usable as
    relevance in this domain (see ranking.relevance_from_logit for the
    measurement), but the ordering they induce is genuinely better than vector
    order — dropping it makes a query like "Denis Villeneuve" return Spirited
    Away first. So it is fused as another rank channel.
    """
    cross = _get_cross_encoder() if ENABLE_RERANK else None
    if cross is None or not cards:
        return None
    docs = [f"{c.get('title','')}. {c.get('description','')}"[:512] for c in cards]
    try:
        scored = sorted(zip(cards, cross.rerank(query, docs)),
                        key=lambda t: float(t[1]), reverse=True)
        return [str(c.get("id")) for c, _ in scored]
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Rerank failed, using retrieval order: {e}")
        return None


def _score_cards(query: str, cards: list[dict], cosines: dict = None,
                 fused: dict = None, ceiling: float = 1.0,
                 lex_ranks: dict = None, lex_total: int = 0) -> list[dict]:
    """Set the %MATCH badge and the final order.

    The badge comes from the calibrated dense cosine — the one signal here that
    measures query-to-document content on a stable scale, so 90% means the same
    thing in every query. Ranks (RRF, cross-encoder) decide WHICH titles make the
    cut; they cannot say how good any of them actually are.

    Final ordering is that same relevance blended with a vote-shrunk quality
    prior, which keeps the badge monotonic down the list while letting the
    better-known of two equally relevant titles win.
    """
    if not cards:
        return []
    cosines, fused = cosines or {}, fused or {}
    lex_ranks = lex_ranks or {}

    for c in cards:
        cid = str(c.get("id"))
        if cid in cosines:
            rel = ranking.relevance_from_cosine(cosines[cid])
        else:
            # Retrieved by the lexical channel only, so there is no cosine.
            rel = ranking.relevance_from_rrf(fused.get(cid, 0.0), ceiling)
        # Matching a query that only a handful of documents match at all is
        # real content evidence, and the dense model has no way to show it:
        # every Ghibli film matched "Studio Ghibli" lexically while Spirited
        # Away's cosine left it 26th. Only ever raises, and only for queries
        # rare enough to mean something.
        if cid in lex_ranks:
            rel = max(rel, ranking.rarity_relevance(lex_total, lex_ranks[cid]))
        c["_rel"] = rel
        c["_badge"] = ranking.match_percent(rel, c.get("_pin"))
        c["_order"] = ranking.blend_score(rel, _rating(c), c.get("votes", 0))

    # Title pins always lead: if you typed the name, that IS the answer.
    #
    # Then the BADGE, and only then the quality blend. Sorting on the blend
    # alone let a well-known 61% sit above an obscure 63%, so the grid showed
    # badges that climbed as you read down it and the ranking looked broken.
    # Badges are whole numbers, so ties are common and the quality prior still
    # decides the order inside each band — it just can never contradict the
    # number on the card any more.
    cards.sort(key=lambda c: (c.get("_pin") == "exact", c.get("_pin") == "prefix",
                              c["_badge"], c["_order"]), reverse=True)
    for c in cards:
        c["score"] = c["_badge"]
        for k in ("_rel", "_order", "_badge", "_pin", "_cos"):
            c.pop(k, None)
    return cards


# --- Search -------------------------------------------------------------------

def hybrid_search(text: str, top_k: int = 12, prefetch: int = None,
                  qfilter=None) -> list[dict]:
    """Dense + lexical retrieval, RRF-fused, reranked, honestly scored."""
    prefetch = prefetch or PREFETCH
    filter_sql, params = _split_filter(qfilter)

    try:
        res = store.hybrid_candidates(text, embed_query(text), prefetch,
                                      filter_sql, params)
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"catalogue search failed: {e}") from e

    cards = res["cards"]
    if not cards:
        # Distinguish "this query matched nothing" from "there is no catalogue".
        if not store.health()["ok"]:
            raise EngineUnavailable("catalogue is empty")
        return []

    # Whole-catalogue fuzzy title matches, pinned above the semantic ranking.
    try:
        for card in store.title_candidates(text, top_k, filter_sql, params):
            # Two complementary signals. Token comparison handles partial titles
            # ("fellowship of the ring"); Postgres trigram similarity handles
            # misspellings ("spirted away"), which token comparison cannot see
            # because the typo'd word simply is not the same token.
            pin = ranking.classify_pin(text, card.get("title", ""))
            if not pin and card.get("_sim", 0) >= TRIGRAM_PIN:
                pin = "prefix"
            if not pin:
                continue
            cid = str(card["id"])
            if cid in cards:
                cards[cid]["_pin"] = pin
            else:
                card["_pin"] = pin
                cards[cid] = card
    except Exception as e:  # noqa: BLE001 - pinning is an enhancement, not the search
        print(f"⚠️  Title lookup failed: {e}")

    channels = {"dense": res["dense"], "sparse": res["lexical"]}

    # Narrow to a working set with the cheap channels, then let the cross-encoder
    # rank that set and fold its ordering in as a third channel. Reranking every
    # candidate would be wasteful; reranking none measurably hurts (a person or
    # studio query returns the wrong title first).
    prelim = ranking.weighted_rrf(channels)
    shortlist = sorted(cards.values(),
                       key=lambda c: (c.get("_pin") is not None,
                                      prelim.get(str(c.get("id")), 0.0)),
                       reverse=True)[:max(top_k * 3, 30)]

    order = rerank_order(text, shortlist)
    if order:
        channels["rerank"] = order

    fused = ranking.weighted_rrf(channels, {"dense": ranking.W_DENSE,
                                            "sparse": ranking.W_SPARSE,
                                            "rerank": ranking.W_RERANK})
    ceiling = ranking.rrf_ceiling(channels.keys(),
                                  {"dense": ranking.W_DENSE,
                                   "sparse": ranking.W_SPARSE,
                                   "rerank": ranking.W_RERANK})

    # Fused rank decides WHICH titles survive; _score_cards then orders and
    # badges them from the calibrated cosine, so the badge stays monotonic.
    survivors = sorted(shortlist,
                       key=lambda c: (c.get("_pin") is not None,
                                      fused.get(str(c.get("id")), 0.0)),
                       reverse=True)[:top_k]
    lex_ranks = {cid: i + 1 for i, cid in enumerate(res.get("lexical") or [])}
    # Rarity is only evidence of DISTINCTIVENESS for a short query. Every term
    # is ANDed into the tsquery, so a six-word sentence matches few documents
    # purely by being long — "something to watch with my parents" matched a
    # handful and shot Silver Linings Playbook to 92%. Two words that match
    # thirty documents mean something; six words that do not. Two words, not
    # three: at three the gate still fired on "mind bending sci fi", and a
    # one-or-two word query is almost always an entity — a studio, a franchise,
    # a person — which is exactly the case the dense model is weakest on.
    content_words = [w for w in re.findall(r"\w+", text) if len(w) > 2]
    lex_total = res.get("lex_total", 0) if len(content_words) <= 2 else 0
    return _score_cards(text, survivors, res["cosines"], fused, ceiling,
                        lex_ranks, lex_total)


def recommend(positive_ids: list, negative_ids: list = None, top_k: int = 12,
              exclude_ids: set = None) -> list[dict]:
    """Taste-vector neighbours. Powers /similar and personalization."""
    if not positive_ids:
        return []
    try:
        cands = store.neighbours(positive_ids, negative_ids, top_k, exclude_ids)
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"recommendation failed: {e}") from e

    results, seen_titles = [], set()
    for card in cands:
        key = _normalize(card.get("title", ""))
        if key in seen_titles:  # de-dup near-identical titles
            continue
        seen_titles.add(key)
        card["score"] = ranking.match_percent(
            ranking.relevance_from_cosine(card.get("_cos", 0.0)))
        card.pop("_cos", None)
        results.append(card)
        if len(results) >= top_k:
            break
    return results


def similar_items(item_id, top_k: int = 12) -> list[dict]:
    """Tight, on-theme neighbours: a wide candidate pool from the source item,
    rescored against the source's own text, with the same medium preferred."""
    if str(item_id).startswith("ai-"):
        return []
    src = store.by_ids([item_id])
    if not src:
        return []
    src = src[0]
    src_text = f"{src.get('title','')}. {src.get('description','')}"[:512]

    try:
        cands = store.neighbours([item_id], limit=top_k * 3, exclude_ids={item_id})
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"similar lookup failed: {e}") from e
    if not cands:
        return []

    cosines = {str(c["id"]): c.get("_cos", 0.0) for c in cands}
    # No same-medium bonus applied after scoring: re-sorting on a hidden boost
    # made the displayed badges non-monotonic (57%, 44%, 48%), which just looks
    # broken. Medium is already part of the embedded document via the `form`
    # tag, so the cosine accounts for it honestly.
    return _score_cards(src_text, cands, cosines)[:top_k]


# --- Feed / personalization / detail ------------------------------------------

def get_by_ids(ids: list) -> list[dict]:
    return store.by_ids(ids)


def get_detail(media_id: str) -> dict | None:
    """Full record including cast, crew and providers."""
    return store.detail(media_id)


def top_popular(top_k: int = 20, qfilter=None) -> list[dict]:
    """Most mainstream titles, by vote-shrunk quality.

    Replaces the old raw vote_count ordering, which surfaced whatever happened
    to be trending rather than what is actually good and well known.
    """
    filter_sql, params = _split_filter(qfilter)
    try:
        rows = store.top_by_quality(top_k, filter_sql, params, min_votes=500)
    except Exception as e:  # noqa: BLE001
        raise EngineUnavailable(f"feed query failed: {e}") from e
    for h in rows:
        h["score"] = ranking.match_percent(
            ranking.quality_prior(_rating(h), h.get("votes", 0)))
    return rows


def top_rated(top_k: int = 20, qfilter=None, min_rating: float = 6.5) -> list[dict]:
    filter_sql, params = _split_filter(qfilter)
    rows = store.top_by_quality(top_k, filter_sql, params, min_votes=200)
    for h in rows:
        h["score"] = ranking.match_percent(
            ranking.quality_prior(_rating(h), h.get("votes", 0)))
    return rows


def for_you(positive_ids: list, negative_ids: list = None, top_k: int = 20,
            exclude_ids: set = None) -> list[dict]:
    """History-driven recommendations: wishlist/likes/views positive, dismisses negative."""
    positive_ids = [i for i in positive_ids if i and not str(i).startswith("ai-")]
    if not positive_ids:
        return []
    seen, pos = set(), []
    for i in positive_ids:
        if str(i) not in seen:
            seen.add(str(i))
            pos.append(i)
    return recommend(pos[:40], [i for i in (negative_ids or []) if i][:20],
                     top_k, exclude_ids)


def facets(namespace: str, limit: int = 40, qfilter=None) -> list[dict]:
    """Tag facet counts for the filter bar."""
    filter_sql, params = _split_filter(qfilter)
    try:
        return store.facet_counts(namespace, limit, filter_sql, params)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  Facet query failed: {e}")
        return []


def random_title() -> dict | None:
    """One well-known title at random — the 'surprise me' feature."""
    return store.random_title()


# --- Grounded RAG (Nemotron via OpenRouter) -----------------------------------

def ground_with_llm(query: str, hits: list[dict], top_k: int = 12) -> list[dict]:
    """
    Grounded RAG: the LLM may only SELECT and ORDER from `hits` (real rows) and
    add a one-line reason. It cannot invent titles or posters. If anything goes
    wrong we return the hybrid `hits` unchanged (still real, still useful).
    """
    if not hits:
        return []
    if not OPENROUTER_API_KEY:
        return hits[:top_k]

    import json

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
