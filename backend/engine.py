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


def _rerank(query: str, hits: list[dict]) -> list[dict]:
    """Cross-encoder rerank (semantic ordering). Returns hits reordered, not truncated."""
    if not hits:
        return []
    cross = _get_cross_encoder() if ENABLE_RERANK else None
    if cross is not None:
        pairs = [[query, f"{h.get('title','')}. {h.get('description','')}"[:512]] for h in hits]
        try:
            scores = cross.predict(pairs)
            for h, s in zip(hits, scores):
                h["rerank"] = float(s)
            hits.sort(key=lambda h: h["rerank"], reverse=True)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Rerank failed, using fusion order: {e}")
    return hits


def build_filter(category=None, min_rating=None, year_min=None, year_max=None):
    """Assemble a Qdrant payload filter for server-side faceting (needs build_indexes.py)."""
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

    exact, prefix = [], []
    for p in points:
        card = _hit_to_dict(p)
        nt = _normalize(card.get("title", ""))
        if nt == nq:
            card["_pin"] = "exact"
            exact.append(card)
        elif nt.startswith(nq + " ") or nt.startswith(nq):
            card["_pin"] = "prefix"
            prefix.append(card)
    exact.sort(key=_rating, reverse=True)
    prefix.sort(key=_rating, reverse=True)
    return (exact + prefix)[:limit]


def _score_for_ui(rank: int, total: int) -> int:
    """Rank-based %MATCH — fallback only, used when no reranker score exists."""
    if total <= 1:
        return 99
    return int(99 - (rank / max(total - 1, 1)) * 39)


def _sigmoid(x: float) -> float:
    import math
    if x <= -30:
        return 0.0
    if x >= 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


# Calibration for the cross-encoder logit -> %MATCH. Raw sigmoid collapses
# (short/vague queries score uniformly negative -> everything ~0%), so we shift
# and temperature-scale first. Tuned on ms-marco-MiniLM logits so an irrelevant
# hit (~-11) reads ~30%, a borderline one (~-3) ~70%, and a strong match (~+3+)
# ~90%+ — an honest spread rather than a fake 99% for the top of every list.
_CAL_SHIFT = 7.0
_CAL_TEMP = 4.6


def _calibrate_scores(query: str, cards: list[dict], logit_key: str = "rerank") -> None:
    """Attach a genuine `score` (% MATCH) to each card from the cross-encoder
    relevance logit (sigmoid -> probability), so the badge reflects how well a
    result actually matches the query — not merely its rank position.

    Cards missing a logit (e.g. exact/prefix title pins that bypassed the
    reranker) are scored on the spot. Exact/prefix pins are floored high (you
    typed the name), other results are capped just below so the pins still read
    as the strongest. Falls back to rank-based % only when the reranker is off.
    """
    missing = [c for c in cards if logit_key not in c]
    cross = _get_cross_encoder() if (ENABLE_RERANK and missing) else None
    if cross is not None:
        pairs = [[query, f"{c.get('title','')}. {c.get('description','')}"[:512]] for c in missing]
        try:
            for c, s in zip(missing, cross.predict(pairs)):
                c[logit_key] = float(s)
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Score calibration failed: {e}")
    for i, c in enumerate(cards):
        if logit_key in c:
            pct = round(_sigmoid((c[logit_key] + _CAL_SHIFT) / _CAL_TEMP) * 100)
            pin = c.get("_pin")
            if pin == "exact":
                pct = max(pct, 97)
            elif pin == "prefix":
                pct = max(pct, 88)
            else:
                pct = min(pct, 96)
            c["score"] = max(3, min(99, pct))
        else:
            c["score"] = _score_for_ui(i, len(cards))
        c.pop(logit_key, None)
        c.pop("_pin", None)


def hybrid_search(text: str, top_k: int = 12, prefetch: int = 60, qfilter=None) -> list[dict]:
    """
    Dense + BM25 retrieval fused with RRF, then cross-encoder reranked.
    Optional `qfilter` (from build_filter) restricts by category/rating/year.
    Returns frontend-ready card dicts.
    """
    from qdrant_client import models

    client = get_qdrant()
    dense = embed_query(text)

    prefetches = [models.Prefetch(query=dense, using=DENSE_VECTOR, limit=prefetch, filter=qfilter)]
    if ENABLE_SPARSE:
        try:
            prefetches.append(
                models.Prefetch(query=sparse_query(text), using=SPARSE_VECTOR, limit=prefetch, filter=qfilter)
            )
        except Exception as e:  # noqa: BLE001 - sparse is best-effort
            print(f"⚠️  Sparse query failed, dense-only: {e}")

    try:
        if len(prefetches) > 1:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=prefetches,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=prefetch,
                with_payload=True,
            )
        else:
            response = client.query_points(
                collection_name=COLLECTION_NAME,
                query=dense,
                using=DENSE_VECTOR,
                query_filter=qfilter,
                limit=prefetch,
                with_payload=True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"❌ Qdrant query failed: {e}")
        return []

    hits = [_hit_to_dict(p) for p in response.points]
    ranked = _rerank(text, hits)

    # Pin whole-DB exact/prefix title matches ahead of the semantic ranking.
    pinned = _title_lookup(text, limit=top_k, qfilter=qfilter)

    final, seen = [], set()
    for card in pinned + ranked:
        cid = str(card.get("id"))
        if cid in seen:
            continue
        seen.add(cid)
        final.append(card)
        if len(final) >= top_k:
            break

    _calibrate_scores(text, final)
    return final


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
        print(f"❌ Qdrant recommend failed: {e}")
        return []

    results, seen_titles = [], set()
    for p in response.points:
        if str(p.id) in exclude:
            continue
        card = _hit_to_dict(p)
        key = _normalize(card.get("title", ""))
        if key in seen_titles:  # de-dup near-identical titles
            continue
        seen_titles.add(key)
        card["score"] = _score_for_ui(len(results), top_k)
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
        print(f"❌ retrieve failed in similar_items: {e}")
        return []
    if not got:
        return []

    src = _hit_to_dict(got[0])
    src_type = str(src.get("type", "")).upper()
    src_text = f"{src.get('title','')}. {src.get('description','')}"[:512]

    cands = recommend(positive_ids=[item_id], top_k=top_k * 3, exclude_ids={item_id})
    if not cands:
        return []

    cross = _get_cross_encoder() if ENABLE_RERANK else None
    if cross is not None:
        pairs = [[src_text, f"{c.get('title','')}. {c.get('description','')}"[:512]] for c in cands]
        try:
            scores = cross.predict(pairs)
            for c, s in zip(cands, scores):
                c["_rr"] = float(s)
        except Exception:  # noqa: BLE001
            for c in cands:
                c["_rr"] = 0.0
    else:
        for c in cands:
            c["_rr"] = 0.0

    # Same-medium first (group), then by rerank score within each group.
    cands.sort(key=lambda c: (
        0 if (src_type and str(c.get("type", "")).upper() == src_type) else 1,
        -c["_rr"],
    ))

    out = cands[:top_k]
    _calibrate_scores(src_text, out, logit_key="_rr")
    return out


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
            card["score"] = _score_for_ui(len(ranked), top_k)
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


def top_rated(top_k: int = 20, qfilter=None, min_rating: float = 7.5, max_rating: float = 9.2) -> list[dict]:
    """
    Highest-rated titles (optionally filtered by category). Uses a rating window to
    avoid obscure perfect-score shorts dominating. Needs build_indexes.py first.
    """
    from qdrant_client import models

    must = [models.FieldCondition(key="rating_f", range=models.Range(gte=min_rating, lte=max_rating))]
    if qfilter is not None and getattr(qfilter, "must", None):
        must += list(qfilter.must)
    try:
        pts, _ = get_qdrant().scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(must=must),
            order_by=models.OrderBy(key="rating_f", direction=models.Direction.DESC),
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"⚠️  top_rated failed (did you run build_indexes.py?): {e}")
        return []
    out = [_hit_to_dict(p) for p in pts]
    for i, h in enumerate(out):
        h["score"] = _score_for_ui(i, len(out))
    return out


def top_popular(top_k: int = 20, qfilter=None) -> list[dict]:
    """
    Most mainstream titles = highest `votes` (vote_count). Only items harvested
    with a vote count qualify, so this naturally returns recognizable titles with
    working TMDB posters (see harvest_popular.py). Falls back to top_rated if the
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
    for i, h in enumerate(out):
        h["score"] = _score_for_ui(i, len(out))
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
