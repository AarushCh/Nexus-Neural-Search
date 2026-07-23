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

# Feature flags (let a low-RAM host such as Render free tier disable the reranker).
ENABLE_RERANK = os.getenv("ENABLE_RERANK", "true").lower() == "true"
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
                from sentence_transformers import SentenceTransformer

                _dense_model = SentenceTransformer(DENSE_MODEL)
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
    return _get_dense().encode(QUERY_INSTRUCTION + text, normalize_embeddings=True).tolist()


def embed_docs(texts: list[str]) -> list[list[float]]:
    vecs = _get_dense().encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    return [v.tolist() for v in vecs]


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


def _title_lookup(query: str, limit: int = 8) -> list[dict]:
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
    try:
        points, _ = get_qdrant().scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="title", match=models.MatchText(text=query))]
            ),
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
            exact.append(card)
        elif nt.startswith(nq + " ") or nt.startswith(nq):
            prefix.append(card)
    exact.sort(key=_rating, reverse=True)
    prefix.sort(key=_rating, reverse=True)
    return (exact + prefix)[:limit]


def _score_for_ui(rank: int, total: int) -> int:
    """Map final rank -> a 60-99 "% MATCH" the frontend badge expects."""
    if total <= 1:
        return 99
    return int(99 - (rank / max(total - 1, 1)) * 39)


def hybrid_search(text: str, top_k: int = 12, prefetch: int = 60) -> list[dict]:
    """
    Dense + BM25 retrieval fused with RRF, then cross-encoder reranked.
    Returns frontend-ready card dicts.
    """
    from qdrant_client import models

    client = get_qdrant()
    dense = embed_query(text)

    prefetches = [models.Prefetch(query=dense, using=DENSE_VECTOR, limit=prefetch)]
    if ENABLE_SPARSE:
        try:
            prefetches.append(
                models.Prefetch(query=sparse_query(text), using=SPARSE_VECTOR, limit=prefetch)
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
                limit=prefetch,
                with_payload=True,
            )
    except Exception as e:  # noqa: BLE001
        print(f"❌ Qdrant query failed: {e}")
        return []

    hits = [_hit_to_dict(p) for p in response.points]
    ranked = _rerank(text, hits)

    # Pin whole-DB exact/prefix title matches ahead of the semantic ranking.
    pinned = _title_lookup(text, limit=top_k)

    final, seen = [], set()
    for card in pinned + ranked:
        cid = str(card.get("id"))
        if cid in seen:
            continue
        seen.add(cid)
        final.append(card)
        if len(final) >= top_k:
            break

    for i, h in enumerate(final):
        h["score"] = _score_for_ui(i, len(final))
        h.pop("rerank", None)
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
    for i, c in enumerate(out):
        c["score"] = _score_for_ui(i, len(out))
        c.pop("_rr", None)
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

    catalogue = [
        {"i": i, "title": h.get("title", ""), "type": h.get("type", ""),
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
