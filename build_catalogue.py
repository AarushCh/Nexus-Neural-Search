"""
build_catalogue.py — the authoritative Nexus catalogue builder.

Harvests mainstream titles from TMDB at scale and VALIDATES + ENRICHES every one
in a single pass: correct poster + backdrop, real overview/rating/year/genres,
runtime, YouTube trailer key, top cast, and streaming providers — all written
straight into the Qdrant payload. TMDB becomes the single source of truth, so
every card is correct by construction (no more Another-Life/Re:Zero poster
mixups, no missing Edgerunners art) and the detail view is instant (no per-open
TMDB call).

Usage:
  python build_catalogue.py                 # deep harvest, upsert into Qdrant
  python build_catalogue.py --pages 150     # go wider (20 titles/page/stream)
  python build_catalogue.py --recreate      # wipe + rebuild the collection clean
  python build_catalogue.py --workers 24    # enrichment concurrency
  python build_catalogue.py --min-votes 80  # include more mid-popularity titles

Env: TMDB_API_KEY, QDRANT_URL, QDRANT_API_KEY (.env).
"""
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from qdrant_client import models

from backend.engine import (
    COLLECTION_NAME, DENSE_VECTOR, SPARSE_VECTOR, VECTOR_SIZE,
    embed_docs, get_qdrant, sparse_docs,
)
from ingest import stable_id

try:  # Windows consoles default to cp1252 and choke on emoji in progress logs.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

load_dotenv()
API_KEY = os.getenv("TMDB_API_KEY")
BASE = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p"


def _arg(flag, default):
    return int(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


PAGES = _arg("--pages", 80)
WORKERS = _arg("--workers", 20)
MIN_VOTES = _arg("--min-votes", 120)
RECREATE = "--recreate" in sys.argv

# (discover path, extra params, category, pages) — all sorted by vote_count.desc.
# Anime streams run FIRST and match by genre 16 + Japanese origin_country (not
# original_language) so Japanese-studio productions Netflix tags as English —
# Cyberpunk: Edgerunners, Arcane-style co-productions — are captured as ANIME
# instead of falling through both the TV net (genre 16 excluded) and a ja-only
# language filter. Running first lets anime claim genre-16 titles before the
# movie/TV streams, keeping category labels deterministic.
STREAMS = [
    ("discover/tv", {"with_genres": "16", "with_origin_country": "JP"}, "ANIME", PAGES),
    ("discover/movie", {"with_genres": "16", "with_origin_country": "JP"}, "ANIME", max(40, PAGES // 2)),
    ("discover/movie", {"without_genres": "99"}, "MOVIE", PAGES),
    ("discover/tv", {"without_genres": "16,99"}, "TV", PAGES),
    ("discover/movie", {"with_genres": "99"}, "DOCUMENTARY", max(20, PAGES // 2)),
    ("discover/tv", {"with_genres": "99"}, "DOCUMENTARY", max(15, PAGES // 3)),
]

_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
    return s


def _get(path, params, tries=3):
    for _ in range(tries):
        try:
            r = _session().get(f"{BASE}/{path}", params=params, timeout=15)
        except Exception:
            time.sleep(1)
            continue
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2)))
            continue
        return r
    return None


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def genres_map():
    g = {}
    for kind in ("movie", "tv"):
        r = _get(f"genre/{kind}/list", {"api_key": API_KEY})
        if r and r.status_code == 200:
            for x in r.json().get("genres", []):
                g[x["id"]] = x["name"]
    return g


def discover():
    """Collect unique (kind, id) -> (category, genre_ids) across every stream."""
    found = {}
    for path, params, category, pages in STREAMS:
        kind = "movie" if "movie" in path else "tv"
        got = 0
        for page in range(1, pages + 1):
            r = _get(path, {"api_key": API_KEY, "sort_by": "vote_count.desc",
                            "vote_count.gte": MIN_VOTES, "page": page, **params})
            if not r or r.status_code != 200:
                break
            results = r.json().get("results", [])
            if not results:
                break
            for it in results:
                k = (kind, it["id"])
                if k not in found:
                    found[k] = (category, it.get("genre_ids", []))
                    got += 1
            time.sleep(0.02)
        print(f"   {path} [{category}] +{got} (total {len(found)})")
    return found


def _trailer_key(videos):
    vids = videos.get("results", [])
    # official YouTube Trailer -> any Trailer -> Teaser
    for typ, need_official in (("Trailer", True), ("Trailer", False), ("Teaser", False)):
        for v in vids:
            if v.get("site") == "YouTube" and v.get("type") == typ and (v.get("official") or not need_official):
                return v.get("key")
    return None


def enrich(kind, tid, category, genre_ids, gmap):
    r = _get(f"{kind}/{tid}", {"api_key": API_KEY,
                               "append_to_response": "videos,credits,watch/providers"})
    if not r or r.status_code != 200:
        return None
    d = r.json()

    title = d.get("title") or d.get("name") or ""
    overview = (d.get("overview") or "").strip()
    poster = d.get("poster_path")
    if not title or not poster or len(overview) < 20:  # validation gate
        return None

    date = d.get("release_date") or d.get("first_air_date") or ""
    year = date[:4] if date else ""
    gs = ", ".join(x["name"] for x in d.get("genres", []) if x.get("name")) or \
        ", ".join(gmap.get(g, "") for g in genre_ids if g in gmap)
    cast = [{"name": c.get("name"), "character": c.get("character"),
             "profile": f"{IMG}/w185{c['profile_path']}" if c.get("profile_path") else None}
            for c in d.get("credits", {}).get("cast", [])[:10]]
    prov, seen = [], set()
    us = d.get("watch/providers", {}).get("results", {}).get("US", {})
    for bucket in ("flatrate", "free", "ads", "rent", "buy"):
        for p in us.get(bucket, []):
            n = p.get("provider_name")
            if n and n not in seen:
                seen.add(n)
                prov.append({"name": n, "logo": f"{IMG}/w92{p['logo_path']}" if p.get("logo_path") else None})
    bd = d.get("backdrop_path")
    rt = d.get("runtime") or (d.get("episode_run_time") or [None])[0]
    rating = round(float(d.get("vote_average") or 0), 1)
    return {
        "title": title,
        "description": overview if not gs else f"{overview} Genres: {gs}.",
        "image": f"{IMG}/w500{poster}",
        "backdrop": f"{IMG}/w780{bd}" if bd else None,
        "type": category.title(), "category": category, "genre": gs,
        "rating": rating, "rating_f": rating,
        "year": year, "year_i": int(year) if year.isdigit() else 0,
        "votes": int(d.get("vote_count") or 0), "pop": float(d.get("popularity") or 0),
        "runtime": rt, "trailer_key": _trailer_key(d.get("videos", {})),
        "cast": cast, "providers": prov,
        "tmdb_id": tid, "tmdb_kind": kind,
    }


def ensure_collection(client):
    exists = client.collection_exists(COLLECTION_NAME)
    if RECREATE and exists:
        client.delete_collection(COLLECTION_NAME)
        exists = False
    if not exists:
        client.create_collection(
            COLLECTION_NAME,
            vectors_config={DENSE_VECTOR: models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
    for field, schema in [("title", models.PayloadSchemaType.TEXT),
                          ("category", models.PayloadSchemaType.KEYWORD),
                          ("rating_f", models.PayloadSchemaType.FLOAT),
                          ("year_i", models.PayloadSchemaType.INTEGER),
                          ("votes", models.PayloadSchemaType.INTEGER)]:
        try:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:
            pass


def main():
    if not API_KEY:
        print("❌ TMDB_API_KEY missing"); sys.exit(1)

    client = get_qdrant()
    ensure_collection(client)
    gmap = genres_map()

    print(f"🔎 Discovering across {len(STREAMS)} streams x up to {PAGES} pages...")
    found = discover()
    if not found:
        print("No titles discovered."); return
    print(f"📥 {len(found)} unique titles. Enriching with {WORKERS} workers...")

    rows, done, kept = [], 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(enrich, k[0], k[1], v[0], v[1], gmap) for k, v in found.items()]
        for f in as_completed(futs):
            done += 1
            r = f.result()
            if r:
                rows.append(r); kept += 1
            if done % 500 == 0:
                print(f"   enriched {done}/{len(found)} (kept {kept})")

    # Dedup by normalized title, keeping the highest-voted version.
    best = {}
    for r in rows:
        k = norm(r["title"])
        if k and (k not in best or r["votes"] > best[k]["votes"]):
            best[k] = r
    rows = list(best.values())
    trailers = sum(1 for r in rows if r["trailer_key"])
    print(f"✅ {len(rows)} validated titles ({trailers} with trailers). Embedding + upserting...")

    B, up = 128, 0
    for start in range(0, len(rows), B):
        chunk = rows[start:start + B]
        texts = [f"{c['title']}. {c['description']} {c['category']}" for c in chunk]
        dense = embed_docs(texts)
        sparse = sparse_docs(texts)
        pts = [models.PointStruct(id=stable_id(c["title"]),
                                  vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv}, payload=c)
               for c, dv, sv in zip(chunk, dense, sparse)]
        client.upsert(COLLECTION_NAME, points=pts)
        up += len(pts)
        print(f"   {up}/{len(rows)} upserted")

    print("✅ Catalogue rebuilt. Search, feeds, and detail pages now serve validated TMDB data.")


if __name__ == "__main__":
    main()
