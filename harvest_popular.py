"""
Harvest MAINSTREAM titles from TMDB, ranked by vote_count (recognizability, not
recency). vote_count.desc surfaces the genuinely well-known — Breaking Bad, The
Godfather, Attack on Titan — instead of rating-only obscurities or currently-airing
K/C-dramas that spike on `popularity`.

Stores `votes` + `pop` + `category` in the Qdrant payload (with working
image.tmdb.org posters) and upserts. Also appends genuinely-new titles to
dataset.csv. Creates a `votes` index so the feed can order by it.

Usage:
    python harvest_popular.py            # movies/TV/anime/docs, push to Qdrant
    python harvest_popular.py --pages 40 # deeper pull
"""

import os
import re
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv
from qdrant_client import models

from backend.engine import (
    COLLECTION_NAME, DENSE_VECTOR, SPARSE_VECTOR,
    embed_docs, get_qdrant, sparse_docs,
)
from ingest import stable_id

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
IMG = "https://image.tmdb.org/t/p/w500"
BASE = "https://api.themoviedb.org/3"
CSV_FILE = "dataset.csv"

PAGES = 30
if "--pages" in sys.argv:
    PAGES = int(sys.argv[sys.argv.index("--pages") + 1])

# (discover path, extra params, category, pages) — all sorted by vote_count.desc
STREAMS = [
    ("discover/movie", {"without_genres": "99"}, "MOVIE", PAGES),
    ("discover/tv", {"without_genres": "16,99"}, "TV", PAGES),
    ("discover/tv", {"with_genres": "16", "with_original_language": "ja"}, "ANIME", PAGES),
    ("discover/movie", {"with_genres": "16", "with_original_language": "ja"}, "ANIME", max(15, PAGES // 2)),
    ("discover/movie", {"with_genres": "99"}, "DOCUMENTARY", max(15, PAGES // 2)),
    ("discover/tv", {"with_genres": "99"}, "DOCUMENTARY", max(10, PAGES // 3)),
]


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def genres_map():
    g = {}
    for kind in ("movie", "tv"):
        try:
            r = requests.get(f"{BASE}/genre/{kind}/list", params={"api_key": API_KEY}, timeout=15)
            for x in r.json().get("genres", []):
                g[x["id"]] = x["name"]
        except Exception:
            pass
    return g


def main():
    if not API_KEY:
        print("❌ TMDB_API_KEY missing")
        sys.exit(1)

    client = get_qdrant()
    try:
        client.create_payload_index(COLLECTION_NAME, field_name="votes", field_schema=models.PayloadSchemaType.INTEGER)
    except Exception:
        pass

    gmap = genres_map()
    existing_csv = set()
    if os.path.exists(CSV_FILE):
        df0 = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
        existing_csv = {norm(t) for t in df0["title"]}

    rows, seen = [], set()
    for path, params, category, pages in STREAMS:
        before = len(rows)
        for page in range(1, pages + 1):
            q = {"api_key": API_KEY, "sort_by": "vote_count.desc", "vote_count.gte": 200,
                 "page": page, **params}
            try:
                r = requests.get(f"{BASE}/{path}", params=q, timeout=15)
                if r.status_code == 429:
                    time.sleep(2); continue
                if r.status_code != 200:
                    break
                results = r.json().get("results", [])
            except Exception:
                continue
            if not results:
                break
            for it in results:
                title = it.get("title") or it.get("name") or ""
                key = norm(title)
                poster = it.get("poster_path")
                overview = (it.get("overview") or "").strip()
                if not title or not poster or len(overview) < 20 or key in seen:
                    continue
                seen.add(key)
                date = it.get("release_date") or it.get("first_air_date") or ""
                gs = ", ".join(gmap.get(g, "") for g in it.get("genre_ids", []) if g in gmap)
                rows.append({
                    "title": title,
                    "description": overview if not gs else f"{overview} Genres: {gs}.",
                    "image": f"{IMG}{poster}",
                    "category": category,
                    "type": category.title(),
                    "rating": float(it.get("vote_average", 0) or 0),
                    "year": (date[:4] if date else ""),
                    "votes": int(it.get("vote_count", 0) or 0),
                    "pop": float(it.get("popularity", 0) or 0),
                })
            time.sleep(0.04)
        print(f"   {path} [{category}] +{len(rows) - before}")

    if not rows:
        print("No rows."); return
    print(f"📥 {len(rows)} mainstream titles harvested. Embedding + upserting...")

    B = 128
    done = 0
    new_for_csv = []
    for start in range(0, len(rows), B):
        chunk = rows[start:start + B]
        texts = [f"{c['title']}. {c['description']} {c['category']}" for c in chunk]
        dense = embed_docs(texts)
        sparse = sparse_docs(texts)
        points = []
        for c, dv, sv in zip(chunk, dense, sparse):
            payload = {
                "title": c["title"], "description": c["description"], "image": c["image"],
                "type": c["type"], "category": c["category"], "genre": "",
                "rating": c["rating"], "rating_f": c["rating"],
                "year": c["year"], "year_i": int(c["year"]) if c["year"].isdigit() else 0,
                "votes": c["votes"], "pop": c["pop"],
            }
            points.append(models.PointStruct(id=stable_id(c["title"]), vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv}, payload=payload))
            if norm(c["title"]) not in existing_csv:
                new_for_csv.append({k: c[k] for k in ["title", "description", "image", "type", "rating", "year"]})
        client.upsert(COLLECTION_NAME, points=points)
        done += len(points)
        print(f"   {done}/{len(rows)} upserted")

    if new_for_csv and os.path.exists(CSV_FILE):
        df0 = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
        add = pd.DataFrame(new_for_csv)[["title", "description", "image", "type", "rating", "year"]]
        pd.concat([df0, add], ignore_index=True).to_csv(CSV_FILE, index=False, lineterminator="\n", encoding="utf-8")
        print(f"📝 Appended {len(add)} new titles to {CSV_FILE}")

    print("✅ Done. Feed can now rank by `votes` (mainstream).")


if __name__ == "__main__":
    main()
