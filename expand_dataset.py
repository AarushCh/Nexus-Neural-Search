"""
Expand dataset.csv with lots of mainstream titles from TMDB.

Harvests several streams so the catalogue spans real categories:
  * Popular + top-rated MOVIES
  * Popular + top-rated TV
  * DOCUMENTARIES (movie + tv, genre 99)
  * ANIME (Japanese animation, genre 16 + original_language=ja)
  * Trending (this week, mixed)

Every row is normalized to the existing schema (title, description, image,
type, rating, year), typed as MOVIE / TV / DOCUMENTARY / ANIME, de-duplicated
against what's already in dataset.csv, and appended. TMDB posters are reliable.

Usage:
    python expand_dataset.py                # append to dataset.csv
    python expand_dataset.py --push         # also embed+upsert the NEW rows to Qdrant
    python expand_dataset.py --scale 2      # multiply all page counts (bigger pull)

Needs TMDB_API_KEY in .env.
"""

import os
import re
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TMDB_API_KEY")
CSV_FILE = "dataset.csv"
BACKUP_FILE = "dataset_backup.csv"
IMG_BASE = "https://image.tmdb.org/t/p/w500"
BASE = "https://api.themoviedb.org/3"

PUSH = "--push" in sys.argv
SCALE = 1.0
if "--scale" in sys.argv:
    SCALE = float(sys.argv[sys.argv.index("--scale") + 1])

# (endpoint, params, type_label, pages)  -- type_label None => derive from media_type
STREAMS = [
    ("movie/popular", {}, "MOVIE", 100),
    ("movie/top_rated", {}, "MOVIE", 80),
    ("tv/popular", {}, "TV", 100),
    ("tv/top_rated", {}, "TV", 80),
    ("discover/movie", {"with_genres": "99", "sort_by": "popularity.desc", "vote_count.gte": 30}, "DOCUMENTARY", 60),
    ("discover/tv", {"with_genres": "99", "sort_by": "popularity.desc", "vote_count.gte": 15}, "DOCUMENTARY", 40),
    ("discover/tv", {"with_genres": "16", "with_original_language": "ja", "sort_by": "popularity.desc", "vote_count.gte": 15}, "ANIME", 60),
    ("discover/movie", {"with_genres": "16", "with_original_language": "ja", "sort_by": "popularity.desc", "vote_count.gte": 15}, "ANIME", 40),
    ("trending/all/week", {}, None, 10),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def fetch_genres():
    genres = {}
    for kind in ("movie", "tv"):
        try:
            r = requests.get(f"{BASE}/genre/{kind}/list", params={"api_key": API_KEY}, timeout=15)
            for g in r.json().get("genres", []):
                genres[g["id"]] = g["name"]
        except Exception:
            pass
    return genres


def normalize_item(it, type_label, genre_map):
    """Return a normalized row dict, or None if too low-quality."""
    media = it.get("media_type")  # only present on trending
    if type_label is None:
        if media == "movie":
            type_label = "MOVIE"
        elif media == "tv":
            type_label = "TV"
        else:
            return None  # skip persons etc.

    title = it.get("title") or it.get("name") or ""
    poster = it.get("poster_path")
    overview = (it.get("overview") or "").strip()
    if not title or not poster or len(overview) < 20:
        return None

    date = it.get("release_date") or it.get("first_air_date") or ""
    year = date[:4] if date else ""
    genres = ", ".join(genre_map.get(g, "") for g in it.get("genre_ids", []) if g in genre_map)
    return {
        "title": title,
        "description": overview if not genres else f"{overview} Genres: {genres}.",
        "image": f"{IMG_BASE}{poster}",
        "type": type_label,
        "rating": it.get("vote_average", 0) or 0,
        "year": year,
    }


def main():
    if not API_KEY:
        print("❌ TMDB_API_KEY missing in .env")
        sys.exit(1)
    if not os.path.exists(CSV_FILE):
        print(f"❌ {CSV_FILE} not found")
        sys.exit(1)

    df = pd.read_csv(CSV_FILE, dtype=str, keep_default_na=False)
    existing = {norm(t) for t in df["title"]}
    print(f"📊 Existing: {len(df)} rows ({len(existing)} unique titles)")

    genre_map = fetch_genres()
    print(f"🎭 {len(genre_map)} genre labels loaded")

    new_rows, seen_new = [], set()

    for path, params, type_label, pages in STREAMS:
        pages = max(1, int(pages * SCALE))
        added_before = len(new_rows)
        for page in range(1, pages + 1):
            q = {"api_key": API_KEY, "page": page, **params}
            try:
                r = requests.get(f"{BASE}/{path}", params=q, timeout=15)
                if r.status_code == 429:
                    time.sleep(2)
                    continue
                if r.status_code != 200:
                    break
                results = r.json().get("results", [])
            except Exception:
                continue
            if not results:
                break

            for it in results:
                row = normalize_item(it, type_label, genre_map)
                if not row:
                    continue
                key = norm(row["title"])
                if key in existing or key in seen_new:
                    continue
                seen_new.add(key)
                new_rows.append(row)
            time.sleep(0.04)
        print(f"   {path} [{type_label or 'trending'}] -> +{len(new_rows) - added_before} (total new {len(new_rows)})")

    if not new_rows:
        print("ℹ️ No new titles (all duplicates).")
        return

    print(f"💾 Backing up -> {BACKUP_FILE}")
    df.to_csv(BACKUP_FILE, index=False)

    add_df = pd.DataFrame(new_rows)[["title", "description", "image", "type", "rating", "year"]]
    combined = pd.concat([df, add_df], ignore_index=True)
    combined.to_csv(CSV_FILE, index=False, lineterminator="\n", encoding="utf-8")
    print(f"✅ Added {len(add_df)} titles. dataset.csv now {len(combined)} rows.")
    print("   New by type:")
    print(add_df["type"].value_counts().to_string())

    if PUSH:
        push_new_rows(add_df)


def push_new_rows(add_df):
    """Embed + upsert only the new rows into Qdrant (no full re-ingest)."""
    from qdrant_client import models

    from backend.engine import (
        COLLECTION_NAME, DENSE_VECTOR, SPARSE_VECTOR,
        embed_docs, get_qdrant, sparse_docs,
    )
    from ingest import stable_id

    client = get_qdrant()
    print(f"🚀 Pushing {len(add_df)} new rows to Qdrant...")
    B = 128
    done = 0
    for start in range(0, len(add_df), B):
        chunk = add_df.iloc[start:start + B]
        payloads, texts = [], []
        for _, row in chunk.iterrows():
            rtype = str(row["type"]).title()
            p = {
                "title": str(row["title"]),
                "description": str(row["description"]),
                "image": str(row["image"]),
                "rating": row["rating"],
                "year": str(row["year"]),
                "genre": "",
                "type": rtype,
            }
            payloads.append(p)
            texts.append(f"{p['title']}. {p['description']} {rtype}")

        dense = embed_docs(texts)
        sparse = sparse_docs(texts)
        points = [
            models.PointStruct(id=stable_id(p["title"]), vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv}, payload=p)
            for p, dv, sv in zip(payloads, dense, sparse)
        ]
        client.upsert(COLLECTION_NAME, points=points)
        done += len(points)
        print(f"   {done}/{len(add_df)} upserted")
    print("✅ Cluster updated with new titles.")


if __name__ == "__main__":
    main()
