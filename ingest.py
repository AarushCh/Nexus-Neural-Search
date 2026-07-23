"""
Single, unified ingest for Nexus.

Replaces the old trio (ingest / upload_csv / seed_cloud) which each used a
different model and ID scheme. This one:

  * embeds locally with the SAME model the API queries with (bge-small-en-v1.5),
  * writes BOTH a dense vector and a BM25 sparse vector per item (hybrid search),
  * uses a stable uuid5(title) id so re-runs upsert instead of duplicating,
  * builds a full-text index on `title` for exact-match filtering.

Run:  python ingest.py            (create/upsert everything)
      python ingest.py --reset    (drop the collection first)

Needs QDRANT_URL + QDRANT_API_KEY in .env (or falls back to local qdrant_storage).
"""

import os
import sys
import uuid

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import models

from backend.engine import (
    COLLECTION_NAME,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    VECTOR_SIZE,
    embed_docs,
    get_qdrant,
    sparse_docs,
)

load_dotenv()

CSV_FILE = os.getenv("DATASET", "dataset.csv")
BATCH_SIZE = 128
RESET = "--reset" in sys.argv


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def stable_id(title: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(title).lower().strip()))


def load_dataframe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    rename = {}
    for std, cands in {
        "title": ["title", "show_title", "name", "series_title", "original_title"],
        "description": ["description", "synopsis", "plot", "summary", "overview", "desc"],
        "image": ["image", "img_url", "poster", "cover", "poster_link", "poster_path", "picture"],
        "rating": ["rating", "score", "imdb_score", "imdb_rating", "vote_average"],
        "year": ["year", "release_year", "date", "aired", "premiered"],
        "genre": ["genre", "listed_in", "category", "genres"],
        "type": ["type", "media_type", "content_type"],
    }.items():
        col = find_col(df, cands)
        if col:
            rename[col] = std
    df = df.rename(columns=rename)

    for std in ["title", "description", "image", "rating", "year", "genre", "type"]:
        if std not in df.columns:
            df[std] = ""
    return df.fillna("")


def refine_type(row) -> str:
    genre = str(row["genre"]).lower()
    base = str(row["type"]).title() or "Movie"
    if "documentary" in genre or "docu" in genre:
        return "Documentary"
    if "anime" in genre:
        return "Anime"
    if "stand-up" in genre:
        return "Stand-Up"
    return base


def main():
    if not os.path.exists(CSV_FILE):
        print(f"❌ {CSV_FILE} not found.")
        sys.exit(1)

    print(f"📊 Reading {CSV_FILE} ...")
    df = load_dataframe(CSV_FILE)
    print(f"✅ {len(df)} rows loaded.")

    client = get_qdrant()

    if RESET and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Dropped existing collection.")

    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={DENSE_VECTOR: models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)},
        )
        # Full-text index enables fast exact/keyword title filtering.
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="title",
            field_schema=models.TextIndexParams(type=models.TextIndexType.TEXT, lowercase=True),
        )
        print("✨ Created collection (dense + bm25 + title index).")

    total = 0
    for start in range(0, len(df), BATCH_SIZE):
        chunk = df.iloc[start:start + BATCH_SIZE]
        payloads, texts = [], []
        for _, row in chunk.iterrows():
            rtype = refine_type(row)
            payload = {
                "title": str(row["title"]),
                "description": str(row["description"]),
                "image": str(row["image"]),
                "rating": row["rating"],
                "year": str(row["year"]),
                "genre": str(row["genre"]),
                "type": rtype,
            }
            payloads.append(payload)
            # Rich text for embeddings: title carries most weight, then desc/genre/type.
            texts.append(f"{payload['title']}. {payload['description']} {payload['genre']} {rtype}")

        dense_vecs = embed_docs(texts)
        sparse_vecs = sparse_docs(texts)

        points = [
            models.PointStruct(
                id=stable_id(p["title"]),
                vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv},
                payload=p,
            )
            for p, dv, sv in zip(payloads, dense_vecs, sparse_vecs)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"   🚀 {total}/{len(df)} upserted")

    print(f"🎉 Done. {total} items in '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
