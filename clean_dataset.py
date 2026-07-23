"""
Clean corrupted/duplicated descriptions from the dataset.

Some earlier enrichment mis-copied a single description onto many unrelated
titles (e.g. one "Music video ... ending theme for Cyberpunk: Edgerunners"
blurb landed on dozens of shows). Identical text -> identical embeddings, which
makes those rows false nearest-neighbours of each other and of whatever their
text happens to mention. That pollutes search follow-ups and "similar" results.

This blanks any description shared by >= THRESHOLD distinct titles, so each such
row re-embeds from its unique title/genre/type instead.

Usage:
    python clean_dataset.py            # fix dataset.csv only
    python clean_dataset.py --push     # also re-embed+upsert affected rows to Qdrant
"""

import html
import sys

import pandas as pd
from qdrant_client import models

from backend.engine import (
    COLLECTION_NAME,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    embed_docs,
    get_qdrant,
    sparse_docs,
)
from ingest import load_dataframe, refine_type, stable_id

CSV = "dataset.csv"
THRESHOLD = 3          # a description shared by >=3 distinct titles is treated as corrupt
PUSH = "--push" in sys.argv


def main():
    raw = pd.read_csv(CSV, dtype=str, keep_default_na=False)
    norm = raw["description"].map(lambda s: html.unescape(str(s)).strip())

    per_desc_titles = raw.assign(_d=norm).groupby("_d")["title"].nunique()
    bad = set(per_desc_titles[per_desc_titles >= THRESHOLD].index)
    bad.discard("")

    mask = norm.isin(bad)
    affected_titles = set(raw.loc[mask, "title"])
    print(f"🧹 {int(mask.sum())} rows across {len(bad)} duplicated descriptions -> blanking")

    raw.loc[mask, "description"] = ""
    raw.to_csv(CSV, index=False, lineterminator="\n", encoding="utf-8")
    print(f"💾 Saved cleaned {CSV}")

    if not PUSH:
        print("ℹ️  Skipped Qdrant push (use --push). Re-run ingest.py to apply, or push now.")
        return
    if not affected_titles:
        print("✅ Nothing to push.")
        return

    df = load_dataframe(CSV)
    subset = df[df["title"].isin(affected_titles)].reset_index(drop=True)
    print(f"🚀 Re-embedding {len(subset)} affected rows into Qdrant...")

    client = get_qdrant()
    B = 128
    done = 0
    for start in range(0, len(subset), B):
        chunk = subset.iloc[start:start + B]
        payloads, texts = [], []
        for _, row in chunk.iterrows():
            rtype = refine_type(row)
            p = {
                "title": str(row["title"]),
                "description": str(row["description"]),
                "image": str(row["image"]),
                "rating": row["rating"],
                "year": str(row["year"]),
                "genre": str(row["genre"]),
                "type": rtype,
            }
            payloads.append(p)
            texts.append(f"{p['title']}. {p['description']} {p['genre']} {rtype}")

        dense = embed_docs(texts)
        sparse = sparse_docs(texts)
        points = [
            models.PointStruct(
                id=stable_id(p["title"]),
                vector={DENSE_VECTOR: dv, SPARSE_VECTOR: sv},
                payload=p,
            )
            for p, dv, sv in zip(payloads, dense, sparse)
        ]
        client.upsert(COLLECTION_NAME, points=points)
        done += len(points)
        print(f"   {done}/{len(subset)} re-upserted")

    print("✅ Cluster updated.")


if __name__ == "__main__":
    main()
