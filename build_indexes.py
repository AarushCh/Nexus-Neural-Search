"""
Add numeric/keyword payload fields + indexes so the API can do server-side
filtering (category, rating, year) and Top-Rated ordering.

For every point it derives:
  rating_f (float)   from `rating`
  year_i   (int)     from `year`
  category (keyword) MOVIE / TV / ANIME / DOCUMENTARY  from `type`

Then creates payload indexes on those fields. Idempotent; safe to re-run.

Usage:  python build_indexes.py
"""

import re

from qdrant_client import models

from backend.engine import COLLECTION_NAME, get_qdrant


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def to_year(v):
    m = re.search(r"(19|20)\d{2}", str(v))
    return int(m.group()) if m else 0


def category_of(type_str):
    t = str(type_str).upper()
    if "ANIME" in t:
        return "ANIME"
    if "DOC" in t:
        return "DOCUMENTARY"
    if "TV" in t:
        return "TV"
    return "MOVIE"


def main():
    client = get_qdrant()

    print("🏗️  Creating payload indexes...")
    for field, schema in [
        ("rating_f", models.PayloadSchemaType.FLOAT),
        ("year_i", models.PayloadSchemaType.INTEGER),
        ("category", models.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception as e:  # already exists / racing
            print(f"   ({field}: {e})")

    print("🔁 Backfilling fields for all points...")
    updated = 0
    offset = None
    while True:
        points, offset = client.scroll(
            COLLECTION_NAME, limit=512, offset=offset, with_payload=True, with_vectors=False
        )
        if not points:
            break
        # Batch all per-point payload updates into ONE request (avoids 27k round-trips).
        ops = [
            models.SetPayloadOperation(
                set_payload=models.SetPayload(
                    payload={
                        "rating_f": to_float((p.payload or {}).get("rating")),
                        "year_i": to_year((p.payload or {}).get("year")),
                        "category": category_of((p.payload or {}).get("type")),
                    },
                    points=[p.id],
                )
            )
            for p in points
        ]
        client.batch_update_points(COLLECTION_NAME, update_operations=ops)
        updated += len(points)
        print(f"   {updated} updated")
        if offset is None:
            break

    print(f"✅ Done. {updated} points now filterable by category / rating_f / year_i.")


if __name__ == "__main__":
    main()
