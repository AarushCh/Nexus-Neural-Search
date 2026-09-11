"""
Measure real bytes-per-title so catalogue size is a decision, not a guess.

    python bench_storage.py [rows]          # default 5000

Loads synthetic rows whose field sizes match real TMDB data (overview length,
tag counts, cast size), builds every index, then reports actual on-disk usage
and projects it to 100k.

Why not just estimate: below a few thousand rows the fixed costs — 8KB pages,
minimum index sizes, TOAST tables — dominate completely. At 7 rows this schema
reports 43KB per title, which is off by more than an order of magnitude.

Run it against a scratch database, NOT the live one: it inserts and deletes
bench rows.
"""

import math
import os
import random
import sys

if not os.getenv("DATABASE_URL", "").startswith("postgres"):
    sys.exit("Set DATABASE_URL to a Postgres instance with pgvector.")

from backend import store  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
DIM = store.VECTOR_SIZE
rnd = random.Random(42)

WORDS = ("dystopia neo noir replicant memory desert spice cyberpunk cyborg identity "
         "consciousness mercenary tragedy heartwarming family comedy thriller haunting "
         "revenge slow burn found family coming of age isolation grief hope").split()
NAMES = ("Ryan Gosling,Ana de Armas,Harrison Ford,Denis Villeneuve,Roger Deakins,"
         "Hans Zimmer,Jared Leto,Robin Wright,Sylvia Hoeks,Dave Bautista").split(",")


def _text(n_words: int) -> str:
    return " ".join(rnd.choice(WORDS) for _ in range(n_words))


def _vec() -> list:
    v = [rnd.gauss(0, 1) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def make_row(i: int) -> dict:
    """Field sizes matched to real TMDB records."""
    return {
        "id": f"bench-{i}",
        "tmdb_id": i, "tmdb_kind": "movie" if i % 3 else "tv",
        "imdb_id": f"tt{1000000 + i}",
        "title": f"Benchmark Title Number {i}",
        "original_title": f"Benchmark Original Title {i}",
        # TMDB overviews average roughly 350-500 characters.
        "description": _text(70),
        "tagline": _text(7),
        "image": f"https://image.tmdb.org/t/p/w500/{i:08d}abcdefgh.jpg",
        "image_sm": f"https://image.tmdb.org/t/p/w342/{i:08d}abcdefgh.jpg",
        "backdrop": f"https://image.tmdb.org/t/p/w1280/{i:08d}ijklmnop.jpg",
        "category": rnd.choice(["MOVIE", "TV", "ANIME", "DOCUMENTARY"]),
        "types": ["film"], "forms": ["live-action"],
        "genres": rnd.sample(["action", "drama", "sci-fi", "horror", "comedy"], 3),
        # A well-tagged title carries 20-30 namespaced tags.
        "tags": [f"theme:{rnd.choice(WORDS)}-{rnd.randint(1, 400)}" for _ in range(24)],
        "rating": round(rnd.uniform(4, 9), 1), "votes": rnd.randint(10, 900_000),
        "year_i": rnd.randint(1950, 2026), "runtime": rnd.randint(60, 180),
        "seasons": None, "episodes": None, "status": "Released",
        "certification": "R", "original_language": "en",
        "release_date": "2017-10-04", "trailer_key": "gCcx85zbxz4",
        # media_extra: 15 cast, 8 crew, 10 providers.
        "cast": [{"name": rnd.choice(NAMES), "character": _text(2),
                  "profile": f"https://image.tmdb.org/t/p/w185/{j:08d}.jpg"}
                 for j in range(15)],
        "crew": [{"name": rnd.choice(NAMES), "job": "Director"} for _ in range(8)],
        "providers": [{"name": f"Provider {j}",
                       "logo": f"https://image.tmdb.org/t/p/w92/{j:04d}.jpg"}
                      for j in range(10)],
        "alt_titles": [f"Alt Title {j} for {i}" for j in range(5)],
        "imdb_rating": 7.5, "imdb_votes": 50_000, "anilist_id": None,
    }


def main() -> None:
    print(f"Loading {N} synthetic titles with realistic field sizes…")
    store.create_schema()
    with store.engine.begin() as cx:
        from sqlalchemy import text
        cx.execute(text("DELETE FROM media WHERE id LIKE 'bench-%'"))

    B = 500
    for start in range(0, N, B):
        rows = [make_row(i) for i in range(start, min(start + B, N))]
        docs = [f"{r['title']} {r['description']} {' '.join(r['tags'])} "
                f"{' '.join(c['name'] for c in r['cast'])}" for r in rows]
        store.upsert(rows, docs, [_vec() for _ in rows])
        store.upsert_extra(rows)
        print(f"   {min(start + B, N)}/{N}")

    print("Building indexes (HNSW is the slow one)…")
    store.create_indexes()
    store.analyze()

    rep = store.storage_report()
    print("\n" + "=" * 58)
    print(f"  titles measured    {rep['titles']}")
    print(f"  media table        {rep['media_mb']} MB")
    print(f"  media_extra        {rep['extra_mb']} MB")
    print(f"  total              {rep['total_mb']} MB")
    print(f"  per title          {rep['bytes_per_title'] / 1024:.1f} KB")
    print(f"  reliable sample    {rep['reliable']}")
    print("-" * 58)
    for n in (25_000, 50_000, 100_000, 150_000):
        mb = rep["bytes_per_title"] * n / 1e6
        fits = "fits 0.5GB free" if mb < 480 else "needs a paid tier"
        print(f"  {n:>7,} titles  ->  {mb:>7.0f} MB   {fits}")
    print("=" * 58)
    print("\nRemove the bench rows with:")
    print("  DELETE FROM media WHERE id LIKE 'bench-%';")


if __name__ == "__main__":
    main()
