"""
Smoke + regression tests for the Nexus engine.

Plain asserts, no framework:  python test_nexus.py

The pure-logic tests run anywhere. The live tests only run when DATABASE_URL points
at Postgres, and are skipped otherwise so this stays useful in CI without secrets.
"""

import os
import sys

from backend.engine import _normalize, _score_cards, collection_health
from catalogue.schema import dedupe_key, stable_id


def test_stable_id_distinguishes_same_title():
    """Regression: ids used to be uuid5(title), so remakes collapsed onto one point."""
    dune_1984 = stable_id("movie", 841)
    dune_2021 = stable_id("movie", 438631)
    assert dune_1984 != dune_2021, "same-titled films must not share an id"

    # Same identity must stay stable across runs, or re-ingest duplicates everything.
    assert stable_id("movie", 841) == dune_1984
    # A movie and a TV series with the same TMDB number are different things.
    assert stable_id("movie", 1399) != stable_id("tv", 1399)
    print("  ok  stable_id separates remakes and kinds")


def test_normalize():
    assert _normalize("Let&#039;s Play!") == "let s play"
    assert _normalize("Cyberpunk: Edgerunners") == "cyberpunk edgerunners"
    assert _normalize("  ") == ""
    print("  ok  _normalize")


def test_score_cards_reflects_similarity_not_position():
    """Regression: /similar wrote a constant 0.0 logit on every card, so all of
    them rendered an identical 82% MATCH. Scores must now track the cosine.

    Cosines span the measured relevant-to-irrelevant range for this catalogue.
    """
    cards = [{"id": f"c{i}", "title": f"T{i}", "description": "d", "rating": 7.0,
              "votes": 5000} for i in range(6)]
    cosines = {"c0": 0.85, "c1": 0.76, "c2": 0.68, "c3": 0.60, "c4": 0.52, "c5": 0.44}
    out = _score_cards("query", cards, cosines)
    scores = [c["score"] for c in out]
    assert len(set(scores)) > 1, f"flat scores are the 82% bug: {scores}"
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > scores[-1] + 30, f"badge barely discriminates: {scores}"
    assert all(k not in c for c in out for k in ("_rel", "_order", "_pin"))
    print(f"  ok  _score_cards tracks similarity: {scores}")


def test_score_cards_is_query_independent():
    """The same match strength must produce the same badge in any result set —
    the old rank-based score gave the top hit 99% however bad the pool was."""
    good = _score_cards("q", [{"id": "a", "title": "A", "rating": 7, "votes": 5000}],
                        {"a": 0.84})[0]["score"]
    poor = _score_cards("q", [{"id": "b", "title": "B", "rating": 7, "votes": 5000}],
                        {"b": 0.45})[0]["score"]
    assert good > 80 and poor < 30, (good, poor)
    print(f"  ok  badges are absolute: strong-alone={good} weak-alone={poor}")


def test_score_cards_pins_exact_title_first():
    """Typing a title must put it top even when the vectors disagree."""
    pinned = {"id": "p", "title": "Dune", "rating": 8.0, "votes": 900_000, "_pin": "exact"}
    better_vector = {"id": "o", "title": "Other", "rating": 8.0, "votes": 900_000}
    out = _score_cards("dune", [better_vector, pinned], {"p": 0.52, "o": 0.84})
    assert out[0]["id"] == "p", [c["id"] for c in out]
    assert out[0]["score"] > out[1]["score"]
    print("  ok  exact-title pin leads regardless of vector order")


def test_dedupe_key_keeps_remakes():
    """Catalogue dedupe keys on title+year so both Dunes survive."""
    assert dedupe_key("Dune", "1984") != dedupe_key("Dune", "2021")
    assert dedupe_key("Dune", "2021") == dedupe_key("dune", "2021")
    print("  ok  catalogue dedupe keeps remakes")


def test_live_collection():
    """Only runs against a live catalogue. Asserts it is actually populated."""
    if not os.getenv("DATABASE_URL", "").startswith("postgres"):
        print("  skip  live collection (no Postgres DATABASE_URL)")
        return
    health = collection_health()
    assert health["ok"], f"index unhealthy: {health}"
    assert health["points"] > 0, f"catalogue is empty: {health}"
    print(f"  ok  live collection has {health['points']} points")


def test_live_search():
    """Smoke test against a real catalogue.

    Skipped on a small table: the other suites leave a handful of synthetic
    fixtures behind, and with seven rows of hand-built vectors every cosine
    lands in the same band, so a score-variety assertion would fail for reasons
    that say nothing about the engine.
    """
    if not os.getenv("DATABASE_URL", "").startswith("postgres"):
        print("  skip  live search (no Postgres DATABASE_URL)")
        return
    if collection_health()["points"] < 100:
        print("  skip  live search (catalogue too small to be real)")
        return
    from backend.engine import hybrid_search

    hits = hybrid_search("cyberpunk anime about identity", top_k=5)
    assert hits, "hybrid search returned nothing"
    assert all(h.get("title") for h in hits), "a hit is missing its title"
    assert all(1 <= h["score"] <= 99 for h in hits), "badge out of range"
    assert len({h["score"] for h in hits}) > 1, "all hits share one score"
    print(f"  ok  live search: {', '.join(h['title'] for h in hits[:3])}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    print(f"Running {len(tests)} checks\n")
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
