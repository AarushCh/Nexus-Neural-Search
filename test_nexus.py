"""
Smoke + regression tests for the Nexus engine.

Plain asserts, no framework:  python test_nexus.py

The pure-logic tests run anywhere. The live tests only run when QDRANT_URL is
configured, and are skipped otherwise so this stays useful in CI without secrets.
"""

import os
import sys

from backend.engine import (
    _calibrate_scores,
    _normalize,
    _score_for_ui,
    collection_health,
)
from build_catalogue import dedupe_key, stable_id


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


def test_score_spread():
    """Rank-based fallback must actually vary, and stay inside the badge range."""
    scores = [_score_for_ui(i, 12) for i in range(12)]
    assert scores[0] == 99
    assert len(set(scores)) > 1, "every card showing the same % is the bug"
    assert all(0 < s <= 99 for s in scores)
    assert scores == sorted(scores, reverse=True)
    print("  ok  _score_for_ui spreads")


def test_calibrate_without_reranker_is_not_flat():
    """Regression: /similar set _rr = 0.0 on every card when rerank was off, so
    _calibrate_scores turned all of them into an identical 82% MATCH."""
    cards = [{"title": f"T{i}", "description": "d"} for i in range(8)]
    _calibrate_scores("query", cards, logit_key="_rr")
    scores = [c["score"] for c in cards]
    assert len(set(scores)) > 1, f"flat scores are the 82% bug: {scores}"
    assert all("_rr" not in c for c in cards), "internal keys must not leak to the UI"
    print(f"  ok  _calibrate_scores varies without a reranker: {scores}")


def test_calibrate_pins_rank_above_plain_hits():
    """An exact title match must outrank an ordinary hit with the same logit."""
    pinned = {"title": "Dune", "description": "d", "_rr": 0.0, "_pin": "exact"}
    plain = {"title": "Other", "description": "d", "_rr": 0.0}
    _calibrate_scores("dune", [pinned, plain], logit_key="_rr")
    assert pinned["score"] > plain["score"], (pinned["score"], plain["score"])
    assert "_pin" not in pinned
    print("  ok  exact-title pins score above plain hits")


def test_dedupe_key_keeps_remakes():
    """Catalogue dedupe keys on title+year so both Dunes survive."""
    assert dedupe_key("Dune", "1984") != dedupe_key("Dune", "2021")
    assert dedupe_key("Dune", "2021") == dedupe_key("dune", "2021")
    print("  ok  catalogue dedupe keeps remakes")


def test_live_collection():
    """Only runs with a configured cluster. Asserts the index is actually populated."""
    if not os.getenv("QDRANT_URL"):
        print("  skip  live collection (QDRANT_URL not set)")
        return
    health = collection_health()
    assert health["ok"], f"index unhealthy: {health}"
    assert health["points"] > 1000, f"suspiciously small catalogue: {health['points']}"
    print(f"  ok  live collection has {health['points']} points")


def test_live_search():
    if not os.getenv("QDRANT_URL"):
        print("  skip  live search (QDRANT_URL not set)")
        return
    from backend.engine import hybrid_search

    hits = hybrid_search("cyberpunk anime about identity", top_k=5)
    assert hits, "hybrid search returned nothing"
    assert all(h.get("title") for h in hits), "a hit is missing its title"
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
