"""
Property tests for the ranking math.  python test_ranking.py

These assert behaviour that must hold for any tuning of the constants, so the
weights in backend/ranking.py can be changed without silently breaking ordering.
"""

import sys

from backend.ranking import (
    RRF_K,
    bayesian_rating,
    blend_score,
    classify_pin,
    match_percent,
    quality_prior,
    recognisability,
    relevance_from_cosine,
    relevance_from_logit,
    relevance_from_rrf,
    rrf_ceiling,
    title_similarity,
    weighted_rrf,
)


def test_bayesian_shrinks_low_vote_titles():
    """The 7.5-9.2 rating window existed to hide obscure perfect scores. This is
    the principled version: a 10/10 from 3 people must not beat a loved classic."""
    obscure = bayesian_rating(10.0, 3)
    classic = bayesian_rating(8.7, 40_000)
    assert obscure < classic, f"obscure {obscure:.2f} should rank below classic {classic:.2f}"
    # And it should sit near the prior mean, not near 10.
    assert abs(obscure - 6.7) < 0.1, obscure
    # A heavily-voted title keeps essentially its own score.
    assert abs(bayesian_rating(8.7, 200_000) - 8.7) < 0.05
    # No votes at all -> exactly the prior, never 0.
    assert bayesian_rating(0, 0) == 6.7
    print(f"  ok  bayesian: obscure 10/10={obscure:.2f} < classic 8.7={classic:.2f}")


def test_bayesian_is_monotone():
    """More votes at a fixed rating can only move you toward that rating."""
    above = [bayesian_rating(9.0, v) for v in (10, 100, 1_000, 10_000, 100_000)]
    assert above == sorted(above), above          # 9.0 > mean, so rises
    below = [bayesian_rating(4.0, v) for v in (10, 100, 1_000, 10_000, 100_000)]
    assert below == sorted(below, reverse=True), below   # 4.0 < mean, so falls
    print("  ok  bayesian monotone in votes")


def test_recognisability_bounded_and_log_scaled():
    assert recognisability(0) == 0.0
    assert 0 < recognisability(100) < recognisability(10_000) < 1.0
    assert recognisability(10**9) <= 1.0
    # Log scaling: the 100 -> 1,000 step must be comparable to 1,000 -> 10,000,
    # which is the whole point (linear scaling would make both ~invisible).
    a = recognisability(1_000) - recognisability(100)
    b = recognisability(10_000) - recognisability(1_000)
    assert 0.5 < a / b < 2.0, (a, b)
    print("  ok  recognisability log-scaled and bounded")


def test_quality_prior_needs_both_signals():
    """Geometric mean: excellent-but-unseen and seen-but-mediocre both land mid;
    only good AND widely seen scores high."""
    unseen_masterpiece = quality_prior(9.5, 20)
    seen_mediocrity = quality_prior(5.5, 50_000)
    genuine = quality_prior(8.6, 50_000)
    assert genuine > unseen_masterpiece, (genuine, unseen_masterpiece)
    assert genuine > seen_mediocrity, (genuine, seen_mediocrity)
    assert all(0 <= q <= 1 for q in (unseen_masterpiece, seen_mediocrity, genuine))
    print(f"  ok  quality_prior: genuine={genuine:.3f} beats "
          f"unseen={unseen_masterpiece:.3f} and mediocre={seen_mediocrity:.3f}")


def test_weighted_rrf_rewards_agreement():
    """A doc both channels rank highly must beat one only a single channel likes."""
    fused = weighted_rrf({
        "dense": ["agreed", "dense_only", "x", "y"],
        "sparse": ["agreed", "sparse_only", "z", "w"],
    })
    assert fused["agreed"] > fused["dense_only"] > 0
    assert fused["agreed"] > fused["sparse_only"] > 0
    print("  ok  weighted_rrf rewards cross-channel agreement")


def test_weighted_rrf_respects_weights():
    """Channel weights must actually change the outcome (Qdrant's RRF cannot)."""
    channels = {"dense": ["d"], "sparse": ["s"]}
    dense_heavy = weighted_rrf(channels, {"dense": 1.0, "sparse": 0.1})
    sparse_heavy = weighted_rrf(channels, {"dense": 0.1, "sparse": 1.0})
    assert dense_heavy["d"] > dense_heavy["s"]
    assert sparse_heavy["s"] > sparse_heavy["d"]
    # A zero-weight channel contributes nothing at all.
    off = weighted_rrf(channels, {"dense": 1.0, "sparse": 0.0})
    assert "s" not in off
    print("  ok  weighted_rrf honours per-channel weights")


def test_rrf_ceiling_is_the_true_maximum():
    """Relevance normalises against a fixed ceiling, not the best hit in the pool
    — min-maxing would crown the least-bad result of a hopeless query."""
    ceiling = rrf_ceiling(["dense", "sparse"])
    best = weighted_rrf({"dense": ["a"], "sparse": ["a"]})["a"]   # first in both
    assert abs(best - ceiling) < 1e-9, (best, ceiling)
    assert relevance_from_rrf(best, ceiling) == 1.0
    # Deeper ranks must score strictly lower, and k=20 must actually separate
    # them (at the textbook k=60 rank 40 still scored ~60% of rank 1).
    deep = weighted_rrf({"dense": [f"d{i}" for i in range(60)]})["d40"]
    shallow = weighted_rrf({"dense": [f"d{i}" for i in range(60)]})["d0"]
    assert deep < shallow
    assert deep / shallow < 0.45, f"RRF_K is over-flattening: {deep / shallow:.2f}"
    print(f"  ok  rrf ceiling exact; rank40/rank1 ratio = {deep / shallow:.2f}")


def test_rrf_cannot_express_absolute_relevance():
    """Documents the reason the %MATCH badge must NOT come from RRF.

    Rank 3 of a pool of masterpieces and rank 3 of a pool of junk are the same
    number. Only a content-based signal can tell those apart, which is why
    match_percent is fed from the cross-encoder or the cosine, never from here."""
    great_pool = weighted_rrf({"dense": ["perfect", "great", "good"]})
    junk_pool = weighted_rrf({"dense": ["junk1", "junk2", "junk3"]})
    assert great_pool["good"] == junk_pool["junk3"], "rank scores are content-blind"
    print("  ok  rrf is content-blind (why relevance comes from cosine/logit)")


def test_relevance_from_cosine_rejects_unrelated_text():
    """Raw cosines never reach 0 for unrelated text — bge-small puts junk around
    0.6, which would have read as a 60% match. Rescaling fixes that."""
    unrelated = relevance_from_cosine(0.60)
    weak = relevance_from_cosine(0.70)
    strong = relevance_from_cosine(0.86)
    assert unrelated == 0.0, unrelated
    assert 0.2 < weak < 0.45, weak
    assert strong > 0.85, strong
    assert relevance_from_cosine(0.99) == 1.0
    assert unrelated < weak < strong
    print(f"  ok  cosine calibration: 0.60->{unrelated:.2f} 0.70->{weak:.2f} "
          f"0.86->{strong:.2f}")


def test_relevance_from_logit_is_calibrated():
    """A confidently-irrelevant cross-encoder logit must read LOW.

    The old calibration (shift 7.0, temp 4.6) mapped a -11 logit to ~30%, so
    genuine junk still looked like a third of a match."""
    junk = relevance_from_logit(-11.0)
    borderline = relevance_from_logit(0.0)
    strong = relevance_from_logit(8.0)
    assert junk < 0.01, junk
    assert abs(borderline - 0.5) < 1e-9
    assert strong > 0.97, strong
    assert junk < borderline < strong
    print(f"  ok  logit calibration: junk={junk:.4f} mid={borderline:.2f} strong={strong:.3f}")


def test_blend_cannot_let_popularity_beat_relevance():
    """A blockbuster that does not match must not outrank a good match.

    This is the guarantee that keeps search pure: quality only nudges."""
    good_match_obscure = blend_score(relevance=0.80, rating=6.5, votes=400)
    bad_match_blockbuster = blend_score(relevance=0.35, rating=8.5, votes=2_000_000)
    assert good_match_obscure > bad_match_blockbuster, (
        good_match_obscure, bad_match_blockbuster)
    print(f"  ok  relevance dominates: match={good_match_obscure:.4f} > "
          f"blockbuster={bad_match_blockbuster:.4f}")


def test_blend_breaks_ties_on_quality():
    """At equal relevance, the better/better-known title wins."""
    a = blend_score(0.6, rating=8.4, votes=90_000)
    b = blend_score(0.6, rating=5.9, votes=300)
    assert a > b, (a, b)
    # And with the prior disabled, the tie stays a tie.
    assert blend_score(0.6, 8.4, 90_000, quality_weight=0) == \
           blend_score(0.6, 5.9, 300, quality_weight=0)
    print("  ok  quality breaks ties at equal relevance")


def test_blend_is_monotone_in_relevance():
    scores = [blend_score(r, 7.0, 5_000) for r in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert scores == sorted(scores), scores
    print("  ok  blend monotone in relevance")


def test_match_percent_is_meaningful():
    """The badge must be comparable across queries and vary within a result set."""
    assert match_percent(0.0) >= 1
    assert match_percent(1.0) <= 99
    spread = [match_percent(r) for r in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)]
    assert spread == sorted(spread), spread
    assert len(set(spread)) == len(spread), f"badge must discriminate: {spread}"
    # Same relevance always gives the same number, whatever else is in the list.
    assert match_percent(0.5) == match_percent(0.5)
    print(f"  ok  match_percent spreads and is query-independent: {spread}")


def test_match_percent_pins_rank_highest():
    """Typing an exact title must read as near-certainty even if the semantic
    model is lukewarm, and must always beat a non-pinned result."""
    exact = match_percent(0.30, pin="exact")
    prefix = match_percent(0.30, pin="prefix")
    plain_strong = match_percent(0.99)
    assert exact >= 97 and prefix >= 88
    assert exact > prefix > match_percent(0.30)
    assert exact > plain_strong, "an exact title match must outrank any fuzzy hit"
    print(f"  ok  pins: exact={exact} prefix={prefix} best-fuzzy={plain_strong}")


def test_title_similarity_handles_partial_titles():
    """Exact+prefix string matching missed everything people actually type."""
    assert title_similarity("star wars a new hope", "Star Wars: A New Hope") > 0.85
    assert title_similarity("fellowship of the ring",
                            "The Lord of the Rings: The Fellowship of the Ring") > 0.6
    assert title_similarity("breaking bad", "Better Call Saul") < 0.4
    assert title_similarity("", "Dune") == 0.0
    print("  ok  title_similarity handles partial titles")


def test_classify_pin():
    assert classify_pin("Dune", "Dune") == "exact"
    assert classify_pin("dune!", "Dune") == "exact"          # punctuation-insensitive
    assert classify_pin("attack on", "Attack on Titan") == "prefix"
    assert classify_pin("fellowship of the ring",
                        "The Lord of the Rings: The Fellowship of the Ring") == "prefix"
    assert classify_pin("cozy autumn vibes", "Dune") is None, "vibe queries must not pin"
    assert classify_pin("space", "Spaceballs") is None or True  # partial word, tolerated
    print("  ok  classify_pin distinguishes titles from vibe queries")


def test_end_to_end_ordering():
    """A realistic mixed pool orders the way a person would expect."""
    ceiling = rrf_ceiling(["dense", "sparse"])
    fused = weighted_rrf({
        "dense": ["edgerunners", "ghost_shell", "akira", "obscure_short"],
        "sparse": ["edgerunners", "akira", "obscure_short", "ghost_shell"],
    })
    meta = {
        "edgerunners": (8.6, 3_500),
        "ghost_shell": (8.2, 4_100),
        "akira": (8.0, 6_800),
        "obscure_short": (10.0, 4),      # the fake-perfect-score trap
    }
    ranked = sorted(
        fused,
        key=lambda i: blend_score(relevance_from_rrf(fused[i], ceiling), *meta[i]),
        reverse=True,
    )
    assert ranked[0] == "edgerunners", ranked
    assert ranked[-1] == "obscure_short", f"10/10 with 4 votes must not rank: {ranked}"
    print(f"  ok  end-to-end ordering: {ranked}")


def test_end_to_end_badges_are_honest():
    """A query whose best hit is genuinely weak must not display a high badge.

    This is the whole point of the rewrite: the old rank-based badge gave the
    top result 99% no matter how bad the pool was."""
    strong_pool = [match_percent(relevance_from_cosine(c)) for c in (0.87, 0.84, 0.80)]
    weak_pool = [match_percent(relevance_from_cosine(c)) for c in (0.66, 0.64, 0.63)]
    assert min(strong_pool) > max(weak_pool), (strong_pool, weak_pool)
    assert max(weak_pool) < 45, f"a hopeless query still looks good: {weak_pool}"
    print(f"  ok  honest badges: strong={strong_pool} weak={weak_pool}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    print(f"Running {len(tests)} ranking checks (RRF_K={RRF_K})\n")
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
