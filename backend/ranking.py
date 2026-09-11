"""
Ranking math for Nexus.

Pure functions, no I/O, no model loading — so every constant here is testable and
tunable without touching the retrieval plumbing.

What this replaces:

  * `_score_for_ui(rank, total)` — the %MATCH badge was a linear map of LIST
    POSITION (99 down to 60). It carried no information: the 12th result of a
    perfect query and the 12th result of a nonsense query both read 60%.
  * A hardcoded 7.5–9.2 rating window in `top_rated`, used to stop obscure
    10/10 shorts winning. That is a Bayesian prior implemented as a guess;
    `bayesian_rating` does it properly.
  * Qdrant's opaque built-in RRF, which fuses dense and sparse with equal,
    unchangeable weight and cannot take a third channel.

The pipeline is:

    per-channel ranks ──► weighted_rrf ──► relevance ∈ [0,1]
                                            │
    votes + rating ──► bayesian_rating ──► quality ∈ [0,1]
                                            │
                                            ▼
                                    blend_score  (ordering)
                                    match_percent (what the user sees)
"""

from __future__ import annotations

import math
import os
import re
from typing import Iterable, Mapping, Sequence

# --- Tunables -----------------------------------------------------------------
# Every one of these is an env override so ranking can be tuned on a deployed
# instance without a code change.

def _f(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# Reciprocal Rank Fusion damping. The canonical k=60 comes from TREC, where it
# fuses dozens of runs and deliberately flattens each one's contribution. With
# two or three channels that flattening is all cost: at k=60 the 40th result
# still scores 60% of the 1st, so nothing separates. k=20 keeps RRF's rank
# robustness while actually discriminating.
RRF_K = _f("RRF_K", 20.0)

# Channel weights. Dense carries meaning, sparse carries exact words, tags carry
# curated theme vocabulary. Sparse is deliberately close to dense: exact title and
# cast-name matches are the queries users notice most when they fail.
W_DENSE = _f("W_DENSE", 1.0)
W_SPARSE = _f("W_SPARSE", 0.9)
W_TAG = _f("W_TAG", 0.5)

# TMDB's global mean vote lands around 6.7. `BAYES_PRIOR_VOTES` is how many votes
# a title needs before its own score outweighs that mean — the classic IMDb-style
# shrinkage. At 500, a 10/10 with 3 votes lands near the mean, while a 8.5 with
# 20k votes barely moves.
BAYES_PRIOR_MEAN = _f("BAYES_PRIOR_MEAN", 6.7)
BAYES_PRIOR_VOTES = _f("BAYES_PRIOR_VOTES", 500.0)

# How hard quality pulls on the final ordering. Relevance stays dominant: this is
# a search engine, not a popularity chart. 0 disables the prior entirely.
QUALITY_WEIGHT = _f("QUALITY_WEIGHT", 0.22)

# Votes at which a title counts as fully "established" for the recognisability
# term. Log-scaled, so the curve is steep early and flat past mainstream.
VOTES_SATURATION = _f("VOTES_SATURATION", 25000.0)

# Cross-encoder logit -> probability temperature. ms-marco cross-encoders are
# trained with a binary objective, so plain sigmoid(logit) is already roughly
# calibrated; T>1 only softens it. The old code used shift=7.0/temp=4.6, which
# pushed genuinely irrelevant results up to ~30% and squashed the useful range.
MATCH_TEMP = _f("MATCH_TEMP", 1.8)

# Cosine-similarity calibration for the dense channel.
#
# Embedding cosines do not span [0,1], so the raw value cannot be shown as a
# percentage — it would call unrelated text a "43% match". These anchors rescale
# the band that actually carries signal.
#
# MEASURED, not guessed, over this catalogue's documents with bge-small
# (10 labelled probe queries, 70 query-document pairs):
#
#              n     min     p25     median    max
#   relevant   10   0.457   0.547    0.625    0.853
#   irrelevant 60   0.319     -      0.433    0.599   (p90 = 0.504)
#
# The floor sits just above the irrelevant mass and the ceiling near the top of
# the relevant mass. An earlier guess of 0.62/0.88 sat above the relevant MEDIAN,
# which floored almost every badge at the minimum.
#
# These are specific to DENSE_MODEL and to the document template in
# catalogue/schema.py. Re-measure if either changes.
COSINE_FLOOR = _f("COSINE_FLOOR", 0.48)
COSINE_CEIL = _f("COSINE_CEIL", 0.82)

# Weight of the cross-encoder's ranking when fused with the retrieval channels.
# It is fused as a RANK, never as a score — see relevance_from_logit.
W_RERANK = _f("W_RERANK", 1.2)

# Displayed %MATCH is clamped into this band. A floor above 0 avoids telling a
# user a result is literally 0% when we chose to show it; the ceiling keeps 100%
# meaning "this is exactly the thing you typed".
MATCH_FLOOR = int(_f("MATCH_FLOOR", 12))
MATCH_CEIL = int(_f("MATCH_CEIL", 99))


# --- Quality ------------------------------------------------------------------

def bayesian_rating(rating: float, votes: float,
                    prior_mean: float = None, prior_votes: float = None) -> float:
    """Vote-count-shrunk rating (the IMDb weighted-rating formula).

        WR = (v / (v + m)) * R  +  (m / (v + m)) * C

    A 10.0 from 3 voters is not evidence of a better film than an 8.7 from
    40,000; shrinking toward the catalogue mean encodes exactly that. Returns a
    value on the same 0–10 scale as the input.
    """
    prior_mean = BAYES_PRIOR_MEAN if prior_mean is None else prior_mean
    prior_votes = BAYES_PRIOR_VOTES if prior_votes is None else prior_votes
    r = max(0.0, min(10.0, _num(rating)))
    v = max(0.0, _num(votes))
    if v <= 0:
        return prior_mean
    return (v / (v + prior_votes)) * r + (prior_votes / (v + prior_votes)) * prior_mean


def recognisability(votes: float) -> float:
    """0–1 measure of how widely seen a title is, log-scaled.

    Vote counts span five orders of magnitude (a short with 12, Fight Club with
    30k), so linear scaling makes everything below the blockbusters identical.
    """
    v = max(0.0, _num(votes))
    return min(1.0, math.log1p(v) / math.log1p(VOTES_SATURATION))


def quality_prior(rating: float, votes: float) -> float:
    """Combined 0–1 quality signal: is it good, and is it actually watched?

    Geometric mean, so a title must score on BOTH. A brilliant film nobody has
    rated and a mediocre film everyone has rated are both mid; something both
    well-reviewed and widely seen wins.
    """
    good = bayesian_rating(rating, votes) / 10.0
    known = recognisability(votes)
    return math.sqrt(max(good, 1e-6) * max(known, 1e-6))


# --- Fusion -------------------------------------------------------------------

def weighted_rrf(channels: Mapping[str, Sequence], weights: Mapping[str, float] = None,
                 k: float = None) -> dict:
    """Weighted Reciprocal Rank Fusion over any number of ranked id lists.

        score(d) = Σ_c  w_c / (k + rank_c(d))

    Rank-based rather than score-based, so channels whose scores live on
    incomparable scales (cosine similarity vs BM25 vs tag overlap) can be fused
    without normalising anything. Qdrant's built-in RRF does this with fixed
    equal weights and exactly two inputs; doing it here buys tunable weights and
    room for the tag channel.

    `channels` maps a channel name to its ranked ids (best first).
    Returns {id: fused_score}, unsorted.
    """
    k = RRF_K if k is None else k
    weights = weights or {"dense": W_DENSE, "sparse": W_SPARSE, "tag": W_TAG}
    fused: dict = {}
    for name, ranked in channels.items():
        w = float(weights.get(name, 1.0))
        if w == 0:
            continue
        for rank, doc_id in enumerate(ranked):
            fused[doc_id] = fused.get(doc_id, 0.0) + w / (k + rank + 1)
    return fused


def rrf_ceiling(channels: Iterable[str], weights: Mapping[str, float] = None,
                k: float = None) -> float:
    """Best attainable RRF score: every channel ranking the same doc first.

    Used to turn a fused score into a 0–1 relevance without min-maxing against
    the pool — min-max would make the top hit of a hopeless query look perfect.
    """
    k = RRF_K if k is None else k
    weights = weights or {"dense": W_DENSE, "sparse": W_SPARSE, "tag": W_TAG}
    return sum(float(weights.get(c, 1.0)) for c in channels) / (k + 1) or 1.0


# --- Relevance and the number the user sees -----------------------------------

def sigmoid(x: float) -> float:
    if x <= -30:
        return 0.0
    if x >= 30:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def relevance_from_logit(logit: float, temp: float = None) -> float:
    """Cross-encoder logit -> 0–1 relevance probability.

    NOT used for the badge by default, and the measurement explains why. Over
    the same 70 labelled pairs used to calibrate the cosine, ms-marco-MiniLM
    produced:

        relevant    median  -10.91   (min -11.33, max +9.54)
        irrelevant  median  -11.33   (max -10.50)

    The two distributions almost entirely overlap and both sit deep in the
    negative tail, so sigmoid() maps essentially everything to ~0. The model is
    trained to answer "does this passage answer this question", which is not the
    question a catalogue asks, and its absolute output does not transfer.

    Its ORDERING is still better than the vector order alone, so the reranker is
    fused into `weighted_rrf` as a rank channel instead. Same lesson as
    relevance_from_rrf: a good ranker is not automatically a relevance measure.

    Kept because a reranker calibrated for this domain (or a differently trained
    one) could legitimately feed the badge — set RERANK_SCORES_BADGE=true then,
    after re-measuring.
    """
    return sigmoid(_num(logit) / (MATCH_TEMP if temp is None else temp))


def relevance_from_cosine(cosine: float) -> float:
    """Dense cosine similarity -> 0–1 relevance.

    This is the ABSOLUTE relevance signal used when no cross-encoder is loaded.
    Unlike an RRF score it measures query-to-document content, so it can say
    "nothing here actually matches" — a rank never can.

    Rescaled from the band where the model carries signal (see COSINE_FLOOR /
    COSINE_CEIL); raw cosines would report unrelated text as a ~62% match.
    """
    c = _num(cosine)
    span = COSINE_CEIL - COSINE_FLOOR
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (c - COSINE_FLOOR) / span))


def relevance_from_rrf(fused: float, ceiling: float) -> float:
    """Fused RRF score -> 0–1 pseudo-relevance. ORDERING ONLY — last resort.

    Read this before using it for anything user-visible: RRF is rank-based, so
    it is structurally incapable of expressing absolute relevance. Rank 40 out
    of 60 useless candidates and rank 40 out of 60 excellent ones produce the
    identical score. Deriving a %MATCH badge from it reproduces exactly the bug
    this module exists to remove.

    Prefer `relevance_from_logit` (cross-encoder) or `relevance_from_cosine`
    (dense similarity) for anything shown to a user. This stays for ordering and
    for the degenerate case where neither signal is available.
    """
    if ceiling <= 0:
        return 0.0
    return max(0.0, min(1.0, _num(fused) / ceiling))


def blend_score(relevance: float, rating: float, votes: float,
                quality_weight: float = None) -> float:
    """Final ordering score: relevance, nudged by quality.

    Log-linear (a weighted geometric mean) rather than additive, so quality
    scales relevance instead of being able to substitute for it — a popular film
    that does not match the query cannot climb over one that does.
    """
    qw = QUALITY_WEIGHT if quality_weight is None else quality_weight
    rel = max(_num(relevance), 1e-9)
    if qw <= 0:
        return rel
    q = max(quality_prior(rating, votes), 1e-9)
    return math.exp(math.log(rel) + qw * math.log(q))


# A lexical hit on a RARE query is strong content evidence that the dense model
# cannot express. "Studio Ghibli" matched ~30 documents in 50k, every one of
# them a Ghibli film — yet Spirited Away ranked 26th, because its cosine against
# a two-word studio name is unremarkable. Corpus rarity is measured exactly by
# Postgres (how many documents match the tsquery at all), so this is a counted
# statistic rather than a tuned guess.
RARE_FLOOR = _f("RARE_FLOOR", 10.0)     # at or below this, maximally distinctive
RARE_CEIL = _f("RARE_CEIL", 1500.0)     # at or above this, an ordinary word
LEX_MAX_RELEVANCE = _f("LEX_MAX_RELEVANCE", 0.88)
# Gentle. Rarity is a property of the QUERY, so when only ~30 documents match
# it at all, every one of them is probably relevant — position inside that small
# set is weak information. A steep decay punished exactly the titles this is
# meant to rescue: the Ghibli films mention "studio ghibli" once, while a
# documentary about Ghibli repeats it and took the whole boost.
LEX_RANK_DECAY = _f("LEX_RANK_DECAY", 0.03)


def rarity_relevance(lex_total: float, lex_rank: float) -> float:
    """Relevance implied by matching a rare query lexically.

    Returns 0 for common queries, so an ordinary descriptive search is
    untouched and only distinctive ones — studios, franchises, proper nouns —
    get lifted.
    """
    n, rank = _num(lex_total), _num(lex_rank)
    if n <= 0 or rank <= 0 or n >= RARE_CEIL:
        return 0.0
    n = max(n, RARE_FLOOR)
    rarity = math.log(RARE_CEIL / n) / math.log(RARE_CEIL / RARE_FLOOR)
    decay = 1.0 / (1.0 + LEX_RANK_DECAY * (rank - 1.0))
    return max(0.0, min(1.0, rarity)) * decay * LEX_MAX_RELEVANCE


def match_percent(relevance: float, pin: str = None) -> int:
    """The 0–99 %MATCH badge.

    A monotone map of genuine relevance, so the number is comparable between
    queries: 90% always means the same strength of match. `pin` marks a result
    that matched the typed title itself ("exact" / "prefix"), which the user
    experiences as certainty regardless of what the semantic model thinks.
    """
    rel = max(0.0, min(1.0, _num(relevance)))
    # Mild gamma: raw probabilities cluster low, and a badge that reads 14% for a
    # decent hit is not useful. Monotone, so ordering is untouched.
    pct = round(100 * (rel ** 0.65))
    if pin == "exact":
        pct = max(pct, 97)
    elif pin == "prefix":
        pct = max(pct, 88)
    else:
        pct = min(pct, 96)  # keep headroom so pins always read strongest
    return max(MATCH_FLOOR, min(MATCH_CEIL, pct))


# --- Title matching -----------------------------------------------------------

_WORD = re.compile(r"[a-z0-9]+")


def title_similarity(query: str, title: str) -> float:
    """0–1 token-set similarity, for deciding whether a hit IS the typed title.

    Exact/prefix string comparison misses "lotr fellowship", "star wars a new
    hope" and every typo. Token-set Jaccard with a containment bonus handles
    partial titles, which is what people actually type.
    """
    q = set(_WORD.findall(str(query).lower()))
    t = set(_WORD.findall(str(title).lower()))
    if not q or not t:
        return 0.0
    inter = len(q & t)
    if not inter:
        return 0.0
    jaccard = inter / len(q | t)
    containment = inter / len(q)  # all query words present => the title contains it
    return max(jaccard, 0.9 * containment if containment == 1.0 else containment * 0.75)


def classify_pin(query: str, title: str, threshold: float = 0.62) -> str | None:
    """'exact' | 'prefix' | None — how strongly a result matches the typed title."""
    nq = " ".join(_WORD.findall(str(query).lower()))
    nt = " ".join(_WORD.findall(str(title).lower()))
    if not nq or not nt:
        return None
    if nq == nt:
        return "exact"
    if nt.startswith(nq + " ") or nq.startswith(nt + " "):
        return "prefix"
    return "prefix" if title_similarity(query, title) >= threshold else None


# --- helpers ------------------------------------------------------------------

def _num(v) -> float:
    try:
        f = float(v)
        return 0.0 if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return 0.0
