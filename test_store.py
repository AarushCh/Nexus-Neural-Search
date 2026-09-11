"""
Integration tests for the pgvector catalogue store.  python test_store.py

Runs every query against a REAL Postgres with pgvector — schema, HNSW search,
full-text search, RRF fusion, Rocchio recommendation, facet counts. Skips
cleanly when no Postgres is configured so CI stays green without a service.

Local:
    docker run -d --name nexus-pg -e POSTGRES_PASSWORD=nexus -e POSTGRES_DB=nexus \
        -p 55432:5432 pgvector/pgvector:pg16
    DATABASE_URL=postgresql://postgres:nexus@localhost:55432/nexus python test_store.py
"""

import math
import os
import random
import sys

if not os.getenv("DATABASE_URL", "").startswith("postgres"):
    print("skip  test_store: set DATABASE_URL to a Postgres instance with pgvector")
    sys.exit(0)

from backend import ranking, store  # noqa: E402

DIM = store.VECTOR_SIZE
random.seed(7)


CLUSTERS = {"cyberpunk": 0, "cozy": 1, "desert": 2, "misc": 3}


_BASE = [random.Random("shared-base").uniform(-1, 1) for _ in range(DIM)]
_BASE = [x / math.sqrt(sum(y * y for y in _BASE)) for x in _BASE]


def _vec(cluster: str, jitter: float = 0.0) -> list:
    """Deterministic unit vector with a REALISTIC cosine spread.

    Two earlier attempts were both wrong in instructive ways:

      sin(seed * i)  — nearby seeds diverge wildly by dimension 384, so
                       "similar" fixtures were not similar at all and the
                       recommendation test was passing on noise.
      pure blocks    — perfectly clustered, so every within-cluster cosine came
                       out ~0.99. That saturates the calibration band, every
                       relevance ties at 1.0, and the tests silently stopped
                       measuring relevance at all.

    Real bge-small cosines sit around 0.85 for a close match and 0.55 for an
    unrelated one. Mixing a shared base direction (common to all text) with an
    orthogonal per-cluster direction reproduces that: the base term sets the
    cross-cluster floor, the cluster term adds the topical signal.
    """
    block = CLUSTERS[cluster]
    width = DIM // len(CLUSTERS)
    rnd = random.Random(f"{cluster}:{jitter}")

    # alpha^2 = cross-cluster cosine floor; alpha^2 + beta^2 = within-cluster.
    alpha, beta, gamma = 0.742, 0.548, 0.387   # -> ~0.55 apart, ~0.85 together
    cluster_dir = [0.0] * DIM
    for i in range(block * width, (block + 1) * width):
        cluster_dir[i] = 1.0
    cn = math.sqrt(sum(x * x for x in cluster_dir))
    cluster_dir = [x / cn for x in cluster_dir]

    noise = [rnd.uniform(-1, 1) for _ in range(DIM)]
    nn = math.sqrt(sum(x * x for x in noise)) or 1.0
    noise = [x / nn for x in noise]

    g = gamma * (1.0 + jitter)   # more jitter = further from its own cluster
    v = [alpha * _BASE[i] + beta * cluster_dir[i] + g * noise[i] for i in range(DIM)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


FIXTURES = [
    # id,   title,                   category, rating, votes, year, genres,     tags,                                              cluster,      jitter
    ("m1", "Blade Runner 2049",      "MOVIE",  7.6, 13000, 2017, ["sci-fi"], ["theme:dystopia", "theme:neo-noir", "era:2010s"],  "cyberpunk", 0.30),
    ("m2", "Ghost in the Shell",     "ANIME",  8.0,  4100, 1995, ["sci-fi"], ["theme:cyberpunk", "theme:identity", "era:1990s"], "cyberpunk", 0.12),
    ("m3", "Cyberpunk: Edgerunners", "ANIME",  8.6,  3500, 2022, ["sci-fi"], ["theme:cyberpunk", "theme:tragedy", "era:2020s"],  "cyberpunk", 0.10),
    ("m4", "Paddington 2",           "MOVIE",  8.4,  6000, 2017, ["family"], ["mood:cozy", "era:2010s"],                          "cozy",      0.10),
    ("m5", "Obscure Perfect Short",  "MOVIE", 10.0,     4, 2019, ["drama"],  ["era:2010s"],                                       "misc",      0.10),
    ("m6", "Dune",                   "MOVIE",  6.6,  1100, 1984, ["sci-fi"], ["franchise:dune", "era:1980s"],                     "desert",    0.20),
    ("m7", "Dune",                   "MOVIE",  8.0, 11000, 2021, ["sci-fi"], ["franchise:dune", "era:2020s"],                     "desert",    0.10),
]

VEC = {f[0]: _vec(f[8], f[9]) for f in FIXTURES}

DOCS = {
    "m1": "Blade Runner 2049 a young blade runner uncovers a secret dystopia neo noir replicant Denis Villeneuve",
    "m2": "Ghost in the Shell cyberpunk cyborg identity consciousness Japanese animation Mamoru Oshii",
    "m3": "Cyberpunk Edgerunners a street kid becomes a mercenary cyberpunk dystopia tragedy Studio Trigger",
    "m4": "Paddington 2 a kind bear searches for the perfect present cozy heartwarming family comedy",
    "m5": "Obscure Perfect Short a very short experimental drama film nobody has seen",
    "m6": "Dune 1984 a noble family takes control of a desert planet spice David Lynch",
    "m7": "Dune 2021 a noble family takes control of a desert planet spice Denis Villeneuve",
}


def setup():
    store.drop_all()
    store.create_schema()
    rows, docs, embs = [], [], []
    for cid, title, cat, rating, votes, year, genres, tags, _cluster, _jit in FIXTURES:
        rows.append({
            "id": cid, "tmdb_id": int(cid[1:]), "tmdb_kind": "movie", "title": title,
            "description": DOCS[cid], "category": cat, "genres": genres, "tags": tags,
            "types": ["film"], "forms": ["live-action"], "rating": rating,
            "votes": votes, "year_i": year, "image": f"http://img/{cid}.jpg",
            "cast": [{"name": "Someone"}], "crew": [], "providers": [],
            "alt_titles": [],
        })
        docs.append(DOCS[cid])
        embs.append(VEC[cid])
    store.upsert(rows, docs, embs)
    store.upsert_extra(rows)
    store.create_indexes()
    store.analyze()


# --- Schema and health --------------------------------------------------------

def test_health_distinguishes_empty_from_missing():
    """An empty catalogue must FAIL health, not look like 'no results'."""
    h = store.health()
    assert h["ok"] and h["titles"] == len(FIXTURES), h

    store.drop_all()
    gone = store.health()
    assert not gone["ok"] and "missing" in gone["error"], gone

    store.create_schema()
    empty = store.health()
    assert not empty["ok"] and empty["error"] == "catalogue empty", empty
    setup()
    print("  ok  health distinguishes populated / empty / missing")


# --- Hybrid retrieval ---------------------------------------------------------

def test_hybrid_returns_both_channels_and_the_cosine():
    """The whole reason for leaving Qdrant's server-side fusion: we need the
    cosine back, not just a fused rank."""
    res = store.hybrid_candidates("cyberpunk identity", VEC["m2"], limit=10)
    assert res["dense"], "dense channel returned nothing"
    assert res["lexical"], "lexical channel returned nothing"
    assert res["cosines"], "cosines missing — %MATCH would be meaningless"
    for cid, cos in res["cosines"].items():
        assert -1.0 <= cos <= 1.0, (cid, cos)
    assert "m2" in res["cards"]
    print(f"  ok  hybrid: {len(res['dense'])} dense, {len(res['lexical'])} lexical, "
          f"{len(res['cosines'])} cosines")


def test_lexical_channel_finds_exact_words_dense_would_miss():
    """This is what the sparse channel is for: a rare proper noun."""
    res = store.hybrid_candidates("Villeneuve", VEC["m1"], limit=10)
    assert "m1" in res["lexical"] or "m7" in res["lexical"], res["lexical"]
    print("  ok  lexical channel matches proper nouns")


def test_fulltext_stemming_works():
    """Postgres 'english' config stems, which raw BM25 did not."""
    res = store.hybrid_candidates("searching for presents", VEC["m4"], limit=10)
    assert "m4" in res["lexical"], "stemming failed (searches/searching)"
    print("  ok  full-text stemming matches inflected forms")


def test_end_to_end_ranking_is_honest():
    """Fuse the two channels with the tested RRF, score with the real cosine,
    and confirm the ordering and the badge both behave."""
    res = store.hybrid_candidates("cyberpunk dystopia anime", VEC["m3"], limit=10)
    fused = ranking.weighted_rrf({"dense": res["dense"], "sparse": res["lexical"]})
    ranked = sorted(fused, key=lambda i: ranking.blend_score(
        ranking.relevance_from_cosine(res["cosines"].get(i, 0)),
        res["cards"][i]["rating"], res["cards"][i]["votes"]), reverse=True)

    titles = [res["cards"][i]["title"] for i in ranked]
    # The three cyberpunk titles must take the top three; which of them leads is
    # a legitimate quality tiebreak, but a cozy bear film or a desert epic
    # appearing above them would mean relevance is not driving the ranking.
    assert set(ranked[:3]) == {"m1", "m2", "m3"}, titles
    # The 10/10 with four votes must not be near the top.
    assert ranked.index("m5") >= len(ranked) - 2, f"obscure perfect score ranked high: {titles}"

    # And the badge must separate the relevant from the irrelevant.
    top = ranking.match_percent(ranking.relevance_from_cosine(res["cosines"][ranked[0]]))
    bottom = ranking.match_percent(ranking.relevance_from_cosine(res["cosines"][ranked[-1]]))
    assert top > bottom + 20, f"badge barely discriminates: {top} vs {bottom}"
    print(f"  ok  end-to-end ranking: {titles[:3]} (badge {top}% .. {bottom}%)")


# --- Filters ------------------------------------------------------------------

def test_filters_are_bound_parameters_not_string_interpolation():
    frag, params = store.build_filter(category="ANIME", min_rating=8.0)
    assert ":category" in frag and params["category"] == "ANIME"
    res = store.hybrid_candidates("cyberpunk", VEC["m3"], 10, frag, params)
    assert res["cards"], "filtered search returned nothing"
    assert all(c["category"] == "ANIME" and c["rating"] >= 8.0
               for c in res["cards"].values()), res["cards"]
    print("  ok  category + rating filters apply inside both channels")


def test_tag_filter_uses_array_containment():
    """The new tag facets must actually filter, which the old comma-joined
    genre string could never do."""
    frag, params = store.build_filter(tags=["theme:cyberpunk"])
    res = store.hybrid_candidates("anything", VEC["m3"], 10, frag, params)
    got = {c["title"] for c in res["cards"].values()}
    assert got <= {"Ghost in the Shell", "Cyberpunk: Edgerunners"}, got
    assert got, "tag filter matched nothing"
    print(f"  ok  tag filter: {sorted(got)}")


def test_year_range_filter():
    frag, params = store.build_filter(year_min=2020, year_max=2025)
    res = store.hybrid_candidates("dune desert", VEC["m7"], 10, frag, params)
    assert all(2020 <= c["year"] <= 2025 for c in res["cards"].values())
    titles = {c["title"] for c in res["cards"].values()}
    assert "Dune" in titles
    print("  ok  year range filter")


# --- Title lookup -------------------------------------------------------------

def test_fuzzy_title_lookup_survives_typos():
    """Trigram similarity catches what exact/prefix matching never could."""
    assert any(c["title"] == "Blade Runner 2049"
               for c in store.title_candidates("blade runer 2049")), "typo not matched"
    assert any(c["title"] == "Paddington 2"
               for c in store.title_candidates("paddington")), "partial not matched"
    print("  ok  fuzzy title lookup handles typos and partials")


def test_both_dunes_coexist():
    """The old title-hashed id physically could not hold two films called Dune."""
    dunes = [c for c in store.title_candidates("Dune", limit=10) if c["title"] == "Dune"]
    years = sorted(c["year"] for c in dunes)
    assert years == [1984, 2021], years
    print(f"  ok  both Dunes present: {years}")


# --- Recommendation -----------------------------------------------------------

def test_neighbours_finds_similar_and_excludes_source():
    near = store.neighbours(["m3"], limit=4)
    ids = [c["id"] for c in near]
    assert "m3" not in ids, "source leaked into its own recommendations"
    assert "m2" in ids, f"the other cyberpunk anime should be near: {ids}"
    assert all("_cos" in c for c in near), "cosine must survive for scoring"
    print(f"  ok  neighbours: {[c['title'] for c in near[:3]]}")


def test_negative_feedback_pushes_away():
    """Dismissing something must demote it, or the dismiss button is decorative.

    The earlier version dismissed an item that already ranked last, so it could
    not have detected a no-op. Dismiss a HIGH-ranking neighbour and assert it
    actually moves down.
    """
    plain = [c["id"] for c in store.neighbours(["m3"], limit=6)]
    victim = plain[0]
    with_neg = [c["id"] for c in store.neighbours(["m3"], negative_ids=[victim], limit=6)]
    assert victim in with_neg, "negative should be demoted, not dropped entirely"
    assert with_neg.index(victim) > plain.index(victim), (
        f"{victim} did not move down: {plain} -> {with_neg}")
    print(f"  ok  negative feedback demotes {victim}: "
          f"{plain.index(victim)} -> {with_neg.index(victim)}")


# --- Quality ranking ----------------------------------------------------------

def test_top_by_quality_shrinks_low_vote_titles():
    """The 10/10 with four votes must not top the chart — and this now happens
    via the Bayesian prior in SQL, not a hardcoded 7.5-9.2 rating window."""
    top = store.top_by_quality(limit=5, min_votes=0)
    titles = [c["title"] for c in top]
    assert titles[0] != "Obscure Perfect Short", titles
    assert "Cyberpunk: Edgerunners" in titles[:3] or "Paddington 2" in titles[:3], titles
    print(f"  ok  quality ranking: {titles[:3]}")


# --- Facets -------------------------------------------------------------------

def test_facet_counts_power_a_filter_bar():
    facets = store.facet_counts("theme", limit=10)
    counts = {f["tag"]: f["count"] for f in facets}
    assert counts.get("theme:cyberpunk") == 2, counts
    assert all(f["label"] and ":" not in f["label"] for f in facets)
    eras = store.facet_counts("era")
    assert any(f["tag"] == "era:2010s" for f in eras), eras
    print(f"  ok  facet counts: {[(f['label'], f['count']) for f in facets[:3]]}")


# --- Detail and misc ----------------------------------------------------------

def test_detail_merges_the_heavy_blob():
    d = store.detail("m1")
    assert d["title"] == "Blade Runner 2049"
    assert d.get("cast"), "media_extra payload did not merge"
    assert store.detail("nope") is None
    print("  ok  detail merges media_extra")


def test_by_ids_preserves_order():
    got = [c["id"] for c in store.by_ids(["m4", "m1", "m3"])]
    assert got == ["m4", "m1", "m3"], got
    assert store.by_ids([]) == []
    assert store.by_ids(["ai-generated"]) == []
    print("  ok  by_ids preserves order and skips ai- ids")


def test_upsert_is_idempotent():
    """Re-running a build must update in place, never duplicate."""
    before = store.health()["titles"]
    rows = [{"id": "m1", "tmdb_id": 1, "tmdb_kind": "movie", "title": "Blade Runner 2049",
             "description": "updated", "category": "MOVIE", "genres": ["sci-fi"],
             "tags": [], "types": [], "forms": [], "rating": 7.6, "votes": 13000,
             "year_i": 2017}]
    store.upsert(rows, ["updated text"], [VEC["m1"]])
    assert store.health()["titles"] == before, "upsert duplicated a row"
    assert store.by_ids(["m1"])[0]["description"] == "updated"
    setup()
    print("  ok  upsert is idempotent")


def test_storage_report_measures_real_bytes():
    """So catalogue size is chosen from measurement, not from an estimate."""
    rep = store.storage_report()
    assert rep["titles"] == len(FIXTURES)
    assert rep["bytes_per_title"] > 0
    assert rep["projected_100k_mb"] > 0
    print(f"  ok  storage: {rep['bytes_per_title']} B/title "
          f"-> ~{rep['projected_100k_mb']} MB at 100k")


def main() -> int:
    print("Setting up fixtures against real Postgres…")
    setup()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    print(f"Running {len(tests)} store checks\n")
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
