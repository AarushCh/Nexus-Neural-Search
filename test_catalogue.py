"""
Tests for the catalogue pipeline's pure logic.  python test_catalogue.py

No network: every test feeds a hand-built TMDB-shaped payload through the same
functions the real build uses, so the tagging and quality rules are verifiable
without spending a 100k-request harvest to find out they were wrong.
"""

import sys

from catalogue import tags as T
from catalogue.schema import build_record, document, passes_quality
from catalogue.tmdb import best_image, pick_poster


# --- Image quality ------------------------------------------------------------

def test_poster_selection_rejects_low_resolution():
    """w500 from a 300px source is an upscale. The gate is the whole point of
    fetching /images instead of trusting poster_path."""
    chosen = best_image([
        {"file_path": "/tiny.jpg", "width": 300, "height": 450, "vote_average": 9.9},
        {"file_path": "/good.jpg", "width": 1000, "height": 1500, "vote_average": 5.0},
    ], 0.667, 500)
    assert chosen == "/good.jpg", chosen
    print("  ok  poster selection rejects upscales even at high vote")


def test_poster_selection_rejects_wrong_aspect():
    """A 16:9 still stretched into a 2:3 card looks broken. Reject, don't crop."""
    chosen = best_image([
        {"file_path": "/wide.jpg", "width": 1920, "height": 1080, "vote_average": 9.0},
        {"file_path": "/right.jpg", "width": 680, "height": 1020, "vote_average": 3.0},
    ], 0.667, 500)
    assert chosen == "/right.jpg", chosen
    print("  ok  poster selection enforces 2:3 aspect")


def test_poster_selection_prefers_textless_and_popular():
    """Between two valid posters, community rating wins; a textless (null
    language) plate breaks a close tie because it has no burnt-in foreign title."""
    chosen = best_image([
        {"file_path": "/meh.jpg", "width": 1000, "height": 1500,
         "vote_average": 5.0, "vote_count": 3, "iso_639_1": "en"},
        {"file_path": "/loved.jpg", "width": 1000, "height": 1500,
         "vote_average": 8.4, "vote_count": 40, "iso_639_1": "en"},
    ], 0.667, 500)
    assert chosen == "/loved.jpg", chosen

    textless = best_image([
        {"file_path": "/en.jpg", "width": 1000, "height": 1500,
         "vote_average": 7.0, "vote_count": 10, "iso_639_1": "en"},
        {"file_path": "/none.jpg", "width": 1000, "height": 1500,
         "vote_average": 7.0, "vote_count": 10, "iso_639_1": None},
    ], 0.667, 500)
    assert textless == "/none.jpg", textless
    print("  ok  poster selection prefers well-rated and textless art")


def test_poster_falls_back_when_nothing_qualifies():
    """Never return no image just because /images was empty or all junk."""
    detail = {"images": {"posters": [{"file_path": "/bad.jpg", "width": 92, "height": 138}]},
              "poster_path": "/fallback.jpg"}
    assert pick_poster(detail) == "/fallback.jpg"
    assert pick_poster({"poster_path": "/only.jpg"}) == "/only.jpg"
    print("  ok  poster falls back to TMDB's primary")


# --- Tag system ---------------------------------------------------------------

def test_genre_vocabulary_is_unified():
    """TMDB uses different genre names for film and TV; AniList a third set.
    A genre filter is impossible unless they map onto one vocabulary."""
    assert T.normalise_genre("Science Fiction") == "sci-fi"
    assert T.normalise_genre("Sci-Fi & Fantasy") == "sci-fi"      # TV-only name
    assert T.normalise_genre("Action & Adventure") == "action"    # TV-only name
    assert T.normalise_genre("Action") == "action"
    assert T.normalise_genre("Not A Genre") is None
    print("  ok  film/TV/AniList genres normalise to one vocabulary")


def test_form_separates_anime_from_western_animation():
    """The old rule (genre 16 AND Japanese) mislabelled Japanese-studio shows
    commissioned in English, and had no home at all for Rick and Morty."""
    edgerunners = T.classify_form(
        {"origin_country": ["JP"], "original_language": "en"}, {"animation"}, None)
    assert edgerunners == ["anime"], edgerunners

    arcane = T.classify_form(
        {"origin_country": ["FR"], "original_language": "en"}, {"animation"}, None)
    assert arcane == ["western-animation"], arcane

    # An AniList match is definitive — that database only holds anime.
    assert T.classify_form({"origin_country": ["US"], "original_language": "en"},
                           {"animation"}, {"id": 1}) == ["anime"]

    assert T.classify_form({"origin_country": ["US"]}, {"drama"}, None) == ["live-action"]
    print("  ok  anime vs western-animation vs live-action")


def test_type_is_multi_label():
    """A documentary film is BOTH. The old single label made it a coin flip
    decided by which harvest stream arrived first."""
    doc_film = T.classify_type("movie", {"runtime": 95}, {"documentary"})
    assert "film" in doc_film and "documentary" in doc_film, doc_film
    assert "short" in T.classify_type("movie", {"runtime": 12}, set())
    assert "series" in T.classify_type("tv", {"number_of_seasons": 4}, set())
    print("  ok  type is multi-label (documentary film is both)")


def test_keywords_become_namespaced_theme_tags():
    """TMDB keywords are the curated vibe vocabulary the old build never fetched."""
    detail = {
        "keywords": {"keywords": [{"name": "dystopia"}, {"name": "artificial intelligence"},
                                  {"name": "woman director"}]},
        "release_date": "2017-10-04", "origin_country": ["US"],
        "original_language": "en", "credits": {"crew": [
            {"job": "Director", "name": "Denis Villeneuve"}]},
    }
    tags = T.build_tags("movie", detail, {"sci-fi"}, ["live-action"])
    assert "theme:dystopia" in tags
    assert "theme:artificial-intelligence" in tags
    # Production bookkeeping is not a theme.
    assert "theme:woman-director" not in tags, "stoplist should drop crew metadata"
    assert "people:denis-villeneuve" in tags, "director must be searchable"
    assert "era:2010s" in tags
    assert "genre:sci-fi" in tags
    print("  ok  keywords -> namespaced theme tags, stoplist applied")


def test_anilist_ranks_are_respected():
    """AniList ranks 0-100. Weak tags stay filterable but must not reach the
    embedding, or every show drifts toward every loosely-associated theme."""
    anilist = {"tags": [
        {"name": "Cyberpunk", "rank": 95},
        {"name": "Tragedy", "rank": 85},
        {"name": "Cute Girls", "rank": 20},                       # noise
        {"name": "Twist Ending", "rank": 90, "isMediaSpoiler": True},
    ]}
    tags = T.build_tags("tv", {"release_date": "2022-09-13"}, {"sci-fi"}, ["anime"], anilist)
    assert tags.get("theme:cyberpunk") == 95
    assert tags.get("theme:tragedy") == 85
    assert "theme:cute-girls" not in tags, "below the rank floor"
    assert "theme:twist-ending" not in tags, "spoiler tags must never be indexed"

    embedded = T.embed_tags(tags)
    assert "cyberpunk" in embedded and "tragedy" in embedded
    print(f"  ok  anilist ranks respected; embedded = {embedded}")


def test_embed_tags_strips_namespaces():
    """The embedding and BM25 index must see the word a user types, not
    'theme:cyberpunk'."""
    out = T.embed_tags({"theme:body-horror": 90, "genre:horror": 90, "where:netflix": 65})
    assert "body horror" in out
    assert not any(":" in w for w in out), out
    assert "netflix" not in out, "provider is a filter facet, not embedding text"
    print("  ok  embed_tags yields bare searchable words")


# --- Quality gates ------------------------------------------------------------

def test_quality_gates_reject_junk_and_explain_why():
    good = {"title": "X", "image": "u", "description": "d" * 60, "year": "2020",
            "votes": 500, "types": ["film"], "runtime": 100}
    assert passes_quality(good)[0]

    for field, value, expect in [
        ("title", "", "no title"),
        ("image", None, "no poster meeting quality bar"),
        ("description", "short", "overview too short"),
        ("year", "", "no release date"),
        ("votes", 2, "too few votes"),
        ("runtime", 2, "runtime too short"),
    ]:
        bad = dict(good, **{field: value})
        ok, why = passes_quality(bad)
        assert not ok and why == expect, (field, ok, why)
    print("  ok  quality gates reject junk with a stated reason")


# --- Record + document --------------------------------------------------------

DETAIL = {
    "id": 1, "title": "Blade Runner 2049", "original_title": "Blade Runner 2049",
    "overview": "A young blade runner discovers a secret that could plunge what is "
                "left of society into chaos, and leads him on a quest to find a "
                "former blade runner missing for thirty years.",
    "tagline": "There is an order to things.",
    "release_date": "2017-10-04", "runtime": 164,
    "vote_average": 7.6, "vote_count": 13000, "popularity": 90.0,
    "genres": [{"name": "Science Fiction"}, {"name": "Drama"}],
    "origin_country": ["US"], "original_language": "en",
    "poster_path": "/p.jpg", "backdrop_path": "/b.jpg",
    "keywords": {"keywords": [{"name": "dystopia"}, {"name": "neo-noir"}]},
    "credits": {"cast": [{"name": "Ryan Gosling", "character": "K", "profile_path": "/g.jpg"}],
                "crew": [{"job": "Director", "name": "Denis Villeneuve"}]},
    "production_companies": [{"name": "Alcon Entertainment"}],
    "external_ids": {"imdb_id": "tt1856101"},
    "videos": {"results": [{"site": "YouTube", "type": "Trailer",
                            "key": "gCcx85zbxz4", "official": True}]},
    "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": [
        {"certification": "R"}]}]},
}


def test_build_record_keeps_the_overview_clean():
    """Genres used to be appended to the overview as prose, which polluted the
    embedding and showed users the literal text 'Genres: Action, Drama.'"""
    rec = build_record("movie", DETAIL)
    assert "Genres:" not in rec["description"], rec["description"]
    assert rec["description"].startswith("A young blade runner")
    assert rec["genres"] == ["drama", "sci-fi"]
    assert rec["category"] == "MOVIE"
    assert rec["certification"] == "R"
    assert rec["trailer_key"] == "gCcx85zbxz4"
    assert rec["imdb_id"] == "tt1856101"
    print("  ok  build_record keeps overview clean, genres structured")


def test_document_exposes_every_search_surface():
    """The old embedded text was title + overview + category, so searching a
    director, an actor, a studio or a theme matched nothing."""
    doc = document(build_record("movie", DETAIL))
    for needle in ("Blade Runner 2049", "Denis Villeneuve", "Ryan Gosling",
                   "dystopia", "neo noir", "sci-fi", "Alcon"):
        assert needle.lower() in doc.lower(), f"{needle!r} missing from:\n{doc}"
    print("  ok  document exposes title, crew, cast, themes, genres, studio")


def test_record_is_deterministic():
    """Same input, same row — otherwise re-ingest churns the whole collection."""
    assert build_record("movie", DETAIL) == build_record("movie", DETAIL)
    print("  ok  build_record is deterministic")


def test_selection_keeps_every_category():
    """Regression: the final cut used raw votes, which erased documentaries
    entirely — they carry far fewer votes than blockbusters regardless of
    quality — and buried anything released in the last few years."""
    from catalogue.build import _select

    rows = []
    # Blockbusters: huge votes, ordinary ratings. These used to take every slot.
    for i in range(200):
        rows.append({"title": f"Blockbuster {i}", "category": "MOVIE",
                     "rating": 6.8, "votes": 900_000, "year_i": 2010})
    # Documentaries: excellent, but two orders of magnitude fewer votes.
    for i in range(50):
        rows.append({"title": f"Doc {i}", "category": "DOCUMENTARY",
                     "rating": 8.4, "votes": 4_000, "year_i": 2015})
    for i in range(50):
        rows.append({"title": f"Anime {i}", "category": "ANIME",
                     "rating": 8.2, "votes": 30_000, "year_i": 2018})
    for i in range(80):
        rows.append({"title": f"Series {i}", "category": "TV",
                     "rating": 7.9, "votes": 60_000, "year_i": 2019})

    picked = _select(rows, 100)
    cats = {}
    for r in picked:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    assert len(picked) == 100, len(picked)
    for c in ("MOVIE", "TV", "ANIME", "DOCUMENTARY"):
        assert cats.get(c, 0) > 0, f"{c} was wiped out: {cats}"
    assert cats["DOCUMENTARY"] >= 7, cats
    print(f"  ok  selection keeps every category: {cats}")


def test_recent_good_titles_beat_older_equals():
    """'Most recent too, if popular and good' — a new release has had less time
    to accumulate votes, so without a recency term it always loses to its own
    older equivalent."""
    from catalogue.build import _selection_score, _THIS_YEAR

    new = {"rating": 7.8, "votes": 40_000, "year_i": _THIS_YEAR}
    old = {"rating": 7.8, "votes": 40_000, "year_i": 1998}
    assert _selection_score(new) > _selection_score(old)

    # But recency must not outrank real quality: a weak new film still loses.
    weak_new = {"rating": 5.2, "votes": 3_000, "year_i": _THIS_YEAR}
    strong_old = {"rating": 8.6, "votes": 500_000, "year_i": 1994}
    assert _selection_score(strong_old) > _selection_score(weak_new)
    print("  ok  recency lifts new titles without beating quality")


def test_adult_content_is_dropped_without_eating_mainstream_anime():
    """Enumerating the anime slice from /discover surfaces hentai next to
    Doraemon, and TMDB's `adult` flag does not catch it. The gate keys on
    unambiguous tags only: `ecchi`/`erotic` sit on Mushoku Tensei, a mainstream
    hit, so treating either as a disqualifier would throw away top-tier anime.
    """
    from catalogue.schema import passes_quality

    base = {"title": "T", "image": "http://x/p.jpg", "year": "2020",
            "votes": 5000, "description": "x" * 60, "types": ["series"]}

    porn = dict(base, tags=["genre:animation", "theme:hentai", "theme:ecchi"])
    ok, why = passes_quality(porn)
    assert not ok and why == "adult content", (ok, why)

    fan_service = dict(base, tags=["genre:animation", "theme:ecchi", "theme:erotic"])
    ok, why = passes_quality(fan_service)
    assert ok, f"mainstream anime wrongly dropped: {why}"

    clean = dict(base, tags=["genre:animation", "theme:friendship"])
    assert passes_quality(clean)[0]
    print("  ok  adult gate drops porn, keeps mainstream anime")


def test_alt_titles_rank_abbreviations_first():
    """Regression: TMDB returns long official variants first, so the [:8] cut
    kept seven restatements of the primary title and dropped "AOT" — the one
    string a person would actually type into a search box."""
    from catalogue.schema import build_record

    detail = {
        "id": 1429, "name": "Attack on Titan", "overview": "x" * 60,
        "first_air_date": "2013-04-07", "vote_average": 8.6, "vote_count": 6000,
        "alternative_titles": {"results": [
            {"iso_3166_1": "US", "title": "Attack on Titan: The Final Season"},
            {"iso_3166_1": "US", "title": "Attack on Titan: The Final Chapters Part 2"},
            {"iso_3166_1": "US", "title": "Attack on Titan: No Regrets"},
            {"iso_3166_1": "US", "title": "Attack on Titan: Lost Girls"},
            {"iso_3166_1": "US", "title": "Attack on Titan: Shingeki no Kyojin"},
            {"iso_3166_1": "US", "title": "Attack on Titan: The Final Season Part 3"},
            {"iso_3166_1": "US", "title": "Attack on Titan: Chronicle"},
            {"iso_3166_1": "US", "title": "AOT"},
        ]},
    }
    rec = build_record("tv", detail)
    assert rec["alt_titles"][0] == "AOT", rec["alt_titles"]
    print("  ok  alt titles put the abbreviation first")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    print(f"Running {len(tests)} catalogue checks\n")
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
