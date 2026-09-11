"""
Tag extraction and normalisation.

Replaces the old tag "system", which was: join TMDB's genre names with commas and
glue the string onto the end of the overview.

    description = f"{overview} Genres: {gs}."

That single line caused three separate problems. The genre words polluted every
embedding and inflated BM25 for common terms; users saw the literal text
"Genres: Action, Drama." in card blurbs; and because it was prose rather than
data, nothing could filter, count, or click on it.

The replacement is four layers, all multi-label:

    TYPE     film | series | limited-series | special | short | documentary
    FORM     live-action | anime | western-animation | stop-motion
    GENRE    ~25 controlled terms, normalised across TMDB and AniList
    TAGS     open vocabulary, namespaced and rank-weighted

Namespaced tags are what the UI groups, colours and filters on, and what feeds
the retrieval document. Every tag looks like `namespace:slug`.
"""

from __future__ import annotations

import re
import unicodedata

# --- Slugging -----------------------------------------------------------------

def slug(text: str) -> str:
    """'Sci-Fi & Fantasy' -> 'sci-fi-fantasy'. Stable, ASCII, url-safe."""
    s = unicodedata.normalize("NFKD", str(text))
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s


def tag(namespace: str, value: str) -> str | None:
    s = slug(value)
    return f"{namespace}:{s}" if s else None


# --- Genre vocabulary ---------------------------------------------------------
# TMDB uses different genre sets for film and TV ("Sci-Fi & Fantasy" and
# "Action & Adventure" exist only on TV), and AniList uses a third set. Mapping
# them all onto one controlled list is what makes a genre filter possible at all.

GENRE_MAP = {
    "action": "action", "action-adventure": "action", "adventure": "adventure",
    "animation": "animation", "comedy": "comedy", "crime": "crime",
    "documentary": "documentary", "drama": "drama", "family": "family",
    "fantasy": "fantasy", "history": "history", "horror": "horror",
    "music": "music", "mystery": "mystery", "romance": "romance",
    "science-fiction": "sci-fi", "sci-fi-fantasy": "sci-fi", "sci-fi": "sci-fi",
    "tv-movie": "tv-movie", "thriller": "thriller", "war": "war",
    "war-politics": "war", "western": "western", "kids": "family",
    "soap": "drama", "talk": "talk", "news": "documentary", "reality": "reality",
    # AniList-only genres
    "mecha": "mecha", "psychological": "thriller", "slice-of-life": "slice-of-life",
    "sports": "sport", "supernatural": "fantasy", "ecchi": "ecchi",
    "mahou-shoujo": "fantasy", "hentai": "adult",
}

# TMDB keywords include bookkeeping entries that describe the production rather
# than the story. They are noise in a theme vocabulary.
KEYWORD_STOPLIST = {
    "woman-director", "based-on-novel-or-book", "based-on-comic",
    "based-on-true-story", "based-on-video-game", "based-on-manga",
    "based-on-young-adult-novel", "based-on-play-or-musical", "remake",
    "sequel", "prequel", "reboot", "live-action-remake", "anime", "aftercreditsstinger",
    "duringcreditsstinger", "independent-film", "short-film", "silent-film",
    "3d", "imax", "b-movie", "cult-film",
}

# Keywords that are really audience/format signals, routed to their own namespace
# so they can be filtered separately from themes.
AUDIENCE_KEYWORDS = {
    "shounen": "shounen", "shoujo": "shoujo", "seinen": "seinen", "josei": "josei",
    "kids": "family", "family-friendly": "family",
}

# Certification -> a coarse audience band. Regional rating systems disagree on
# labels but agree on intent.
CERTIFICATION_AUDIENCE = {
    "G": "family", "TV-G": "family", "TV-Y": "family", "TV-Y7": "family", "U": "family",
    "PG": "family", "TV-PG": "family", "UA": "teen", "PG-13": "teen", "TV-14": "teen",
    "12": "teen", "12A": "teen", "15": "mature", "R": "mature", "TV-MA": "mature",
    "16": "mature", "18": "adult", "NC-17": "adult", "A": "adult", "X": "adult",
}

# AniList ranks each tag 0-100 for how strongly it applies. Below this it is
# noise; above it the tag genuinely characterises the show.
ANILIST_TAG_FLOOR = 55

# Only tags at least this strong go into the embedded document. Weaker ones stay
# filterable but must not dilute the vector.
EMBED_TAG_FLOOR = 70


def normalise_genre(name: str) -> str | None:
    return GENRE_MAP.get(slug(name))


# --- Layer 1 & 2: type and form ----------------------------------------------

def classify_type(kind: str, detail: dict, genre_slugs: set) -> list:
    """Structural type. Multi-label: a documentary film is BOTH film and documentary.

    The old scheme forced one label from {MOVIE, TV, ANIME, DOCUMENTARY} and
    assigned it by which harvest stream happened to reach the title first, so a
    documentary about anime landed wherever the race finished.
    """
    out = []
    if kind == "movie":
        out.append("film")
        runtime = detail.get("runtime") or 0
        if 0 < runtime < 40:
            out.append("short")
    else:
        tv_types = {slug(t) for t in [detail.get("type") or ""]}
        if "miniseries" in tv_types or (detail.get("number_of_seasons") or 0) == 1 and \
                (detail.get("number_of_episodes") or 0) <= 10:
            out.append("limited-series")
        out.append("series")
    if "documentary" in genre_slugs:
        out.append("documentary")
    return out


def classify_form(detail: dict, genre_slugs: set, anilist: dict = None) -> list:
    """How it was made. Animation is not one thing: anime and western animation
    are different products and users search for them differently.

    The previous rule was "TMDB genre 16 AND Japanese origin", which mislabelled
    Japanese-studio shows commissioned in English and had no category at all for
    Rick and Morty or Bojack Horseman.
    """
    countries = {str(c).upper() for c in (detail.get("origin_country") or [])}
    lang = str(detail.get("original_language") or "").lower()
    if not countries:
        countries = {str((c or {}).get("iso_3166_1", "")).upper()
                     for c in (detail.get("production_countries") or [])}

    if "animation" not in genre_slugs:
        return ["live-action"]
    # An AniList match is definitive: that database only holds anime.
    if anilist or lang == "ja" or "JP" in countries:
        return ["anime"]
    return ["western-animation"]


# --- Layer 4: namespaced tags -------------------------------------------------

def build_tags(kind: str, detail: dict, genre_slugs: set, forms: list,
               anilist: dict = None, certification: str = None) -> dict:
    """All namespaced tags for one title, mapped to a 0-100 strength.

    Strength drives two different things: which tags are strong enough to enter
    the embedded document (EMBED_TAG_FLOOR), and how the UI orders the chips it
    shows on a card.
    """
    tags: dict = {}

    def add(ns: str, value, weight: int):
        t = tag(ns, value)
        if t:
            tags[t] = max(tags.get(t, 0), int(weight))

    # Curated TMDB keywords — the closest thing to a real theme vocabulary that
    # exists for film and TV, and previously not fetched at all.
    kw_block = detail.get("keywords") or {}
    for kw in (kw_block.get("keywords") or kw_block.get("results") or []):
        name = kw.get("name") or ""
        s = slug(name)
        if not s or s in KEYWORD_STOPLIST:
            continue
        if s in AUDIENCE_KEYWORDS:
            add("audience", AUDIENCE_KEYWORDS[s], 80)
        else:
            add("theme", name, 75)

    # AniList tags carry a real strength rank, so they can be weighted honestly
    # instead of being flattened to "present".
    for t in (anilist or {}).get("tags", []):
        rank = int(t.get("rank") or 0)
        if rank >= ANILIST_TAG_FLOOR and not t.get("isGeneralSpoiler") and not t.get("isMediaSpoiler"):
            add("theme", t.get("name", ""), rank)

    # Era, from the release date.
    year = _year(detail)
    if year:
        add("era", f"{(year // 10) * 10}s", 60)

    # Origin: country and language are how people ask for "Korean thriller" or
    # "French New Wave", and neither was stored before.
    for c in (detail.get("origin_country") or []):
        add("origin", c, 60)
    for c in (detail.get("production_countries") or [])[:3]:
        add("origin", c.get("iso_3166_1", ""), 55)
    if detail.get("original_language"):
        add("lang", detail["original_language"], 60)

    # People: director/creator, so "Villeneuve sci-fi" and "Miyazaki" resolve.
    for person in _directors(kind, detail):
        add("people", person, 85)

    # Studio and network: "A24 horror", "Ghibli", "HBO drama".
    for co in (detail.get("production_companies") or [])[:4]:
        add("studio", co.get("name", ""), 70)
    for net in (detail.get("networks") or [])[:3]:
        add("studio", net.get("name", ""), 70)

    # Franchise, so sequels and universes group.
    coll = detail.get("belongs_to_collection") or {}
    if coll.get("name"):
        add("franchise", re.sub(r"\s+Collection$", "", coll["name"]), 90)

    # Where to watch — a filter people genuinely want ("what's on Netflix").
    for region, block in (detail.get("watch/providers") or {}).get("results", {}).items():
        if region not in ("US", "GB", "IN"):
            continue
        for bucket in ("flatrate", "free", "ads"):
            for p in block.get(bucket, []):
                add("where", p.get("provider_name", ""), 65)

    if certification:
        band = CERTIFICATION_AUDIENCE.get(certification.upper())
        if band:
            add("audience", band, 75)
        add("rated", certification, 70)

    for f in forms:
        add("form", f, 95)
    for g in genre_slugs:
        add("genre", g, 90)

    return tags


def embed_tags(tags: dict, limit: int = 10) -> list:
    """The strongest tags, as bare words, for the retrieval document.

    Namespaces are stripped here: the embedding should see "cyberpunk", not
    "theme:cyberpunk", and BM25 should match the word a user actually types.
    """
    strong = [(w, t) for t, w in tags.items()
              if w >= EMBED_TAG_FLOOR and t.split(":", 1)[0] in
              ("theme", "genre", "mood", "style", "setting", "form")]
    strong.sort(reverse=True)
    seen, out = set(), []
    for _, t in strong:
        word = t.split(":", 1)[1].replace("-", " ")
        if word not in seen:
            seen.add(word)
            out.append(word)
        if len(out) >= limit:
            break
    return out


# --- helpers ------------------------------------------------------------------

def _year(detail: dict) -> int | None:
    date = detail.get("release_date") or detail.get("first_air_date") or ""
    return int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None


def _directors(kind: str, detail: dict) -> list:
    if kind == "tv":
        return [c.get("name", "") for c in (detail.get("created_by") or [])[:3]]
    crew = (detail.get("credits") or {}).get("crew") or []
    return [c.get("name", "") for c in crew if c.get("job") == "Director"][:3]
