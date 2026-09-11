"""
Record construction: identity, quality gates, and the text that gets embedded.

Three things here decide catalogue quality more than anything downstream:

  `stable_id`      — what counts as one title
  `passes_quality` — what is allowed in
  `document`       — what the retrieval models actually see
"""

from __future__ import annotations

import re
import uuid

from catalogue import tags as T
from catalogue.tmdb import img_url, pick_backdrop, pick_poster

# --- Identity -----------------------------------------------------------------

def stable_id(kind: str, tmdb_id: int) -> str:
    """Deterministic point id keyed on TMDB identity.

    This used to hash the lowercased TITLE, so Dune (1984) and Dune (2021), The
    Office (UK) and The Office (US), and every remake ever made collapsed onto a
    single point. The catalogue could not physically hold both, and the builder
    made it worse by keeping only the higher-voted one.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"tmdb:{kind}:{int(tmdb_id)}"))


def dedupe_key(title: str, year) -> tuple:
    """Collapse true duplicates without collapsing remakes (title alone did)."""
    return (re.sub(r"[^a-z0-9]+", "", str(title).lower()), str(year or ""))


# --- Quality gates ------------------------------------------------------------
#
# Tuned to keep the catalogue large but clean. The failure mode to avoid is not
# "too few titles", it is a search result full of untranslated stubs with no
# artwork, which is what a naive "ingest everything" pass produces.

MIN_OVERVIEW = 40        # characters; below this the embedding has nothing to learn
MIN_VOTES = 8            # some evidence a real audience exists
MIN_RUNTIME = 5          # minutes; filters trailers and clips mis-filed as films

# Tag slugs that mark a title as adult. TMDB's `adult` boolean is unreliable
# for animation, so enumerating the anime slice directly surfaces hentai next
# to Doraemon. Mainstream erotic thrillers are not at risk here: Basic Instinct
# carries no TMDB keywords at all, so it cannot trip this.
# Unambiguous markers only. "ecchi" and "erotic" are NOT here: both sit on
# Mushoku Tensei, a mainstream hit, so either as a sole disqualifier throws away
# top-tier anime. Precision over recall — a wrongly dropped favourite is a worse
# failure for a discovery engine than a borderline title that slipped through.
ADULT_TAGS = {"theme:hentai", "theme:softcore", "theme:animated-porn",
              "theme:pornography", "theme:porn", "theme:sexploitation",
              "theme:adult-video"}


def passes_quality(rec: dict, min_votes: int = MIN_VOTES) -> tuple[bool, str]:
    """(ok, reason). The reason is kept so the build can report WHY it dropped
    things — a silent filter is impossible to tune."""
    if not rec.get("title"):
        return False, "no title"
    if ADULT_TAGS & set(rec.get("tags") or ()):
        return False, "adult content"
    if not rec.get("image"):
        return False, "no poster meeting quality bar"
    if len(rec.get("description") or "") < MIN_OVERVIEW:
        return False, "overview too short"
    if not rec.get("year"):
        return False, "no release date"
    if int(rec.get("votes") or 0) < min_votes:
        return False, "too few votes"
    if "film" in (rec.get("types") or []) and 0 < (rec.get("runtime") or 0) < MIN_RUNTIME:
        return False, "runtime too short"
    return True, ""


# --- Record -------------------------------------------------------------------

def build_record(kind: str, detail: dict, anilist: dict = None,
                 imdb: dict = None) -> dict | None:
    """Turn one TMDB detail response into a catalogue row."""
    title = detail.get("title") or detail.get("name") or ""
    if not title:
        return None

    date = detail.get("release_date") or detail.get("first_air_date") or ""
    year = int(date[:4]) if len(date) >= 4 and date[:4].isdigit() else None

    genre_slugs = {g for g in (T.normalise_genre(x.get("name", ""))
                               for x in (detail.get("genres") or [])) if g}
    forms = T.classify_form(detail, genre_slugs, anilist)
    types = T.classify_type(kind, detail, genre_slugs)
    certification = _certification(kind, detail)
    tag_map = T.build_tags(kind, detail, genre_slugs, forms, anilist, certification)

    poster = pick_poster(detail)
    backdrop = pick_backdrop(detail)

    cast = [{"name": c.get("name"), "character": c.get("character"),
             "profile": img_url(c.get("profile_path"), "w185")}
            for c in ((detail.get("credits") or {}).get("cast") or [])[:15]]
    crew = _key_crew(kind, detail)

    providers, seen = [], set()
    for bucket in ("flatrate", "free", "ads", "rent", "buy"):
        block = (detail.get("watch/providers") or {}).get("results", {}).get("US", {})
        for p in block.get(bucket, []):
            n = p.get("provider_name")
            if n and n not in seen:
                seen.add(n)
                providers.append({"name": n, "logo": img_url(p.get("logo_path"), "w92")})

    rating = round(float(detail.get("vote_average") or 0), 1)
    votes = int(detail.get("vote_count") or 0)
    # IMDb votes dwarf TMDB's and are a better popularity signal where present.
    if imdb:
        rating = round(float(imdb.get("rating") or rating), 1)
        votes = max(votes, int(imdb.get("votes") or 0))

    rec = {
        "tmdb_id": int(detail["id"]), "tmdb_kind": kind,
        "imdb_id": (detail.get("external_ids") or {}).get("imdb_id"),
        "title": title,
        "original_title": detail.get("original_title") or detail.get("original_name") or "",
        "alt_titles": _alt_titles(detail),
        # The overview stays CLEAN. Genres used to be appended to it as prose,
        # which polluted every embedding and showed up in card blurbs as the
        # literal string "Genres: Action, Drama."
        "description": (detail.get("overview") or "").strip(),
        "tagline": (detail.get("tagline") or "").strip(),
        "image": img_url(poster, "w500"),
        "image_sm": img_url(poster, "w342"),
        "backdrop": img_url(backdrop, "w1280"),
        "types": types, "forms": forms, "genres": sorted(genre_slugs),
        "tags": sorted(tag_map),
        "tag_weights": tag_map,
        # Kept for the existing UI and filters, which expect one coarse bucket.
        "category": _legacy_category(types, forms),
        "type": _legacy_category(types, forms).title(),
        "genre": ", ".join(sorted(genre_slugs)),
        "rating": rating, "rating_f": rating,
        "votes": votes, "pop": float(detail.get("popularity") or 0),
        "year": str(year or ""), "year_i": year or 0,
        "release_date": date,
        "runtime": detail.get("runtime") or (detail.get("episode_run_time") or [None])[0],
        "seasons": detail.get("number_of_seasons"),
        "episodes": detail.get("number_of_episodes"),
        "status": detail.get("status"),
        "original_language": detail.get("original_language"),
        "certification": certification,
        "trailer_key": _trailer(detail),
        "cast": cast, "crew": crew, "providers": providers,
        "anilist_id": (anilist or {}).get("id"),
    }
    return rec


def document(rec: dict) -> str:
    """The text that gets embedded AND indexed for BM25.

    Previously this was `f"{title}. {description} {category}"`, so a search for
    a director, an actor, a studio or a theme could not match anything — cast
    was stored in the payload but never indexed, and there were no themes at all.

    Every line below is a retrieval surface someone actually searches on.
    """
    parts = [f"{rec['title']} ({rec.get('original_title') or rec['title']}) "
             f"[{rec.get('year') or 'n/a'}]"]
    if rec.get("alt_titles"):
        parts.append("Also known as: " + ", ".join(rec["alt_titles"][:6]))
    if rec.get("tagline"):
        parts.append(rec["tagline"])
    if rec.get("description"):
        parts.append(rec["description"])
    if rec.get("genres"):
        parts.append("Genres: " + ", ".join(rec["genres"]))
    themes = T.embed_tags(rec.get("tag_weights") or {})
    if themes:
        parts.append("Themes: " + ", ".join(themes))
    directors = [c["name"] for c in rec.get("crew", []) if c.get("job") == "Director"]
    if directors:
        parts.append("Directed by " + ", ".join(directors) + ".")
    names = [c["name"] for c in rec.get("cast", [])[:6] if c.get("name")]
    if names:
        parts.append("Starring " + ", ".join(names) + ".")
    tail = [t.split(":", 1)[1].replace("-", " ")
            for t in rec.get("tags", []) if t.startswith(("studio:", "origin:", "franchise:"))]
    if tail:
        parts.append(" · ".join(sorted(set(tail))[:6]))
    return "\n".join(parts)


# --- helpers ------------------------------------------------------------------

def _legacy_category(types: list, forms: list) -> str:
    """Collapse the multi-label model back to the single bucket the current UI
    filter expects. Anime wins over the film/series split because that is how
    people browse it."""
    if "anime" in forms:
        return "ANIME"
    if "documentary" in types:
        return "DOCUMENTARY"
    if "series" in types or "limited-series" in types:
        return "TV"
    return "MOVIE"


def _trailer(detail: dict) -> str | None:
    vids = (detail.get("videos") or {}).get("results", [])
    for typ, official in (("Trailer", True), ("Trailer", False), ("Teaser", False)):
        for v in vids:
            if v.get("site") == "YouTube" and v.get("type") == typ and \
                    (v.get("official") or not official):
                return v.get("key")
    return None


def _alt_titles(detail: dict) -> list:
    """English alternative titles, so abbreviations and localisations resolve
    ("AoT", "SnK", "Shingeki no Kyojin" all reaching Attack on Titan)."""
    block = detail.get("alternative_titles") or {}
    rows = block.get("titles") or block.get("results") or []
    out, seen = [], set()
    primary = str(detail.get("title") or detail.get("name") or "").lower()
    for r in rows:
        if r.get("iso_3166_1") not in ("US", "GB", "CA", "AU", None, ""):
            continue
        t = (r.get("title") or "").strip()
        if t and t.lower() != primary and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:8]


def _key_crew(kind: str, detail: dict) -> list:
    if kind == "tv":
        return [{"name": c.get("name"), "job": "Director"}
                for c in (detail.get("created_by") or [])[:3]]
    wanted = {"Director", "Screenplay", "Writer", "Original Music Composer",
              "Director of Photography"}
    out, seen = [], set()
    for c in ((detail.get("credits") or {}).get("crew") or []):
        job = c.get("job")
        if job in wanted and (c.get("name"), job) not in seen:
            seen.add((c.get("name"), job))
            out.append({"name": c.get("name"), "job": job})
    return out[:8]


def _certification(kind: str, detail: dict) -> str | None:
    if kind == "tv":
        for r in (detail.get("content_ratings") or {}).get("results", []):
            if r.get("iso_3166_1") == "US" and r.get("rating"):
                return r["rating"]
        return None
    for r in (detail.get("release_dates") or {}).get("results", []):
        if r.get("iso_3166_1") != "US":
            continue
        for d in r.get("release_dates", []):
            if d.get("certification"):
                return d["certification"]
    return None
