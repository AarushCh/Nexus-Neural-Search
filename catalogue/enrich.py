"""
Outside sources that TMDB cannot provide.

AniList  — the only free, structured, RANK-WEIGHTED vibe taxonomy for anime.
IMDb     — ratings and vote counts on a scale TMDB's cannot match.

Both are free and neither needs a key.
"""

from __future__ import annotations

import csv
import gzip
import io
import time
from typing import Iterable

import requests

# --- AniList ------------------------------------------------------------------
#
# TMDB keywords say what a title is ABOUT. AniList tags say how strongly each
# theme applies, 0-100:
#
#   Cyberpunk: Edgerunners -> Cyberpunk 95, Dystopian 88, Tragedy 85,
#                             Guns 78, Male Protagonist 70, Body Horror 60
#
# Nothing else in media metadata gives you a weight, and the weight is what lets
# a 95 tag into the embedding while a 40 stays a filter-only facet.

ANILIST_URL = "https://graphql.anilist.co"

_QUERY = """
query ($page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: ANIME, sort: POPULARITY_DESC) {
      id idMal title { romaji english native }
      startDate { year }
      averageScore popularity
      tags { name rank isGeneralSpoiler isMediaSpoiler }
      studios(isMain: true) { nodes { name } }
    }
  }
}
"""


def fetch_anilist(max_pages: int = 120) -> list:
    """Most-popular-first anime with ranked tags. ~50 per page.

    AniList asks for <=90 requests/minute; the sleep keeps us well under and
    backs off on the documented 429.
    """
    out, page = [], 1
    while page <= max_pages:
        try:
            r = requests.post(ANILIST_URL, json={"query": _QUERY, "variables": {"page": page}},
                              timeout=30)
        except requests.RequestException:
            time.sleep(3)
            continue
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 60)))
            continue
        if r.status_code != 200:
            break
        payload = (r.json().get("data") or {}).get("Page") or {}
        media = payload.get("media") or []
        if not media:
            break
        out.extend(media)
        if not (payload.get("pageInfo") or {}).get("hasNextPage"):
            break
        page += 1
        time.sleep(0.75)
    return out


def index_anilist(rows: Iterable[dict]) -> dict:
    """Lookup keyed on every title variant, so TMDB rows can be matched by name.

    TMDB does not carry AniList or MAL ids, so title+year is the only join
    available. Indexing romaji, english and native separately is what makes it
    hit for shows known by different names in different places.
    """
    index = {}
    for m in rows:
        year = (m.get("startDate") or {}).get("year")
        for name in (m.get("title") or {}).values():
            if not name:
                continue
            key = _key(name)
            index.setdefault(key, m)
            if year:
                index.setdefault(f"{key}|{year}", m)
    return index


def match_anilist(index: dict, title: str, original_title: str = "", year=None) -> dict | None:
    """Year-qualified match first (it is much safer), then bare title."""
    for name in (title, original_title):
        if not name:
            continue
        k = _key(name)
        if year and f"{k}|{year}" in index:
            return index[f"{k}|{year}"]
    for name in (title, original_title):
        if name and _key(name) in index:
            return index[_key(name)]
    return None


def _key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


# --- IMDb ---------------------------------------------------------------------
#
# IMDb publishes ratings for ~1.4M titles as a bulk TSV. TMDB vote counts are
# often in the hundreds where IMDb has hundreds of thousands, which makes IMDb
# by far the better input to the Bayesian quality prior.

IMDB_RATINGS = "https://datasets.imdbws.com/title.ratings.tsv.gz"


def fetch_imdb_ratings(min_votes: int = 50) -> dict:
    """{imdb_id: {"rating": float, "votes": int}} for titles above a vote floor.

    The floor matters: the full file is ~1.4M rows and most of the tail is
    titles with single-digit vote counts that add nothing but memory.
    """
    try:
        r = requests.get(IMDB_RATINGS, timeout=300)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"⚠️  IMDb ratings unavailable, continuing with TMDB only: {e}")
        return {}
    out = {}
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
        reader = csv.DictReader(io.TextIOWrapper(gz, encoding="utf-8"), delimiter="\t")
        for row in reader:
            try:
                votes = int(row["numVotes"])
            except (KeyError, ValueError):
                continue
            if votes < min_votes:
                continue
            try:
                out[row["tconst"]] = {"rating": float(row["averageRating"]), "votes": votes}
            except (KeyError, ValueError):
                continue
    return out
