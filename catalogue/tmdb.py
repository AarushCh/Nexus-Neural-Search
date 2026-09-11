"""
TMDB access layer: enumeration, detail fetch, and image selection.

The important thing here is `iter_export_ids`.

The old harvester paged `/discover`, which TMDB caps at **page 500**. Six streams
x 500 pages x 20 results is a hard ceiling of ~60k raw candidates before any
filtering, and after the vote threshold and dedupe it produced ~6,850 titles.
No `--pages` value could ever get past that: the ceiling is structural, not a
setting.

TMDB publishes a daily gzipped JSONL export of EVERY id it holds (~1M movies,
~200k series). Reading that and fetching details by id removes the wall
entirely, which is what makes a 100k+ catalogue possible.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Iterator

import requests

BASE = "https://api.themoviedb.org/3"
EXPORTS = "https://files.tmdb.org/p/exports"
IMG = "https://image.tmdb.org/t/p"

API_KEY = os.getenv("TMDB_API_KEY")

# TMDB tolerates roughly 50 requests/second. Stay under it: a 429 storm costs
# far more wall-clock time than pacing does.
RATE_LIMIT = float(os.getenv("TMDB_RATE_LIMIT", 40))

# Everything worth having in one request. TMDB allows up to 20 appended
# namespaces; `images` is what makes real poster selection possible, and
# `keywords` is the curated theme vocabulary the tag system is built on.
MOVIE_APPEND = "videos,credits,keywords,images,watch/providers,alternative_titles,external_ids,release_dates"
TV_APPEND = "videos,credits,keywords,images,watch/providers,alternative_titles,external_ids,content_ratings"


class RateLimiter:
    """Token bucket shared across worker threads."""

    def __init__(self, per_second: float):
        self.interval = 1.0 / max(per_second, 1.0)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._next < now:
                self._next = now
            delay = self._next - now
            self._next += self.interval
        if delay > 0:
            time.sleep(delay)


_limiter = RateLimiter(RATE_LIMIT)
_local = threading.local()


def _session() -> requests.Session:
    s = getattr(_local, "s", None)
    if s is None:
        s = _local.s = requests.Session()
        s.headers["Accept"] = "application/json"
    return s


def get(path: str, params: dict = None, tries: int = 4):
    """GET with pacing, retry, and 429 backoff. Returns parsed JSON or None."""
    params = {"api_key": API_KEY, **(params or {})}
    for attempt in range(tries):
        _limiter.wait()
        try:
            r = _session().get(f"{BASE}/{path}", params=params, timeout=20)
        except requests.RequestException:
            time.sleep(1 + attempt * 2)
            continue
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 2)))
            continue
        if r.status_code == 404:
            return None          # deleted/merged title: not an error worth retrying
        if r.status_code >= 500:
            time.sleep(1 + attempt * 2)
            continue
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None
    return None


# --- Enumeration --------------------------------------------------------------

def iter_export_ids(kind: str, min_popularity: float = 0.0) -> Iterator[dict]:
    """Stream every id TMDB knows about, from the daily export dump.

    `kind` is "movie" or "tv". Yields {"id", "popularity", "original_title"}.
    Falls back through recent days because the current day's file is not
    published until ~08:00 UTC.
    """
    name = {"movie": "movie_ids", "tv": "tv_series_ids"}[kind]
    for days_back in range(0, 5):
        day = datetime.now(timezone.utc) - timedelta(days=days_back)
        url = f"{EXPORTS}/{name}_{day.strftime('%m_%d_%Y')}.json.gz"
        try:
            r = requests.get(url, timeout=120, stream=True)
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
            for line in io.TextIOWrapper(gz, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("adult"):
                    continue
                if float(row.get("popularity") or 0) < min_popularity:
                    continue
                yield row
        return
    raise RuntimeError(f"Could not download the TMDB {kind} id export (tried 5 days).")


def fetch_detail(kind: str, tmdb_id: int) -> dict | None:
    """Full record for one title, with every append we need, in one request."""
    return get(
        f"{kind}/{tmdb_id}",
        {
            "append_to_response": MOVIE_APPEND if kind == "movie" else TV_APPEND,
            # Textless (null) and English artwork only — keeps the images block
            # small and is exactly the set we want to choose from.
            "include_image_language": "en,null",
        },
    )


# --- Image selection ----------------------------------------------------------
#
# The single biggest visible quality problem in a catalogue like this is bad
# artwork: upscaled posters, fan edits with burnt-in text, 4:3 crops stretched
# into a 2:3 card. `poster_path` from the detail response is whatever TMDB
# currently considers primary, which is often none of the best options.
# /images returns every candidate with dimensions and community votes, so we can
# actually choose.

POSTER_ASPECT = 0.667            # 2:3, the standard poster shape
POSTER_MIN_WIDTH = 500           # below this, w500 is an upscale
BACKDROP_MIN_WIDTH = 1280


def _score_image(img: dict, target_aspect: float, min_width: int) -> float | None:
    """Rank one artwork candidate. None = fails the quality floor."""
    width = int(img.get("width") or 0)
    height = int(img.get("height") or 0)
    if width < min_width or height <= 0:
        return None
    aspect = width / height
    # Reject anything noticeably the wrong shape rather than letting CSS crop it.
    if abs(aspect - target_aspect) / target_aspect > 0.12:
        return None
    votes = float(img.get("vote_average") or 0)
    count = int(img.get("vote_count") or 0)
    # Community rating leads, then confidence in that rating, then resolution.
    # Textless art (iso_639_1 is null) gets a nudge: no burnt-in foreign title.
    textless = 0.35 if img.get("iso_639_1") in (None, "") else 0.0
    return votes + min(count, 20) * 0.05 + min(width, 4000) / 20000.0 + textless


def best_image(images: list, target_aspect: float, min_width: int) -> str | None:
    """Highest-quality artwork path from an /images list, or None if none qualify."""
    ranked = []
    for img in images or []:
        s = _score_image(img, target_aspect, min_width)
        if s is not None:
            ranked.append((s, img.get("file_path")))
    if not ranked:
        return None
    ranked.sort(key=lambda t: t[0], reverse=True)
    return ranked[0][1]


def pick_poster(detail: dict) -> str | None:
    """Best poster path, falling back to TMDB's primary choice."""
    block = (detail.get("images") or {}).get("posters") or []
    return best_image(block, POSTER_ASPECT, POSTER_MIN_WIDTH) or detail.get("poster_path")


def pick_backdrop(detail: dict) -> str | None:
    block = (detail.get("images") or {}).get("backdrops") or []
    return best_image(block, 16 / 9, BACKDROP_MIN_WIDTH) or detail.get("backdrop_path")


def img_url(path: str | None, size: str) -> str | None:
    return f"{IMG}/{size}{path}" if path else None
