"""
Catalogue builder — four resumable stages.

    python -m catalogue.build ids        # enumerate every TMDB id worth fetching
    python -m catalogue.build fetch      # download details (resumable, cached)
    python -m catalogue.build normalise  # join, tag, quality-gate -> catalogue.jsonl
    python -m catalogue.build index      # embed + upsert into Qdrant
    python -m catalogue.build all        # run the lot

Why stages instead of one pass:

The previous builder did discover -> enrich -> embed -> upsert in a single
process. If it died at 90% you started over and re-spent the whole API budget,
and there was no way to re-tag the catalogue without re-downloading it. Each
stage here writes to disk, so `fetch` is paid once and `normalise` can be re-run
as many times as the tag rules change, for free.

Options:
    --target 120000     how many titles to aim for
    --min-votes 8       quality floor
    --workers 24        fetch concurrency
    --no-imdb           skip the IMDb ratings join
    --no-anilist        skip AniList tags
    --recreate          drop and rebuild the Qdrant collection
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from catalogue import enrich, tmdb                     # noqa: E402
from catalogue.schema import build_record, dedupe_key, document, passes_quality, stable_id  # noqa: E402
from backend.ranking import quality_prior                # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_THIS_YEAR = time.gmtime().tm_year
DATA = Path(os.getenv("CATALOGUE_DIR", "data"))
IDS_FILE = DATA / "ids.jsonl"
RAW_FILE = DATA / "raw.jsonl.gz"
CATALOGUE_FILE = DATA / "catalogue.jsonl"


def _arg(flag: str, default):
    if flag in sys.argv:
        return type(default)(sys.argv[sys.argv.index(flag) + 1])
    return default


TARGET = _arg("--target", 120000)
MIN_VOTES = _arg("--min-votes", 8)
WORKERS = _arg("--workers", 24)
RECREATE = "--recreate" in sys.argv
NO_IMDB = "--no-imdb" in sys.argv
NO_ANILIST = "--no-anilist" in sys.argv


# --- Stage 1: ids -------------------------------------------------------------

def stage_ids() -> None:
    """Enumerate candidate ids from the daily TMDB export.

    Sorted by popularity and truncated to a generous multiple of the target,
    because roughly half of any slice fails the quality gates in `normalise`.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    # Skew toward film: TMDB holds far more movies than series, and a catalogue
    # of only long-tail TV reads badly.
    plan = [("movie", int(TARGET * 1.6), 0.6), ("tv", int(TARGET * 0.7), 0.5)]
    rows = []
    for kind, want, min_pop in plan:
        got = []
        print(f"📇 Enumerating {kind} ids from the TMDB daily export…")
        for row in tmdb.iter_export_ids(kind, min_popularity=min_pop):
            got.append({"kind": kind, "id": row["id"],
                        "pop": float(row.get("popularity") or 0)})
        got.sort(key=lambda r: r["pop"], reverse=True)
        rows += got[:want]
        print(f"   {kind}: {len(got)} above popularity {min_pop}, keeping {min(want, len(got))}")

    # Anime is ~7% of a popularity-ranked catalogue but a headline category
    # here, and global popularity buries everything past its few breakouts. Ask
    # TMDB for the slice directly, so the pool actually contains enough of it
    # for the quota in `normalise` to have something to choose from.
    seen = {(r["kind"], r["id"]) for r in rows}
    want_anime = int(TARGET * ANIME_SHARE)
    added = 0
    for kind in ("tv", "movie"):
        if added >= want_anime:
            break
        print(f"-> Enumerating Japanese animation ({kind})...")
        for row in tmdb.iter_discover_ids(
                kind, {"with_genres": 16, "with_original_language": "ja"}):
            key = (kind, row["id"])
            if key in seen:
                continue
            seen.add(key)
            rows.append({"kind": kind, "id": row["id"],
                         "pop": float(row.get("popularity") or 0)})
            added += 1
            if added >= want_anime:
                break
    print(f"   +{added} anime candidates beyond the popularity slice")

    with IDS_FILE.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"✅ {len(rows)} candidate ids -> {IDS_FILE}")


# --- Stage 2: fetch -----------------------------------------------------------

def stage_fetch() -> None:
    """Download full details. Resumable: already-fetched ids are skipped.

    This is the only expensive stage (one request per title), so it is the one
    that must never have to start over.
    """
    if not IDS_FILE.exists():
        sys.exit("Run `python -m catalogue.build ids` first.")

    todo = [json.loads(l) for l in IDS_FILE.open(encoding="utf-8")]
    done = set()
    if RAW_FILE.exists():
        with gzip.open(RAW_FILE, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done.add((d["_kind"], d["id"]))
                except (ValueError, KeyError):
                    continue
        print(f"↻ Resuming: {len(done)} already fetched")

    todo = [t for t in todo if (t["kind"], t["id"]) not in done]
    if not todo:
        print("✅ Nothing left to fetch.")
        return

    print(f"⬇️  Fetching {len(todo)} titles with {WORKERS} workers…")
    started, ok = time.time(), 0
    with gzip.open(RAW_FILE, "at", encoding="utf-8") as out, \
            ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(tmdb.fetch_detail, t["kind"], t["id"]): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                detail = fut.result()
            except Exception:  # noqa: BLE001 - one bad title must not stop the run
                detail = None
            if detail:
                detail["_kind"] = t["kind"]
                out.write(json.dumps(detail, ensure_ascii=False) + "\n")
                ok += 1
            if i % 2000 == 0:
                rate = i / max(time.time() - started, 1)
                eta = (len(todo) - i) / max(rate, 0.1) / 60
                print(f"   {i}/{len(todo)}  kept {ok}  {rate:.0f}/s  eta {eta:.0f}m")
                out.flush()
    print(f"✅ Fetched {ok} titles -> {RAW_FILE}")


# --- Stage 3: normalise -------------------------------------------------------

def stage_normalise() -> None:
    """Join AniList + IMDb, build tags, apply quality gates. Costs no API calls,
    so tag rules can be iterated on freely."""
    if not RAW_FILE.exists():
        sys.exit("Run `python -m catalogue.build fetch` first.")

    anilist_index = {}
    if not NO_ANILIST:
        print("🏷️  Fetching AniList ranked tags…")
        anilist_index = enrich.index_anilist(enrich.fetch_anilist())
        print(f"   {len(anilist_index)} anime title keys indexed")

    imdb = {}
    if not NO_IMDB:
        print("⭐ Fetching IMDb ratings…")
        imdb = enrich.fetch_imdb_ratings()
        print(f"   {len(imdb)} IMDb ratings loaded")

    kept, dropped, best = 0, {}, {}
    with gzip.open(RAW_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                detail = json.loads(line)
            except ValueError:
                continue
            kind = detail.pop("_kind", "movie")

            year = (detail.get("release_date") or detail.get("first_air_date") or "")[:4]
            anime = None
            if anilist_index:
                anime = enrich.match_anilist(
                    anilist_index,
                    detail.get("title") or detail.get("name") or "",
                    detail.get("original_title") or detail.get("original_name") or "",
                    int(year) if year.isdigit() else None,
                )
            rec = build_record(kind, detail, anime, None)
            if not rec:
                dropped["unbuildable"] = dropped.get("unbuildable", 0) + 1
                continue
            if rec.get("imdb_id") and rec["imdb_id"] in imdb:
                r = imdb[rec["imdb_id"]]
                rec["rating"] = rec["rating_f"] = round(r["rating"], 1)
                rec["votes"] = max(rec["votes"], r["votes"])
                rec["imdb_rating"] = r["rating"]
                rec["imdb_votes"] = r["votes"]

            ok, why = passes_quality(rec, MIN_VOTES)
            if not ok:
                dropped[why] = dropped.get(why, 0) + 1
                continue

            # Keep the better-known of any true duplicate; remakes survive
            # because the key includes the year.
            k = dedupe_key(rec["title"], rec["year"])
            if k not in best or rec["votes"] > best[k]["votes"]:
                best[k] = rec

    rows = _select(list(best.values()), TARGET)
    kept = len(rows)
    with CATALOGUE_FILE.open("w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ {kept} titles -> {CATALOGUE_FILE}")
    print("   dropped: " + ", ".join(f"{v} {k}" for k, v in
                                     sorted(dropped.items(), key=lambda t: -t[1])))
    _report(rows)


# Share of the catalogue each bucket is guaranteed before the global fill.
# Ranking the whole pool on raw votes erases documentaries — they carry an order
# of magnitude fewer votes than a blockbuster, not an order of magnitude less
# worth — and a floor is the only thing that keeps the four categories browsable.
QUOTAS = {"MOVIE": 0.45, "TV": 0.22, "ANIME": 0.12, "DOCUMENTARY": 0.07}

# Anime enumerated directly from /discover, as a share of TARGET. The quota
# above can only choose from what the ids stage actually collected.
ANIME_SHARE = float(os.getenv("ANIME_SHARE", "0.18"))

RECENT_YEARS = 4         # how long a title counts as "new"
RECENCY_BOOST = 0.35     # peak multiplier for this year's releases


def _selection_score(r: dict) -> float:
    """How much this title deserves a slot.

    Votes alone are a popularity contest with a 20-year head start: a 2008
    blockbuster outvotes a better 2025 film purely by having existed longer.
    quality_prior already balances "is it good" against "is it actually
    watched", and the recency term pays back the time a new release has not
    had yet.
    """
    s = quality_prior(float(r.get("rating") or 0), float(r.get("votes") or 0))
    year = int(r.get("year_i") or 0)
    if not year:
        return s
    age = _THIS_YEAR - year
    if 0 <= age <= RECENT_YEARS:
        s *= 1.0 + RECENCY_BOOST * (RECENT_YEARS - age + 1) / (RECENT_YEARS + 1)
    return s


def _select(rows: list, target: int) -> list:
    """Best `target` titles, with every category represented.

    Two passes: fill each bucket's floor from that bucket's best, then fill
    whatever is left from the global ranking. A bucket short of its quota
    simply leaves its slots to the global pass.
    """
    scored = sorted(rows, key=_selection_score, reverse=True)
    quota = {c: int(target * share) for c, share in QUOTAS.items()}
    picked, taken, count = [], set(), {}
    for r in scored:
        c = r.get("category")
        if count.get(c, 0) < quota.get(c, 0):
            picked.append(r)
            taken.add(id(r))
            count[c] = count.get(c, 0) + 1
    for r in scored:
        if len(picked) >= target:
            break
        if id(r) not in taken:
            picked.append(r)
    return sorted(picked, key=_selection_score, reverse=True)[:target]


def _report(rows: list) -> None:
    """Print what actually got built. A build you cannot inspect is a build you
    cannot trust."""
    if not rows:
        return
    cats, tagged, trailers, anime_tagged, providers = {}, 0, 0, 0, 0
    tag_total = 0
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
        n = len(r.get("tags") or [])
        tag_total += n
        if n >= 5:
            tagged += 1
        if r.get("trailer_key"):
            trailers += 1
        if r.get("anilist_id"):
            anime_tagged += 1
        if r.get("providers"):
            providers += 1
    n = len(rows)
    print(f"\n   categories: " + "  ".join(f"{k} {v}" for k, v in sorted(cats.items())))
    print(f"   tags:       {tag_total / n:.1f} avg, {tagged * 100 // n}% have 5+")
    print(f"   anilist:    {anime_tagged} matched")
    print(f"   trailers:   {trailers * 100 // n}%")
    print(f"   providers:  {providers * 100 // n}%")
    print(f"   posters:    100% (quality gate rejects anything without one)")


# --- Stage 4: index -----------------------------------------------------------

def stage_index() -> None:
    """Embed and load into Postgres.

    Indexes are built AFTER the bulk load: inserting 100k rows into an existing
    HNSW index is several times slower than loading first and indexing once.
    """
    if not CATALOGUE_FILE.exists():
        sys.exit("Run `python -m catalogue.build normalise` first.")

    from backend import store
    from backend.engine import embed_docs

    rows = [json.loads(line) for line in CATALOGUE_FILE.open(encoding="utf-8")]

    if RECREATE:
        print("🗑️  Dropping existing catalogue tables…")
        store.drop_all()
    store.create_schema()
    print(f"🧮 Vector type: {store.vector_type()}")

    print(f"🔢 Embedding + loading {len(rows)} titles…")
    B, done = 256, 0
    started = time.time()
    for start in range(0, len(rows), B):
        chunk = rows[start:start + B]
        for rec in chunk:
            rec["id"] = stable_id(rec["tmdb_kind"], rec["tmdb_id"])
        texts = [document(r) for r in chunk]
        store.upsert(chunk, texts, embed_docs(texts))
        store.upsert_extra(chunk)
        done += len(chunk)
        if done % 2560 == 0 or done == len(rows):
            rate = done / max(time.time() - started, 1)
            print(f"   {done}/{len(rows)}  ({rate:.0f}/s)")

    print("🏗️  Building indexes (HNSW is the slow one)…")
    store.create_indexes()
    store.analyze()

    rep = store.storage_report()
    print(f"✅ Indexed {done} titles.")
    print(f"   storage: {rep['total_mb']} MB total "
          f"({rep['bytes_per_title'] / 1024:.1f} KB/title)")
    if rep["reliable"]:
        print(f"   projected at 100k: {rep['projected_100k_mb']:.0f} MB")


STAGES = {"ids": stage_ids, "fetch": stage_fetch,
          "normalise": stage_normalise, "index": stage_index}


def main() -> None:
    if not os.getenv("TMDB_API_KEY"):
        sys.exit("❌ TMDB_API_KEY missing (see .env.example)")
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    stage = args[0] if args else "all"
    if stage == "all":
        for name, fn in STAGES.items():
            print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")
            fn()
    elif stage in STAGES:
        STAGES[stage]()
    else:
        sys.exit(f"Unknown stage '{stage}'. One of: {', '.join(STAGES)}, all")


if __name__ == "__main__":
    main()
