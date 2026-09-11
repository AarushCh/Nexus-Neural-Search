# Nexus Neural Search — Master Summary & Rebuild Plan

**Status date:** 2026-09-11
**Repo:** `E:\CV Project Files\Github Clone\Nexus-Neural-Search` · branch `main` · clean tree · HEAD `d6a4bcc`
**Live frontend:** https://nexus-neural-search-pi.vercel.app/ (200 OK, 0.9s)
**Live API:** https://nexus-neural-search.onrender.com/ (200 OK, **42.6s cold start**)

---

## 0. 🚨 Read this first — the site is currently broken in production

Measured against the live deployment:

| Check | Result | Meaning |
|---|---|---|
| `GET /` (cold) | 200 in **42.6 s** | Render free tier had spun down. Every AFK return pays this. |
| `POST /recommend` `{"text":"cyberpunk anime about identity"}` | `[]` in 24.3 s | **Zero results** |
| `POST /recommend` `{"text":"Breaking Bad"}` (warm) | `[]` in 0.84 s | **Zero results, fast** |
| `GET /discover` (warm) | `[]` in 0.30 s | **Zero results, fast** |

The API is alive and fast once warm. It returns **nothing** because **the Qdrant collection is gone or empty**. The 0.3s response proves Qdrant is being reached and answering — it just has no points, or the collection `freeme_collection` no longer exists and the exception path in `hybrid_search` swallows it and returns `[]`.

So right now the live product is a beautiful shell with no data behind it. Every symptom you described ("click search or random and it doesn't work") has **three independent causes stacked on top of each other**, and this is the biggest one.

**Immediate triage order:**
1. Confirm the Qdrant cluster state (is it deleted, or just empty?).
2. Rebuild the catalogue into whatever vector store we land on (§6).
3. Fix the cold start + silent failure UX (§5).

---

## 1. What this project is

A hybrid AI discovery engine for movies, TV, anime and documentaries. You can type a **vibe** ("cyberpunk anime about identity") or an **exact title** ("Cyberpunk: Edgerunners") and it handles both, because it runs dense semantic retrieval and BM25 keyword retrieval in parallel and fuses them.

On top of retrieval sits a full product: JWT accounts, a personalized home feed, wishlist, search history, view/like/dismiss tracking, and standalone detail pages with trailers, cast and streaming providers.

**Positioning:** this is a CV/portfolio flagship. That matters for the plan — the goal is not just "works", it's "demonstrably sophisticated and demonstrably reliable when a recruiter opens it cold at 11pm". A 42-second cold start that returns an empty grid is the single worst thing this project currently does to that goal.

---

## 2. Current architecture (as-built)

```
Browser (Vercel, Next.js 16 / React 19)
   │  fetch  NEXT_PUBLIC_API_URL  (default: nexus-neural-search.onrender.com)
   ▼
FastAPI (Render free, 512MB)
   ├── backend/engine.py  ── fastembed ONNX ──► bge-small-en-v1.5 (384d, dense)
   │                       ── fastembed ONNX ──► Qdrant/bm25 (sparse)
   │                       ── (optional) sentence-transformers cross-encoder [OFF in prod]
   │                       ── OpenRouter → nvidia/nemotron-nano-12b-v2-vl:free
   ├── Qdrant Cloud  ── named vectors: "dense" + "bm25", payload = full card
   └── Neon Postgres ── users, wishlist, search_history, interactions, media_details
```

### 2.1 File-by-file inventory

**Backend (1,310 lines)**

| File | Lines | Role | Health |
|---|---|---|---|
| [backend/engine.py](backend/engine.py) | 706 | The whole retrieval core: lazy singletons, dense/sparse embedding, hybrid RRF search, title pinning, score calibration, Qdrant recommend, similar-items, grounded RAG, feed helpers, TMDB detail enrichment | **Doing too much.** Solid logic, but this is 5 modules in a trench coat |
| [backend/main.py](backend/main.py) | 397 | FastAPI routes: search, personalized search, similar, auth, feed, discover, interactions, history, title detail, wishlist | Thin, correct, but has dead code and N+1 DB calls |
| [backend/models.py](backend/models.py) | 81 | SQLAlchemy: `User`, `WishlistItem`, `SearchHistory`, `Interaction`, `MediaDetail` | Fine. No migrations (auto-create only) |
| [backend/auth.py](backend/auth.py) | 97 | JWT (HS256), bcrypt via passlib, `get_current_user_db` dep | **60-minute token, no refresh** — see §5 |
| [backend/database.py](backend/database.py) | 29 | Env-driven engine, SQLite local / Postgres prod, `pool_pre_ping` | Good |

**Frontend (Next.js 16 App Router, ~1,116 lines TSX + 1,353 lines CSS)**

| File | Lines | Role |
|---|---|---|
| [frontend/app/page.tsx](frontend/app/page.tsx) | 322 | The entire app: home/results/wishlist/similar views, search, filters, URL state, history panel, About modal |
| [frontend/app/title/[id]/page.tsx](frontend/app/title/[id]/page.tsx) | 131 | Standalone detail page (trailer, cast, providers) |
| [frontend/components/Chrome.tsx](frontend/components/Chrome.tsx) | 117 | Menu button, sidebar HUD, model selector, theme toggle |
| [frontend/components/NeuralBg.tsx](frontend/components/NeuralBg.tsx) | 127 | Reactive HTML5 canvas neural-network background |
| [frontend/components/DetailModal.tsx](frontend/components/DetailModal.tsx) | 138 | In-page detail modal (duplicates the title page ~80%) |
| [frontend/components/AuthModal.tsx](frontend/components/AuthModal.tsx) | 98 | Login/signup |
| [frontend/components/Feed.tsx](frontend/components/Feed.tsx) | 77 | Paged carousel rows |
| [frontend/components/Card.tsx](frontend/components/Card.tsx) | 68 | Result card with %MATCH bar, type badge, heart |
| [frontend/lib/api.ts](frontend/lib/api.ts) | 128 | Typed API client |
| [frontend/lib/store.tsx](frontend/lib/store.tsx) | 132 | React Context: token, user, online, wishlist ids, model, toast |
| [frontend/lib/img.ts](frontend/lib/img.ts) | 17 | wsrv.nl poster proxy + dead-CDN fallback |
| [frontend/lib/types.ts](frontend/lib/types.ts) | 32 | `Media`, `FeedRow`, `Filters`, `SearchModel` |

**Data pipeline (8 scripts, 6 of them legacy)**

| File | Lines | Role | Verdict |
|---|---|---|---|
| [build_catalogue.py](build_catalogue.py) | 266 | **Authoritative** builder: TMDB discover → validate → enrich → embed → upsert | Keep, rewrite (§7) |
| [build_indexes.py](build_indexes.py) | 92 | Backfills `rating_f`/`year_i`/`category` + payload indexes | Superseded by build_catalogue |
| [ingest.py](ingest.py) | 158 | Legacy CSV ingester + `stable_id()` helper | Only `stable_id` still used — and it's **broken** (§4.3) |
| [harvest_popular.py](harvest_popular.py) | 166 | Earlier vote-count harvester | Dead |
| [expand_dataset.py](expand_dataset.py) | 214 | Earlier TMDB expander | Dead |
| [enrich_data.py](enrich_data.py) | 124 | Earlier Jikan/TMDB enricher | Dead |
| [clean_dataset.py](clean_dataset.py) | 104 | One-off fix for duplicated descriptions | Dead (was a real bug once) |
| [debug.py](debug.py) | 30 | Sanity check: point count + one hybrid search | Keep, expand into a real smoke test |
| `dataset.csv` | 28,697 rows / 12 MB | Legacy MAL-derived seed | Dead weight in git |
| `raw_data/AnimeList.csv` | 14,479 rows | Original MAL dump | Dead weight |
| [docs/](docs/) | 1,816 lines | The **old vanilla-JS frontend** (index.html + script.js + style.css) | Fully superseded by `frontend/` |

**Roughly 2,700 lines of the repo are dead.** See §12.

### 2.2 How a search actually flows today

1. `page.tsx:runSearch()` → `api.search()` → picks `/recommend/personalized` if a token exists, else `/recommend`.
2. `main.py:_search()` builds a Qdrant filter from `category` / `min_rating` / `year_min` / `year_max`.
3. `engine.py:hybrid_search()`:
   - embeds the query with bge-small, prefixed `"Represent this sentence for searching relevant passages: "`
   - builds a BM25 sparse vector
   - sends **both as Qdrant `Prefetch`** and fuses with `FusionQuery(fusion=RRF)`, limit 60
   - `_rerank()` — cross-encoder, **disabled in prod**
   - `_title_lookup()` — scrolls the full-text `title` index for exact/prefix matches and **pins** them to the top
   - `_calibrate_scores()` — sigmoid over the cross-encoder logit → %MATCH badge
4. If `model == "api"`, `ground_with_llm()` sends the top 40 real hits to Nemotron as JSON and asks it to re-order + add a one-line reason. Falls back to hybrid order on any failure.
5. `_patch_posters()` overrides stored images with corrected TMDB posters from the `media_details` cache.

### 2.3 Qdrant schema

- Collection: `freeme_collection`
- Vectors: `dense` (384, cosine) + `bm25` (sparse, IDF modifier)
- Point id: `uuid5(NAMESPACE_DNS, title.lower().strip())`
- Payload: `title, description, image, backdrop, type, category, genre, rating, rating_f, year, year_i, votes, pop, runtime, trailer_key, cast[], providers[], tmdb_id, tmdb_kind`
- Payload indexes: `title` (TEXT), `category` (KEYWORD), `rating_f` (FLOAT), `year_i` (INT), `votes` (INT)

---

## 3. What actually works well (don't break these)

- **Hybrid RRF retrieval is the right architecture.** Dense + sparse fusion genuinely solves the "vibe vs exact title" problem that a single embedding model cannot.
- **Grounded RAG is honest.** The LLM only reorders real DB rows and can't hallucinate a title or a poster. This is the correct pattern and most portfolio projects get it wrong.
- **Search purity.** `/recommend/personalized` logs history but returns *pure* query relevance — taste only leaks into the feed, never into an explicit search. That's a deliberate, correct product call.
- **Graceful degradation everywhere.** Every heavy component is lazy-loaded and wrapped: rerank fails → fusion order; sparse fails → dense only; RAG fails → hybrid hits; Qdrant fails → `[]`. Nothing crashes the API.
- **Single embedding model for ingest and query.** Killed the old MiniLM-vs-bge mismatch. Non-negotiable invariant going forward.
- **ONNX/fastembed instead of torch.** Cut resident memory enough to fit a 512MB box. Good engineering under constraint.
- **TMDB as single source of truth** with a validation gate (poster + ≥20-char overview required). Cards are correct by construction.
- **The UI.** The cyberpunk aesthetic, the reactive canvas background, the glitch title, dual themes, the paged carousels — this is genuinely striking and it's a real asset.

---

## 4. Faults — ranked by damage

### 4.1 🔴 CRITICAL — The vector store is empty in production
Covered in §0. Everything else is cosmetic until this is fixed. Also note: `hybrid_search` catches the Qdrant exception, prints to stdout, and returns `[]`. From the outside, "collection deleted" and "no results for your query" are **indistinguishable**. That's why you can't tell what's wrong from the browser.

### 4.2 🔴 CRITICAL — Silent failure is the default everywhere
`except Exception → print → return []` appears **nine times** in [engine.py](backend/engine.py). This was a deliberate uptime choice and it worked, but it means the app cannot tell you it's broken. There is no error surface, no logging sink, no alerting. Failure and emptiness look identical.

### 4.3 🔴 HIGH — `stable_id()` collides on title
[ingest.py:50](ingest.py#L50) — `uuid5(NAMESPACE_DNS, title.lower().strip())`.

Every title that shares a name **shares an id**:
- *Dune* (1984) vs *Dune* (2021)
- *The Office* (UK) vs *The Office* (US)
- *Battlestar Galactica* (1978) vs (2004)
- Every remake, every anime with a same-named live-action adaptation

And [build_catalogue.py](build_catalogue.py) actively enforces the loss — it dedupes by normalized title and keeps only the highest-voted version. **You are structurally incapable of holding both Dunes.** Fix: id must be `uuid5(f"{tmdb_kind}:{tmdb_id}")`.

### 4.4 🔴 HIGH — The catalogue is capped at ~6,850 by design
TMDB's `/discover` endpoint **hard-caps at page 500**. With 6 streams × up to `--pages` you cannot page past that ceiling per stream, so no amount of `--pages 500` gets you past roughly 10k results per query shape. The 6,850 number is not a choice, it's a wall. See §7 for how to break it.

### 4.5 🟠 The README describes a system that isn't running
- README: "reranks with a cross-encoder". Reality: `ENABLE_RERANK` defaults to `false` ([engine.py:50](backend/engine.py#L50)) and `sentence-transformers`/`torch` are **commented out of requirements.txt**. The reranker cannot load in prod.
- README: "Calibrated match %". Reality: with rerank off, `_calibrate_scores` finds no logit and falls through to `_score_for_ui()` — pure rank position. The %MATCH badge is decorative.
- `GET /` hardcodes `{"engine": "hybrid+rerank"}` ([main.py:93](backend/main.py#L93)) regardless of actual state.

Not fraud — drift. But a recruiter who reads the README and then tests the claim finds a gap. Fix by making the claim true (§9) *and* making `/` report real state.

### 4.6 🟠 Every "Explore Similar" card shows exactly 82% MATCH
[backend/engine.py:436-457](backend/engine.py#L436-L457). With rerank off, every candidate gets `_rr = 0.0`. Then `_calibrate_scores` sees the key present and computes `sigmoid((0.0 + 7.0) / 4.6) = 0.8206` → **82** for all of them, identically. Go look at the similar view — it's a wall of 82%.

### 4.7 🟠 Feed does 8+ round trips per load
[main.py:_popular_rows()](backend/main.py#L253) loops 4 categories; each does a Qdrant `scroll` **and** `_patch_posters()` opens its own fresh `SessionLocal()` for a Postgres query. Anonymous `/discover` is 4 Qdrant scrolls + 4 Neon connections, on **every single page load**, with zero caching. On a cold Neon branch that's seconds.

### 4.8 🟠 `/recommend` is unauthenticated, unthrottled, and can spend your money
`model: "api"` triggers an OpenRouter call. No rate limit, no auth, no captcha. Anyone with curl can drain your OpenRouter quota in a loop. Also: `allow_origins=["*"]` **with** `allow_credentials=True` ([main.py:45](backend/main.py#L45)) is an invalid CORS combination that browsers reject and that signals "not hardened".

### 4.9 🟠 Genre/tag data is a comma-joined string glued into the description
[build_catalogue.py](build_catalogue.py) writes `description = f"{overview} Genres: {gs}."`. Consequences:
- Genre text **pollutes the embedding** and inflates BM25 for common genre words
- The literal string `"Genres: Action, Drama."` is shown to users in card blurbs
- There is **no structured, filterable, clickable tag facet** at all
- You cannot filter by genre, only by the 4 hard-coded categories

This is exactly the thing you said you want to rip out. Full redesign in §8.

### 4.10 🟡 Category is single-label and order-dependent
[build_catalogue.py STREAMS](build_catalogue.py) — anime streams run first so they claim genre-16 titles before movie/TV. Consequences:
- *Arcane* (French, animated) → **MOVIE/TV**, not ANIME
- A documentary about anime → whichever stream got there first
- Western animation (*Rick and Morty*, *Bojack*) has **no home**
- A title is exactly one of {MOVIE, TV, ANIME, DOCUMENTARY} forever

### 4.11 🟡 Cast is stored but not searchable
`cast[]` sits in the payload. It is **not** in the embedded text and **not** in the BM25 text. Searching *"Christopher Nolan"*, *"Bryan Cranston"*, or *"Hayao Miyazaki"* finds nothing unless the name happens to appear in an overview. For a media search engine this is a glaring hole.

### 4.12 🟡 No pagination
`top_k` is fixed at 12 in [api.ts](frontend/lib/api.ts). No "load more", no infinite scroll, no page 2. Ever.

### 4.13 🟡 No tests, no CI, no migrations
Zero test files. No GitHub Actions. `Base.metadata.create_all()` on startup means any schema change to `models.py` requires manual SQL against Neon.

### 4.14 🟡 Duplication and dead code
- `DetailModal.tsx` and `app/title/[id]/page.tsx` are ~80% the same JSX
- `_interleave()` ([main.py:380](backend/main.py#L380)) is never called
- `enrich_detail()` is now mostly unreachable (build_catalogue pre-fills the payload)
- `build_indexes.py` is superseded
- `docs/` is the entire old frontend, still shipped

### 4.15 🟡 Security housekeeping
- `SECRET_KEY` falls back to `"dev-only-insecure-secret-change-me"` ([auth.py:15](backend/auth.py#L15)) — if the env var is ever missing in prod, every token is forgeable
- No rate limit on `/login` → unlimited brute force
- No password strength rule, no email verification
- `bcrypt==3.2.0` pinned to dodge the passlib 1.7.4 incompatibility — passlib is effectively unmaintained; move to `bcrypt` directly or `argon2-cffi`

---

## 5. 🎯 The AFK bug — full root cause chain

You said: *"While hosted, after going AFK, when I click search / go / random it would not work."*

There are **four** independent causes. All four are real. All four need fixing.

### Cause 1 — Render free tier spins down after 15 minutes idle
Measured: **42.6 seconds** to first byte on a cold `GET /`. The frontend's `fetch` has **no timeout and no retry**. You click search, nothing visibly happens for 40+ seconds, and you conclude it's broken. It is, functionally.

### Cause 2 — The health check runs exactly once, at mount
[store.tsx:64](frontend/lib/store.tsx#L64) — `api.health().then(setOnline)` fires once in a `useEffect` on mount. If the API is cold at that moment, the promise resolves `false` and the header reads **OFFLINE forever**, until you manually reload. There is no re-check on focus, on visibility change, or on retry.

### Cause 3 — JWT expires in 60 minutes with no refresh
[auth.py:17](backend/auth.py#L17) — `ACCESS_TOKEN_EXPIRE_MINUTES = 60`. You log in, go AFK for over an hour, come back and click search. Because a token exists in localStorage, `api.search()` routes to `/recommend/personalized`. The expired token → **401** → `api.ts` throws `UNAUTHORIZED` → `page.tsx` catches it, shows a toast, and **sets results to `[]`**. You get an empty grid.

This is the purest version of your symptom: it fails *only* after being away, *only* when logged in.

### Cause 4 — The collection is empty (§0)
Even with all of the above fixed, right now you'd get `[]` anyway.

### The fix stack

| Fix | Effort | Effect |
|---|---|---|
| **Keep-alive cron** — GitHub Actions or cron-job.org pings `/` every 10 min | 15 min | Kills the 42s cold start entirely |
| **Frontend timeout + retry + "waking up" state** — `AbortController` at 8s, retry twice, show *"Waking the neural core… ~40s"* instead of a dead spinner | 1 hr | Honest UX even when cold |
| **Health re-check on `visibilitychange` + `focus`** | 20 min | OFFLINE badge self-heals when you come back |
| **JWT → 30 days + refresh endpoint**, and on 401 **fall back to anonymous search** instead of returning `[]` | 1 hr | Never silently empty again |
| **Move off Render free** (§10) | half day | Removes cause 1 permanently |
| **Rebuild catalogue** (§6, §7) | see below | Removes cause 4 |

**Ship the cron + the frontend retry today.** They are the highest ratio of user-visible fix to effort in the entire document.

---

## 6. 🗄️ The Vector DB problem — Qdrant keeps deleting your clusters

### Why it happens
Qdrant Cloud's free tier (1GB, 1 node) is explicitly *not* for production. Clusters are **suspended after a period of inactivity and eventually deleted**, and free clusters get no backup guarantee. You are hosting a portfolio piece that sits idle for days between recruiter visits — you are the exact profile that gets reaped. This will happen again. It is not a bug you can fix by being careful.

### What your data actually is
This is the key sizing fact that unlocks every cheap option:

- ~6,850 titles today, targeting **60k–150k** (§7)
- 384-dim float32 vectors
- **150,000 × 384 × 4 bytes = 230 MB** — and 115 MB at float16, 58 MB at int8
- Payloads (title, overview, cast, tags) ≈ 2–4 KB each → 300–600 MB of JSON

**Your entire vector index fits in RAM on a laptop.** You are not running a system that needs a distributed vector database. Every architecture below is viable; the question is only which one is most reliable and most impressive.

### The options

| # | Option | Cost | Deletion risk | Cold start | Impressiveness | Verdict |
|---|---|---|---|---|---|---|
| **A** | **Neon Postgres + `pgvector`** | £0 (already have Neon) | **None** — you already trust it with users | ~1s (pooled) | High — hybrid search in raw SQL | ⭐ **Recommended** |
| B | Qdrant self-hosted on Fly.io / Railway with a persistent volume | ~£0–5/mo | None (your disk) | 1–3s | High — you operate the DB | Strong runner-up |
| C | Qdrant Cloud paid (starts ~$25/mo) | £££ | None | ~0 | Same as today | Only if funded |
| D | Qdrant Cloud free + **keep-alive cron** | £0 | Reduced, not eliminated | ~0 | Same as today | Band-aid |
| E | Embedded index baked into the image (`sqlite-vec`, LanceDB, FAISS, or Qdrant local mode) | £0 | **Zero** — it's a file in your repo/image | 0ms | Medium — clever, but "no database" | Great fallback |
| F | Turso (libSQL) + `sqlite-vec` | £0 generous tier | Low | ~50ms edge | Medium-high | Interesting |
| G | Pinecone / Weaviate / Milvus free tiers | £0 | Same reaping risk as Qdrant | varies | Medium | Trading one reaper for another |
| H | Postgres full-text only, no vectors | £0 | None | fast | **Low** — abandons the whole premise | No |

### ⭐ Recommendation: **A — consolidate onto Neon Postgres + pgvector**

Rationale:

1. **You already have Neon.** Users, wishlist, history and interactions live there. Adding one more table deletes an entire external dependency instead of adding one. Fewer moving parts, fewer secrets, fewer things to be deleted overnight.
2. **Neon does not reap your data.** Free Neon branches auto-suspend compute after 5 minutes but the storage persists, and `pool_pre_ping` (already in [database.py](backend/database.py)) handles the wake.
3. **Hybrid search works natively.** `pgvector` HNSW gives you the dense side. Postgres `tsvector` + `ts_rank_cd` + `websearch_to_tsquery` gives you the sparse/keyword side — a proper BM25-ish ranking with stemming and phrase support. RRF fusion is ~20 lines of SQL with two CTEs.
4. **Filtering gets dramatically better.** Right now every facet needs a hand-built Qdrant payload index. In Postgres, tags are a `text[]` with a GIN index, year is an `int`, providers are a join. You can `WHERE 'cyberpunk' = ANY(tags) AND year BETWEEN 1980 AND 1999` for free, and you can do faceted **counts**, which Qdrant makes awkward.
5. **It reads better on a CV.** "I implemented hybrid dense+lexical retrieval with reciprocal rank fusion in PostgreSQL" is a stronger sentence than "I called a vector database's search method", because it proves you understand what RRF *is*.
6. At 150k rows, pgvector HNSW returns in **single-digit milliseconds**. Scale is a non-issue.

**Insurance:** implement the store behind a thin interface with two backends — `pgvector` (primary) and an **embedded index built at deploy time** (option E, fallback). If Neon is asleep or unreachable, serve from the local index. Then no single provider can take your site down again. This is the one abstraction in this whole document that earns its keep, because it is the direct answer to the failure you've actually experienced twice.

**Migration path:** `build_catalogue.py` already produces validated row dicts. Point the writer at Postgres instead of Qdrant, keep the same payload shape, and `hybrid_search()` becomes one SQL query. Roughly a day of work, most of it in the SQL.

---

## 7. 📚 Dataset v2 — "way, way, way better"

This is the highest-leverage section. Your retrieval quality has a hard ceiling set by your data, and right now that ceiling is low: **6,850 titles, one genre string, no cast search, no keywords, no franchise, no language, no certification.**

### 7.1 The wall you have to break first

`build_catalogue.discover()` pages TMDB `/discover`. **TMDB caps `/discover` at page 500.** Six streams cannot produce more than ~60k raw candidates even in theory, and with `vote_count.gte=120` and dedupe-by-title you land at 6,850. More `--pages` will not help.

**The fix: TMDB daily ID exports.**
```
http://files.tmdb.org/p/exports/movie_ids_MM_DD_YYYY.json.gz
http://files.tmdb.org/p/exports/tv_series_ids_MM_DD_YYYY.json.gz
http://files.tmdb.org/p/exports/person_ids_MM_DD_YYYY.json.gz
```
These are gzipped JSONL files listing **every id in TMDB** with its popularity. ~1M movies, ~200k TV series. Download, filter by popularity, then hit `/movie/{id}` and `/tv/{id}` directly. No pagination wall. This single change is what takes you from 6,850 to 150,000.

### 7.2 Target catalogue

| Segment | Source | Target count | Gate |
|---|---|---|---|
| Movies | TMDB export | 60,000 | poster + overview ≥ 40 chars + `vote_count ≥ 10` |
| TV series | TMDB export | 25,000 | same |
| Anime (series + film) | TMDB ∪ **AniList** | 12,000 | AniList id match, or TMDB genre 16 |
| Documentaries | TMDB genre 99 | 8,000 | same |
| Western animation | TMDB genre 16, non-JP | 4,000 | new category |
| Stand-up / specials | TMDB | 1,500 | new category |
| **Total** | | **~110,000** | |

At ~3 KB/row that's ~330 MB of payload and ~170 MB of float32 vectors. **Check the Neon free-tier storage limit before committing to 110k** — see open question #2.

### 7.3 Fields to capture (vs. today)

| Field | Today | v2 | Why it matters |
|---|---|---|---|
| `tmdb_id`, `tmdb_kind` | ✅ | ✅ **becomes the primary key** | Fixes the Dune collision (§4.3) |
| `imdb_id` | ❌ | ✅ | Joins to IMDb ratings + external links |
| `title`, `original_title` | partial | ✅ both | Anime searchable by romaji *and* English |
| `alt_titles[]` | ❌ | ✅ via `/alternative_titles` | *"AoT"*, *"SnK"*, *"Shingeki no Kyojin"* all hit *Attack on Titan* |
| `overview` | ✅ | ✅ **clean, no genre string appended** | Stops embedding pollution (§4.9) |
| `tagline` | ❌ | ✅ | Punchy, embeds well |
| `genres[]` | string blob | ✅ structured array | Filterable |
| `keywords[]` | ❌ | ✅ **TMDB `/keywords`** | ⭐ The single biggest quality win — see 7.4 |
| `cast[]` (top 15) | stored, unsearchable | ✅ **in embedding + lexical index** | Fixes §4.11 |
| `crew` (director, writer, composer) | ❌ | ✅ | *"Villeneuve sci-fi"*, *"Miyazaki"*, *"Hans Zimmer score"* |
| `studios` / `networks` | ❌ | ✅ | *"A24 horror"*, *"Ghibli"*, *"HBO drama"* |
| `collection` / franchise | ❌ | ✅ | Groups *Dune*, *Marvel*, *Star Wars* |
| `original_language`, `spoken_languages`, `origin_country` | ❌ | ✅ | *"Korean thriller"*, *"French New Wave"* |
| `certification` (MPAA/TV) | ❌ | ✅ | Family-safe filtering |
| `runtime`, `episode_count`, `season_count`, `status` | partial | ✅ | *"under 100 minutes"*, *"finished, bingeable"* |
| `release_date` (full) | year only | ✅ full date | Decade + recency ranking |
| `vote_average`, `vote_count`, `popularity` | ✅ | ✅ | |
| **IMDb `rating`, `votes`** | ❌ | ✅ via IMDb datasets | Better signal than TMDB alone |
| `trailer_key`, `backdrop`, `poster` | ✅ | ✅ + multiple sizes | |
| `providers[]` per region | US only | ✅ US/GB/IN at minimum | *"what's on Netflix in India"* |
| **`tags[]` (unified)** | ❌ | ✅ | §8 |
| **`mood[]` (LLM-derived)** | ❌ | ✅ | §8 |
| `content_warnings[]` | ❌ | ✅ | Real user need, nobody does it well |

### 7.4 ⭐ TMDB Keywords — the free goldmine you're not using

`GET /movie/{id}/keywords` and `/tv/{id}/keywords` return a **human-curated** keyword list per title. Real examples:

- *Blade Runner 2049* → `dystopia`, `artificial intelligence`, `neo-noir`, `replicant`, `memory`, `desert`
- *Everything Everywhere All at Once* → `multiverse`, `absurdism`, `mother daughter relationship`, `immigrant`, `martial arts`
- *Your Name* → `body swap`, `comet`, `time travel`, `rural`, `first love`

These are *exactly* the vibe vocabulary your semantic search is trying to reconstruct from plot summaries. Feeding them into both the embedding text and the tag facet is a step-change in vibe-query quality, for **one extra API call per title** and zero cost.

### 7.5 ⭐ AniList — proper anime tags

AniList's GraphQL API is **free, no key required**. Its tag system is the best structured vibe taxonomy that exists for anime — each tag comes with a **rank 0–100** for how strongly it applies:

```
Cyberpunk: Edgerunners → Cyberpunk 95, Dystopian 88, Tragedy 85,
                          Guns 78, Male Protagonist 70, Drugs 65, Body Horror 60
```

Ranked tags mean you can weight them: a 95 tag belongs in the embedding text, a 40 tag belongs only in the filter facet. Nothing else in the media-metadata world gives you this. Match AniList ↔ TMDB via `idMal` / title + year fuzzy match.

### 7.6 Additional free sources worth pulling

| Source | What you get | Cost |
|---|---|---|
| **IMDb Datasets** (`title.ratings.tsv.gz`, `title.basics.tsv.gz`) | IMDb rating + vote count for ~1.4M titles, joined on `imdb_id` | Free, bulk download, no API |
| **Jikan** (MyAnimeList) | Anime studios, themes, demographics, OP/ED | Free, no key |
| **Wikidata SPARQL** | Awards (Oscars, Emmys, Annie), franchise graphs | Free |
| **OMDb** | Rotten Tomatoes + Metacritic scores | Free tier ~1000/day |
| **TMDB `/changes`** | Daily deltas for incremental updates | Free |

### 7.7 Pipeline architecture v2

Today's builder does discover → enrich → embed → upsert in one process. If it dies at 90%, you start over and re-burn the API budget. Rebuild it as **four idempotent, resumable stages** writing to disk between each:

```
stage 1  ids       TMDB exports + AniList index      → data/ids.parquet
stage 2  fetch     /movie/{id}?append_to_response=   → data/raw/*.jsonl.gz   [resumable, cached]
                   keywords,credits,videos,images,
                   watch/providers,alternative_titles,
                   release_dates,external_ids
stage 3  normalize join IMDb + AniList, derive tags  → data/catalogue.parquet
                   + moods, quality gates, dedupe
                   by tmdb_id
stage 4  index     embed + write to pgvector          → live
                   + build embedded fallback index
```

Benefits: stage 2 is cached so re-running stage 3 costs nothing; the parquet artifact is versioned and diffable; you can rebuild the whole index from disk in minutes without touching TMDB; and each stage gets a smoke test.

**Budget:** ~110k titles × 1 detail call ≈ 110k requests. At ~20 concurrent workers with backoff that's roughly **2–4 hours** for a full cold build, and minutes for incremental daily updates via `/changes`.

### 7.8 What goes into the embedding text

Today: `f"{title}. {description} {category}"` — with genres already glued into `description`.

v2 — a structured, deliberate document:
```
{title} ({original_title}) [{year}] · {category}
{tagline}
{overview}
Genres: {genres}
Themes: {top-8 tags by rank}
Mood: {moods}
Directed by {director}. Starring {top 5 cast}.
{studio} · {origin_country} · {certification}
```
Every clause is a retrieval surface. *"Villeneuve"*, *"A24"*, *"body horror"*, *"Korean revenge thriller"* all become findable, and the lexical index gets the same document so BM25 catches exact names.

---

## 8. 🏷️ Tag system v2 — full redesign

You asked to change the whole tag system. Here's the design.

### 8.1 What's wrong today
- Genres are a **comma-joined string** appended to the description
- That string is shown to users (`"...Genres: Action, Drama."`)
- It pollutes the embedding and skews BM25 toward common genre words
- There is no tag facet, no tag index, no clickable tag, no tag filter
- The only "taxonomy" is 4 hard-coded, mutually-exclusive categories

### 8.2 The new model — four layers

```
┌─ LAYER 1: TYPE (structural, multi-label)  ──────────────────────────┐
│  film · series · limited-series · special · short · documentary     │
│  A title can be BOTH film AND documentary. Fixes §4.10.             │
└─────────────────────────────────────────────────────────────────────┘
┌─ LAYER 2: FORM (how it's made, multi-label) ────────────────────────┐
│  live-action · anime · western-animation · stop-motion · hybrid     │
│  Arcane = series + western-animation. Edgerunners = series + anime. │
└─────────────────────────────────────────────────────────────────────┘
┌─ LAYER 3: GENRE (controlled vocab, ~25, multi-label, weighted) ─────┐
│  action, comedy, drama, horror, sci-fi, thriller, romance, …        │
│  Sourced from TMDB genres ∪ AniList genres, normalized to one vocab.│
└─────────────────────────────────────────────────────────────────────┘
┌─ LAYER 4: TAGS (open vocab, ranked 0–100, ~2000 terms) ─────────────┐
│  TMDB keywords ∪ AniList tags ∪ derived facets ∪ LLM moods          │
│  cyberpunk 95 · time-loop 90 · found-family 72 · slow-burn 60       │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Tag namespaces

Prefix every tag so the UI can group and colour them, and so filters can target a namespace:

| Namespace | Source | Examples |
|---|---|---|
| `theme:` | TMDB keywords, AniList tags | `theme:dystopia`, `theme:time-loop`, `theme:coming-of-age` |
| `mood:` | LLM, controlled vocab of ~60 | `mood:cozy`, `mood:bleak`, `mood:cathartic`, `mood:unsettling` |
| `setting:` | derived | `setting:space`, `setting:1980s`, `setting:post-apocalyptic`, `setting:small-town` |
| `style:` | derived + LLM | `style:neo-noir`, `style:found-footage`, `style:one-shot`, `style:nonlinear` |
| `pace:` | runtime + LLM | `pace:slow-burn`, `pace:relentless`, `pace:episodic` |
| `audience:` | certification + demographics | `audience:family`, `audience:adult`, `audience:shounen`, `audience:josei` |
| `era:` | release date | `era:1980s`, `era:2020s` |
| `origin:` | country + language | `origin:korea`, `origin:japan`, `origin:france` |
| `people:` | credits | `people:christopher-nolan`, `people:hayao-miyazaki` |
| `studio:` | production companies | `studio:a24`, `studio:ghibli`, `studio:trigger` |
| `franchise:` | TMDB collection | `franchise:dune`, `franchise:mcu` |
| `where:` | watch providers | `where:netflix`, `where:crunchyroll`, `where:prime` |
| `warn:` | LLM + certification reasons | `warn:graphic-violence`, `warn:self-harm`, `warn:flashing` |

### 8.4 The mood layer (the actually novel bit)

TMDB and AniList give you *what a thing is about*. They don't give you *how it feels*. That's the gap your "describe your mood" search bar promises to fill and currently fills only by accident, via plot-summary embeddings.

**Approach: one offline LLM pass over the catalogue.**

- Fix a **controlled vocabulary of ~60 moods** (`cozy`, `bleak`, `hopeful`, `melancholic`, `tense`, `whimsical`, `cathartic`, `unsettling`, `triumphant`, `wistful`, `absurd`, `meditative`, `feral`, …). Controlled = filterable, consistent, and countable. Free-text moods would be useless as facets.
- For each title, send `title + overview + genres + top keywords` and ask for **3–6 moods from the fixed list, plus a one-sentence "vibe line"**.
- Batch it. A small, fast instruction-following model is the right tool here — the task is constrained enough that frontier quality buys nothing. At ~400 input / ~60 output tokens per title, 110k titles is a **few dollars, one time**.
- Store `mood[]` (filterable) and `vibe_line` (shown on the card and fed into the embedding).
- Cache by `tmdb_id` so re-runs are free; only new titles cost anything.

The `vibe_line` is also a product feature: instead of a card showing a dry TMDB overview, it can lead with *"A neon-drowned tragedy about a kid who burns out beautifully."* That's what makes a discovery engine feel alive.

### 8.5 How tags flow through retrieval

1. **Embedding** — top-8 tags by rank + moods are written into the document text (§7.8). Vibe queries now match vibe vocabulary directly rather than inferring it from plot.
2. **Lexical** — the same text goes into the `tsvector`, so exact tag words score hard on BM25.
3. **Filter facet** — `tags text[]` with a GIN index. `WHERE tags @> ARRAY['theme:cyberpunk']` is instant.
4. **Tag-similarity channel** — a separate small embedding of *just* the tag set gives a third retrieval channel for "more like this, thematically" that ignores plot noise. Fuse it into RRF as a third input.
5. **Explanations** — because you know which tags matched, the card can say **"matched: cyberpunk · dystopia · identity"** instead of an unexplained 82%.

### 8.6 Tags in the UI

- **Tag chips on every card** (top 3 by rank) and the **full ranked set on the detail page**
- **Click a tag → faceted search** on that tag, with a header (*"47 titles tagged `theme:time-loop`"*)
- **Tag filter bar** replacing today's 4 hard-coded category buttons: multi-select, grouped by namespace, with **live counts**
- **Tag combination** — `cyberpunk + slow-burn − violence`, which is a genuinely differentiated discovery flow
- **Tag pages** — `/tag/cyberpunk` as a real, indexable route. Free SEO, and it makes the catalogue browsable without searching
- **Mood cloud on the home page** — click a mood, get a shelf

---

## 9. 🧠 Models & retrieval v2

### 9.1 Turn reranking back on — without torch
The reranker is off because `sentence-transformers` needs `torch` (~600MB) and Render free is 512MB. But **fastembed ships ONNX cross-encoders**:

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder
# candidates include:
#   Xenova/ms-marco-MiniLM-L-6-v2       (~90MB, same model you're already naming)
#   Xenova/ms-marco-MiniLM-L-12-v2
#   jinaai/jina-reranker-v1-tiny-en     (~30MB, very fast)
#   jinaai/jina-reranker-v2-base-multilingual  (best quality, handles JP titles)
```
Same runtime you already use for dense + sparse, no torch, fits the RAM budget. **Verify before committing** (needs fastembed ≥ 0.4):
```bash
python -c "from fastembed.rerank.cross_encoder import TextCrossEncoder; print([m['model'] for m in TextCrossEncoder.list_supported_models()])"
```

This makes the README claim true, un-breaks the 82%-for-everything bug (§4.6), and materially improves ranking. **Highest quality-per-effort change in the retrieval stack.**

### 9.2 Embedding model
Current: `bge-small-en-v1.5`, 384d. Fine, but English-only and small.

Since documents are embedded **offline**, document-side model size costs you nothing at runtime — only the *query* embedding runs per request. Options:

| Model | Dim | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` (current) | 384 | Fast, English-only |
| `BAAI/bge-base-en-v1.5` | 768 | Meaningfully better, still ONNX-able |
| `jinaai/jina-embeddings-v3` | 1024 (Matryoshka → truncate to 256/512) | **Multilingual** — matters for anime/K-drama/foreign titles. Matryoshka lets you store 512d and keep quality |
| `Snowflake/snowflake-arctic-embed-m-v2.0` | 768 | Multilingual, strong on MTEB retrieval |
| `BAAI/bge-m3` | 1024 | Best multilingual, but heavy |

**Recommendation:** move to a multilingual model (jina-v3 with Matryoshka truncation to 512d, or arctic-embed-m-v2.0). Your catalogue is heavily non-English after the anime/K-drama expansion; an English-only encoder is leaving quality on the floor. If host RAM ever becomes a constraint, embed queries via a hosted API (Jina/Voyage/Cohere free tiers, or OpenAI `text-embedding-3-small` at ~$0.02/1M tokens ≈ **free at your query volume**) and keep the box tiny.

**Invariant to preserve:** same model for documents and queries. Changing the model means a full re-embed. Store `embedding_model` and `embedding_version` on every row so you can detect a mismatch instead of silently serving garbage.

### 9.3 Query understanding
Add a light layer before retrieval:

- **Intent classification** — is this a title lookup, a person lookup, a vibe query, or a filtered browse? Route accordingly. `"breaking bad"` shouldn't take the same path as `"something to watch while it rains"`.
- **Filter extraction** — parse `"90s korean horror under 2 hours"` into `era:1990s + origin:korea + genre:horror + runtime<120` plus a semantic remainder. Rules cover 80%; a cheap LLM covers the rest.
- **HyDE (Hypothetical Document Embeddings)** — for vague vibe queries, have a small model write a fake 2-sentence plot summary matching the request, embed *that*, and retrieve with it. Well-documented, large win on exactly the queries your product is built around. Optional third RRF channel.
- **Query expansion from the tag vocabulary** — map `"mind-bending"` → `theme:nonlinear-narrative, theme:unreliable-narrator, mood:disorienting` and boost those facets.

### 9.4 The LLM layer
Current: `nvidia/nemotron-nano-12b-v2-vl:free` via OpenRouter, used to reorder + explain.

Problems: free-tier models are rate-limited and flaky, so "API mode" silently degrades to hybrid often; the prompt asks for JSON with no schema enforcement, parsed by regex; and there's no timeout budget separate from the 30s client timeout.

Upgrades:
- **Primary model:** a paid small/fast model on OpenRouter — reliable throughput and good instruction-following for a constrained reranking task, for a fraction of a cent per query. Keep a free model as an explicit fallback tier.
- **Structured output** instead of regex-matching a JSON array out of prose.
- **Cache by `(query_hash, candidate_id_set_hash)`** — the same query should never pay twice.
- **Streaming explanations** — return hybrid results *instantly*, then stream the LLM's reasoning in. Today you wait for the whole LLM call before anything renders. This is a big perceived-speed win.
- **New capability — conversational refinement:** *"more like that but less violent"*, *"shorter"*, *"but animated"*. You already have grounded reranking; adding a turn-based refine loop over the current result set is a small step with a large product payoff.
- **Also new — "explain the match":** one sentence per result on why *this* title answers *this* query, grounded in the tags that matched. Directly replaces the meaningless %MATCH number.

### 9.5 Personalization v2
Current: `/feed` collects up to 40 positive ids and sends them to Qdrant `recommend` on every request. That's expensive and it gets muddier the more you save.

Better: maintain a **taste vector** per user — an exponentially-weighted mean of the embeddings of liked/viewed/saved items, updated incrementally on each interaction and stored as a single vector on the user row. Then:
- "For You" = one ANN query against the taste vector. O(1) instead of O(profile size).
- Support **multiple taste clusters** (k-means over a user's liked vectors, k=3) so a user who likes both cozy anime and brutal thrillers gets *both* shelves instead of a mushy average of the two. This is the single most common failure of naive recommenders and fixing it is visibly impressive.
- Add **negative feedback that works** — dismisses should subtract from the relevant cluster, not the global average.
- **Cold start:** a 60-second onboarding — "pick 5 titles you love" from a diverse grid — beats any algorithm for a brand new account.

---

## 10. 🏗️ Infrastructure & hosting v2

### 10.1 The API host
Render free is the root of the AFK problem. Options:

| Host | Free tier | Cold start | RAM | Verdict |
|---|---|---|---|---|
| **Render free** (current) | yes, spins down 15 min | **~43s** measured | 512 MB | The problem |
| **Hugging Face Spaces** | yes, sleeps after ~48h idle | seconds | **16 GB, 2 vCPU** | ⭐ Best free ML host. Torch, big rerankers, everything fits. Thematically perfect for an ML project |
| **Fly.io** | small allowance | 1–3s (Machines wake fast) | configurable | ⭐ Best if you want real control + a persistent volume for self-hosted Qdrant |
| **Google Cloud Run** | 2M req/mo free | 2–5s, or 0 with min-instances | configurable | Solid, scale-to-zero, generous |
| **Oracle Cloud Always Free** | 4 ARM cores / **24 GB RAM**, always on | **0** | 24 GB | ⭐ The "go all out" answer. Host API + Qdrant + everything on one box that never sleeps |
| **Railway** | $5 credit/mo | fast | configurable | Fine, credit runs out |
| **Koyeb** | small free tier | fast | 512 MB | Similar to Render |

**Recommendation:**
- **Fast path (this week):** stay on Render, add the **keep-alive cron**. Cost: 15 minutes. Removes 95% of the pain.
- **Real path:** **Hugging Face Spaces** (if you want max RAM for models and the ML-project framing) or **Oracle Always Free** (if you want a genuinely always-on box you control). Either kills cold starts permanently.

### 10.2 Frontend
Vercel is correct, keep it. Improvements:
- Add **server-side rendering** for `/title/[id]` and `/tag/[tag]` → real SEO, real OG cards when links are shared, instant first paint
- Add `next/image` with the TMDB domain allowlisted, replacing the wsrv.nl proxy hop in [img.ts](frontend/lib/img.ts)
- **Route handlers as a proxy** (`/api/*` → backend) so the API URL isn't exposed and you can add edge caching for `/discover`

### 10.3 Caching
- `/discover` → in-memory TTL cache (5 min) + `Cache-Control` header. It's identical for every anonymous visitor and currently costs 8 round trips per load.
- Search results → cache by `(query, filters, model)` for 60s.
- Detail pages → already cached in `media_details`; add an `ETag`.
- **CDN the feed** entirely via a Vercel route handler with `revalidate: 300`.

### 10.4 Observability
Currently: `print()` to stdout, nothing else.
- **Structured logging** (`structlog`) with request ids
- **Sentry** on both frontend and backend — free tier is plenty
- A **real `/health`** that reports: vector store reachable, **point count**, model loaded, DB reachable, build version. If point count is 0, the health check must **fail**. That one change would have told you the site was broken instead of you finding out from an empty grid.
- Query logging → **the most valuable dataset you'll ever have**: real queries and which results people clicked. That's what lets you actually measure whether a change improved retrieval (§15).

### 10.5 CI/CD
- GitHub Actions: lint (`ruff`), typecheck (`mypy` light / `tsc --noEmit`), run the smoke test, build the frontend
- A **retrieval regression test** — 30 fixed queries with expected top-3 titles. Any change to models, prompts or scoring that breaks them fails the build. This is the single most valuable test a search project can have.
- Nightly catalogue delta via TMDB `/changes`

---

## 11. 🎨 Site & search UX v2

### 11.1 Search experience
| Feature | Status | Plan |
|---|---|---|
| Autocomplete / typeahead | ❌ | Debounced prefix search over titles + tags + people, with type badges. The single most-missed feature |
| Pagination / load more | ❌ | Infinite scroll or "load 12 more" — currently hard-capped at 12 results, forever |
| Result explanations | ❌ | "matched: cyberpunk · dystopia" replacing the fake % |
| Search-as-you-type preview | ❌ | Instant results at 3+ chars |
| Recent searches | partial | Exists as a nav panel; should be a dropdown under the input |
| Trending searches | ❌ | Aggregate `search_history` — free social proof |
| Zero-result recovery | ❌ | "No matches — did you mean…?" + relax the tightest filter automatically |
| Keyboard shortcuts | ❌ | `/` focus, `↑↓` navigate, `Enter` open, `Esc` close |
| Share a search | partial | URL state works; add an OG image per search |
| Voice search | ❌ | Web Speech API — ~20 lines, demos beautifully |

### 11.2 Filters
Replace the 4 hard-coded buttons with the tag system (§8.6): multi-select tag chips grouped by namespace with live counts, a year range slider, a runtime slider, a provider picker ("only what I can actually watch"), a language picker, a certification picker, and sort by relevance / rating / popularity / newest / runtime.

### 11.3 🎲 Random — more prompts, better prompts
Today: **6 hard-coded strings** in [page.tsx:20](frontend/app/page.tsx#L20).

Replace with a **generative prompt engine**:

1. **A curated pool of 150–300 handwritten prompts**, grouped by intent:
   - *Vibe*: "rain-soaked melancholy", "cozy autumn evening", "loud dumb fun", "quietly devastating", "beautiful and mean", "hopeful sci-fi that isn't naive"
   - *Scenario*: "something to watch at 3am", "a film for a first date that isn't a rom-com", "hungover Sunday", "background noise while I work", "to watch with my mum"
   - *Structural*: "one-location thriller", "single-take", "unreliable narrator", "non-linear timeline", "no dialogue for the first 20 minutes"
   - *Aesthetic*: "neon-drenched", "sun-bleached western", "brutalist and cold", "shot on film, feels like memory"
   - *Genre-bending*: "horror that's secretly a comedy", "sci-fi that's actually about grief", "sports anime for people who hate sports"
   - *Specific hungers*: "found family", "the villain is right", "slow-burn romance with actual tension", "competence porn", "heist where the plan goes wrong"

2. **A combinatorial generator** — `{mood} × {genre} × {era/origin}` templates drawing from the **live tag vocabulary**, so as the catalogue grows the dice get smarter automatically. `"{mood:bleak} {origin:korea} {genre:thriller}"` → *"bleak Korean thriller"*.

3. **Never repeat within a session** — keep a seen-set, reshuffle when exhausted. Today it can roll the same prompt twice in a row out of six.

4. **Show the prompt animating into the input** before it fires — it reads as intentional rather than random.

5. **A "surprise me" mode** distinct from the dice: pick a genuinely random *title* from the catalogue weighted by quality, and open its detail page. Different feature, one line of SQL, very sticky.

6. **Time/context aware** — late-night dice favour `mood:unsettling`; Sunday-morning dice favour `mood:cozy`. Small touch, feels magic.

### 11.4 Discovery surfaces beyond search
- **Tag pages** (`/tag/cyberpunk`) — browsable, SEO-indexable
- **Person pages** (`/person/christopher-nolan`) — filmography + similar directors
- **Franchise pages** — watch order for the MCU, Dune, Star Wars
- **"Mood board"** home surface — pick a mood tile, get a shelf
- **Comparison view** — two titles side by side with shared tags highlighted
- **Watchlist enhancements** — mark watched, rate 1–5, add notes, sort, export
- **Lists** — user-created public lists ("my top 20 sci-fi"), shareable

### 11.5 Polish & correctness
- **Accessibility**: cards are `<div onClick>` with no keyboard handler or role; modals don't trap focus or close on `Esc`; the theme toggle has no visible label; the neural canvas ignores `prefers-reduced-motion`. All easy, all currently failing.
- **Loading states**: skeleton cards instead of the "NEURAL SCAN IN PROGRESS…" text block
- **Error states**: distinguish *offline*, *waking up*, *no results*, and *server error*. Today all four render as "NO PATTERNS FOUND"
- **Mobile**: verify the sidebar, carousels and detail modal at 375px
- **De-duplicate** `DetailModal.tsx` and `title/[id]/page.tsx` into one shared `<TitleDetail>` (~80% identical today)
- **Break up `page.tsx`** — 322 lines holding four views and all their state
- Strip the literal `"Genres: …"` string from displayed descriptions (§4.9)

---

## 12. 🧹 Cleanup — what to delete

~2,700 lines of dead weight:

| Path | Why | Action |
|---|---|---|
| `docs/` (1,816 lines) | The old vanilla-JS frontend, fully superseded by `frontend/` | **Delete** (tag the commit first if you want it in history) |
| `harvest_popular.py` (166) | Superseded by `build_catalogue.py` | Delete |
| `expand_dataset.py` (214) | Superseded | Delete |
| `enrich_data.py` (124) | Superseded | Delete |
| `clean_dataset.py` (104) | One-off fix, already applied | Delete |
| `build_indexes.py` (92) | Superseded — `build_catalogue.ensure_collection` creates the indexes | Delete |
| `ingest.py` (158) | Only `stable_id()` survives, and it's broken (§4.3) | Delete; move a fixed id fn into the new pipeline |
| `dataset.csv` (12 MB) | Legacy MAL seed, no longer read by anything | Delete from the tree |
| `raw_data/AnimeList.csv` | Original dump | Delete from the tree |
| `main.py:_interleave()` | Never called | Delete |
| `engine.py:enrich_detail()` | Mostly unreachable | Keep only if legacy rows exist; otherwise delete |
| `frontend/public/*.svg` (next/vercel/file/globe/window) | Next.js scaffold defaults | Delete |

Repo `.git` is 8.7 MB — small enough that history rewriting isn't worth it. Just remove from HEAD.

---

## 13. 🗺️ Roadmap

### Phase 0 — Stop the bleeding (this week, ~1 day total)
| # | Task | Effort |
|---|---|---|
| 0.1 | Diagnose the Qdrant cluster — deleted or empty? | 15 min |
| 0.2 | **Keep-alive cron** pinging `/` every 10 min | 15 min |
| 0.3 | **Real `/health`** — fails if point count is 0 | 30 min |
| 0.4 | Frontend timeout + retry + "waking up" state | 1 hr |
| 0.5 | Health re-check on focus/visibilitychange | 20 min |
| 0.6 | JWT → 30 days + on-401 fall back to anonymous search | 1 hr |
| 0.7 | Rebuild the catalogue into the current schema so the site works *today* | 1–2 hrs |
| 0.8 | Expand the random pool to 150+ prompts | 30 min |
| 0.9 | Delete the dead code (§12) | 30 min |

### Phase 1 — Durable foundation (week 2)
| # | Task | Effort |
|---|---|---|
| 1.1 | Migrate the vector store to **Neon + pgvector** (§6) | 1–2 days |
| 1.2 | Hybrid search in SQL: HNSW + `tsvector` + RRF | 1 day |
| 1.3 | Embedded fallback index (deploy artifact) | half day |
| 1.4 | Fix `stable_id` → `tmdb_kind:tmdb_id` | 1 hr |
| 1.5 | Alembic migrations | half day |
| 1.6 | Retrieval regression test — 30 queries, expected top-3 | half day |
| 1.7 | Sentry + structured logging | half day |
| 1.8 | Rate limiting + CORS lockdown + `/login` throttle | half day |

### Phase 2 — The dataset (weeks 3–4) ⭐ biggest payoff
| # | Task | Effort |
|---|---|---|
| 2.1 | Four-stage resumable pipeline (§7.7) | 2 days |
| 2.2 | TMDB ID exports → break the 500-page wall | half day |
| 2.3 | Pull keywords, credits, alt titles, providers, certifications | 1 day |
| 2.4 | IMDb dataset join | half day |
| 2.5 | AniList integration for anime tags | 1 day |
| 2.6 | Full cold build → ~110k titles | 2–4 hrs runtime |
| 2.7 | Nightly incremental via `/changes` | half day |

### Phase 3 — Tags & retrieval (week 5)
| # | Task | Effort |
|---|---|---|
| 3.1 | Four-layer tag model + namespaces (§8) | 1 day |
| 3.2 | LLM mood pass over the catalogue (Haiku) | 1 day + ~$5 |
| 3.3 | Restructure the embedding document (§7.8) | half day |
| 3.4 | Re-embed with a multilingual model | 1 day + runtime |
| 3.5 | **ONNX cross-encoder rerank back on** (§9.1) | half day |
| 3.6 | Query understanding + filter extraction | 1 day |
| 3.7 | Tag-similarity as a third RRF channel | half day |

### Phase 4 — The product (weeks 6–7)
| # | Task | Effort |
|---|---|---|
| 4.1 | Tag chips, tag filter bar, tag pages | 2 days |
| 4.2 | Autocomplete / typeahead | 1 day |
| 4.3 | Pagination / infinite scroll | half day |
| 4.4 | Match explanations replacing %MATCH | half day |
| 4.5 | Generative random-prompt engine (§11.3) | half day |
| 4.6 | Streaming LLM explanations | 1 day |
| 4.7 | Conversational refinement | 2 days |
| 4.8 | Person / franchise pages, SSR + SEO | 2 days |
| 4.9 | Accessibility pass | 1 day |
| 4.10 | De-duplicate detail views, split `page.tsx` | 1 day |

### Phase 5 — Personalization & polish (week 8+)
Taste vectors, multi-cluster For You, onboarding, watched/rated/notes, public lists, comparison view, A/B harness for retrieval changes.

---

## 14. ❓ Open questions — decisions to make

| # | Question | Why it matters | My recommendation |
|---|---|---|---|
| 1 | **Is the Qdrant cluster deleted, or just empty?** | Determines whether Phase 0.7 is a re-upsert or a full rebuild | Check the console first thing |
| 2 | **Neon free tier storage limit?** | 110k titles with full payloads may exceed the free plan. Options: trim payloads (move cast/providers to a side table), float16 vectors, or the paid tier | Measure at 10k rows, extrapolate, then decide |
| 3 | **Budget — £0 hard, or £5–25/mo?** | £0 → HF Spaces / Oracle Always Free + Neon. £25 → Qdrant Cloud paid + Render paid and the problem disappears | Assume £0; design so money is optional |
| 4 | **Catalogue size target** | 110k is ambitious. 30k covers ~99% of what anyone searches for. More rows = more storage, more build time, more noise in results | Build the pipeline for 110k, ship 30k first, scale after measuring quality |
| 5 | **Keep Qdrant at all?** | pgvector consolidation is cleaner, but "Qdrant" on a CV is recognizable | pgvector primary; keep a Qdrant adapter behind the interface so you can still say you've done both |
| 6 | **Anime as its own vertical?** | Anime metadata (AniList tags, studios, seasons, source manga) is much richer than TMDB's. Could justify a dedicated experience | Unify the tag model; let anime carry extra fields |
| 7 | **LLM budget for mood tagging** | ~$5 one-time at 110k titles with Haiku. Free-tier models would take days and be inconsistent | Spend the $5 |
| 8 | **Ratings — TMDB, IMDb, or both?** | They disagree. Two ratings side by side is more honest and more useful | Store both, display both, rank on a blend |
| 9 | **Region for providers** | US-only today. Multi-region multiplies payload size | Detect region client-side, store US/GB/IN, degrade to US |
| 10 | **Is `docs/` worth keeping as a "v1 archive"?** | It's 1,816 dead lines but it *is* project history | Tag the commit, delete from HEAD |

---

## 15. 📊 How we'll know it's actually better

Right now there is **no way to tell if a change to retrieval helped or hurt**. That has to change before you start tuning models, or you'll be guessing.

| Metric | How | Target |
|---|---|---|
| **Cold-start time** | Cron-ping timing | **< 2s** (from 42.6s) |
| **Empty-result rate** | Log `len(results) == 0` per query | **< 2%** |
| **Search latency p95** | Server timing | **< 400ms** warm |
| **Exact-title recall@1** | 200 known titles, is it #1? | **> 98%** |
| **Vibe-query relevance** | 50 hand-labelled vibe queries, nDCG@10 | Baseline now, then improve |
| **Click-through @ rank 1** | Log clicks | Trend up |
| **Catalogue coverage** | % of a "top 500 films/shows" list present | **100%** |
| **Tag coverage** | % of titles with ≥ 5 tags | **> 95%** |
| **Uptime** | Sentry / uptime monitor | **> 99%** |

**Build the labelled query set before Phase 3.** 50 vibe queries with hand-picked expected results, checked in as a fixture. It takes an afternoon and it's the difference between engineering and vibes-based tuning.

---

## 16. The three sentences that matter

1. **The site is down right now** because the vector store is empty — and the code cannot tell you that, because every failure returns `[]`. Fix the data, then fix the silence.
2. **Your dataset is the ceiling on everything else.** 6,850 titles with a comma-joined genre string will never feel magical no matter how good the model is. TMDB keywords + AniList tags + an LLM mood pass over 100k titles is the change that transforms the product.
3. **Stop depending on things that delete your data.** Consolidate onto Neon (which already holds your users), keep an embedded index as insurance, and no free-tier reaping can take the site down again.
