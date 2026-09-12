# 🦅 Nexus Neural Search

> **A hybrid AI discovery engine for movies, TV, anime & documentaries — it nails the exact title *and* the vibe.** Semantic "mood" search and precise keywords are fused, reranked, and (optionally) reasoned over by an LLM that can only speak about *real* results.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs)](https://nextjs.org/)
[![pgvector](https://img.shields.io/badge/Postgres-pgvector-336791?style=for-the-badge&logo=postgresql)](https://github.com/pgvector/pgvector)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

# [Live Website](https://nexus-neural-search-pi.vercel.app/)

---

## 📖 Overview

Ask Nexus for *"cyberpunk anime about identity"* **or** an exact title like *"Cyberpunk: Edgerunners"* — it handles both. Every query runs **hybrid retrieval** (dense vectors + Postgres full-text), fuses the two with Reciprocal Rank Fusion, and can **rerank with a cross-encoder** where there is memory for it. On top of that sits a full product: user accounts, a personalized home feed, wishlist/favourites, search history, view tracking, and flagship detail pages with trailers, cast, and where-to-watch.

The catalogue is built from TMDB's daily id exports and quality-gated on artwork, overview and vote count, then enriched with AniList's rank-weighted tags and IMDb ratings. Every card has a poster that passed a real resolution and aspect-ratio bar, so artwork is correct by construction and detail pages load instantly.

**Everything lives in one PostgreSQL database** — the catalogue, the vectors, and the users. There is no separate vector service to provision, keep awake, or have deleted overnight.

---

## ✨ How retrieval works

Every `internal` search runs as **one SQL query against one database**.

1. **Dense channel** — the query is embedded with `BAAI/bge-small-en-v1.5` (384-dim, ONNX), and `pgvector` HNSW returns the nearest titles *with their cosines*.
2. **Lexical channel** — Postgres `tsvector` + `ts_rank_cd` catches exact words the vectors miss: a director's name, a studio, an exact title. It brings stemming, so "haunting" matches "haunted".
3. **Fusion** — the two rankings are fused with **weighted Reciprocal Rank Fusion** in `backend/ranking.py`, with per-channel weights.
4. **Cross-encoder rerank (optional)** — `Xenova/ms-marco-MiniLM-L-6-v2` reranks the shortlist and its *ordering*
   joins the fusion as a third channel. It runs on the same ONNX runtime as the encoder, so unlike the old torch
   build it can run in production at all — but it costs a measured 106 MB on top of the encoder's 186 MB, which does
   not fit a 512 MB instance alongside the app. `ENABLE_RERANK` therefore defaults to **off**; the public deployment
   runs the two retrieval channels only. Turn it on where there is ~1 GB.
5. **Title pinning** — token-set similarity **and** Postgres trigram similarity, so `fellowship of the ring` (partial) and `spirted away` (typo) both resolve.
6. **Ranking** — relevance blended with a **vote-shrunk Bayesian quality prior** (the IMDb weighted-rating formula), log-linearly, so a popular title that *doesn't* match can never climb over one that does, while quality breaks ties.

### The %MATCH number

The badge is a monotone function of the **calibrated dense cosine**, so **90% means the same strength of match in every query**.

It is deliberately *not* derived from list position, from RRF, or from the cross-encoder logit — because none of those measure content. A rank is content-blind: rank 3 of a pool of masterpieces and rank 3 of a pool of junk are the same number, which is why the old badge gave the top hit 99% however badly the query went.

The cross-encoder was measured and rejected for the badge for the same reason. Over 70 labelled query-document pairs on this catalogue its logits came out:

| | median | range |
|---|---|---|
| relevant | **-10.91** | -11.33 … +9.54 |
| irrelevant | **-11.33** | -11.43 … -10.50 |

Near-total overlap, both deep in the negative tail — `sigmoid()` maps essentially everything to zero. It answers "does this passage answer this question", which is not what a catalogue is asked. Its *ordering* is still better than vector order alone (dropping it makes `Denis Villeneuve` return Spirited Away first), so it is fused as a rank and never as a score.

The cosine band was measured the same way — relevant pairs centre on **0.63**, irrelevant on **0.43** — and `COSINE_FLOOR`/`COSINE_CEIL` are set from those numbers rather than guessed. Real output today:

```
cyberpunk dystopia about identity   62%  Cyberpunk: Edgerunners
                                    24%  Blade Runner 2049
                                    12%  Paddington 2
quantum accounting seminar          12%  …everything floored
```

Every constant is an env override, and `test_ranking.py` asserts the properties that must survive any retuning.

`GET /` reports what is actually live (`rerank`, `dense_model`, `store`, and the real **title count**), and returns **503** when the catalogue is missing or empty rather than reporting a healthy service that silently answers every search with `[]`.

### 🧠 Grounded RAG ("API" mode)

The Nemotron mode (`model: "api"`, via OpenRouter `nvidia/nemotron-nano-12b-v2-vl`) is **grounded**: the LLM only reorders and explains the **real** hits from hybrid retrieval — it can't invent titles or fake posters. If the call fails, it transparently falls back to the hybrid ranking.

### 🎯 Search is pure; recommendations are personal

Explicit search returns **pure query relevance** — favouriting horror never bleeds into a "cyberpunk anime" query. Taste-based recommendation lives only in the home feed:

- **Your Favourites** — your saved wishlist.
- **For You** — a taste centroid built in SQL from your wishlist + likes + recent views, with dismissed titles subtracted (Rocchio feedback).
- **Because you liked X** — items similar to your top signal.
- **Recently Viewed** — your view history.
- **Top Movies / TV / Anime / Documentaries** — most-voted mainstream titles per category.

Logged-out visitors get the four "Top …" rows.

---

## 🛠️ Tech Stack

**Backend:** FastAPI · Uvicorn · **PostgreSQL + pgvector** (HNSW dense index, `tsvector` lexical index, GIN tag facets — one database for the catalogue *and* the users) · SQLAlchemy · `fastembed` ONNX (bge-small encoder + ms-marco cross-encoder, no torch) · JWT (python-jose) + passlib/bcrypt · OpenAI SDK → OpenRouter (Nemotron).

**Frontend:** **Next.js 16** (App Router, React 19, TypeScript) · React Context store · reactive HTML5 Canvas neural background · cyberpunk dark/light themes · standalone `/title/[id]` detail pages.

**Data:** TMDB (daily id exports, keywords, credits, images, providers) · **AniList** (rank-weighted anime tags) · **IMDb** (ratings + vote counts).

---

## 📂 Project Structure

```text
nexus-neural-search/
├── backend/
│   ├── main.py          # FastAPI routes: search, feed, auth, wishlist, history, detail
│   ├── engine.py        # Retrieval core: embeddings, hybrid search, recommend, RAG
│   ├── ranking.py       # Ranking math: RRF, Bayesian quality, match calibration
│   ├── auth.py          # JWT auth (SECRET_KEY from env)
│   ├── database.py      # Env-driven DB: SQLite local, Postgres via DATABASE_URL
│   ├── models.py        # User, WishlistItem, SearchHistory, Interaction
│   └── store.py         # Catalogue on pgvector: schema, hybrid SQL, facets
├── frontend/            # Next.js app (App Router)
│   ├── app/             # pages: / (search+feed), /title/[id] (detail), layout, globals.css
│   ├── components/      # Feed, Card, DetailModal, AuthModal, Chrome, NeuralBg
│   └── lib/             # api client, store (Context), types, image helpers, dice prompts
├── catalogue/           # Dataset pipeline: ids → fetch → normalise → index
│   ├── tmdb.py          # ID exports, detail fetch, poster/backdrop selection
│   ├── tags.py          # Four-layer tag model + namespaces
│   ├── enrich.py        # AniList ranked tags, IMDb ratings
│   ├── schema.py        # Identity, quality gates, retrieval document
│   └── build.py         # Resumable CLI (python -m catalogue.build all)
├── test_ranking.py      # Ranking math property tests
├── test_catalogue.py    # Pipeline tests (tags, gates, image selection)
├── test_store.py        # pgvector integration tests (real Postgres)
├── test_nexus.py        # Engine regression + smoke tests
├── bench_storage.py     # Measure real bytes-per-title
├── debug.py             # One-shot engine sanity check
├── .github/workflows/   # CI, hosted catalogue build, keep-alive ping
├── requirements.txt
├── .env.example         # Copy to backend/.env and fill in
└── README.md
```

---

## 🚀 Setup

### 1. Backend

```bash
git clone https://github.com/aarushch/Nexus-Neural-Search.git
cd Nexus-Neural-Search
python -m venv .venv && .venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Create `backend/.env`:

```env
TMDB_API_KEY=...            # themoviedb.org (catalogue build)
DATABASE_URL=...          # Neon (or any Postgres with pgvector + pg_trgm)
SECRET_KEY=...              # python -c "import secrets; print(secrets.token_hex(32))"
OPENROUTER_API_KEY=...      # optional, enables Nemotron "API" mode
# DATABASE_URL=postgresql://user:pass@host/db?sslmode=require   # optional; omit for local SQLite
```

### 2. Build the catalogue

**The build runs on GitHub's runners, not on your machine.** Add two repository
secrets (`TMDB_API_KEY`, `DATABASE_URL`), then **Actions → Build catalogue → Run
workflow**. It harvests TMDB, tags everything, embeds it and loads it straight
into Neon, then prints the title count and storage used in the run summary. A
weekly cron re-runs the cheap stages so ratings and providers stay current.

Nothing needs to stay on. Your laptop can be shut.

<details>
<summary>Running it locally instead</summary>

Four resumable stages. Each writes to `data/`, so the expensive one is paid once.

```bash
python -m catalogue.build all --target 50000 --recreate

# or stage by stage
python -m catalogue.build ids         # enumerate every TMDB id from the daily export
python -m catalogue.build fetch       # download details (resumable — safe to interrupt)
python -m catalogue.build normalise   # join AniList + IMDb, tag, quality-gate
python -m catalogue.build index       # embed + load into Postgres
```

</details>

**Why the stages matter.** The previous builder paged TMDB's `/discover`, which is
**hard-capped at page 500** — the ~6,850-title catalogue was a structural ceiling,
not a setting, and no `--pages` value could pass it. Stage 1 instead reads TMDB's
daily gzipped export of *every* id it holds (~1M movies, ~200k series), which is
what makes 100k+ possible.

`fetch` is the only expensive stage (one request per title, ~1–2 hours at 100k
with 24 workers) and it is cached and resumable, so `normalise` can be re-run for
free every time the tag rules change.

**Quality is enforced, not assumed.** A title is only indexed if it has a poster
that passes a real bar — TMDB's `/images` is fetched and every candidate is scored
on resolution (≥500px so `w500` is never an upscale), aspect ratio (2:3, anything
noticeably off-shape is rejected rather than CSS-cropped), community rating, and
textless artwork — plus a ≥40-character overview, a release date, and a vote floor.
`normalise` prints exactly how many titles each gate dropped and why.

**Composition is chosen, not inherited.** TMDB's export is ranked on *global*
popularity, which is nothing like the shape of this catalogue: a large slice of any
top-N is regional TV drama with a few hundred votes. So a foreign-language title has
to clear `FOREIGN_MIN_VOTES` (3,000) to earn a slot, while English titles and anime
stay on the ordinary floor. The filter is on **reach, not origin** — *Parasite*,
*Memories of Murder* and *Dark* clear it several times over; a soap nobody outside
its home market has rated does not. On top of that, each category has a guaranteed
floor before the global fill (`QUOTAS`: 46% film, 22% anime, 20% TV, 4% documentary),
because ranking one pool on votes erases whole categories.

Anime gets enumerated from `/discover` in full rather than rationed: TMDB holds 5,427
Japanese animated series and 6,007 films, so the entire universe of it is ~11.4k
candidates and there is nothing to ration — the anime floor is deliberately set above
that supply, which makes it read as "take every one that passes". `normalise` prints
the resulting language and category mix.

**Tags** come from TMDB `/keywords` (a curated theme vocabulary the old build never
requested), AniList's **rank-weighted** tags for anime (`Cyberpunk 95, Tragedy 85` —
the weight decides what reaches the embedding), and derived facets for era, origin,
language, studio, franchise, certification and streaming provider. Everything is
namespaced (`theme:dystopia`, `people:denis-villeneuve`, `studio:a24`) and indexed
as a GIN-indexed array, so it is filterable, countable and clickable rather than prose glued
onto the overview.

### 3. Run the API

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 10000 --reload
```

> ⚠️ Run from the **repo root** and use the module path **`backend.main:app`** (not `main:app`).

### 4. Run the frontend

```bash
cd frontend
npm install
npm run dev                 # http://localhost:3000
```

`frontend/.env.local` points the UI at the API:

```env
NEXT_PUBLIC_API_URL=http://localhost:10000
```

You need **both** processes running — if the backend is down, the UI shows **OFFLINE** and searches return nothing.

---

## 🗄️ Durable database (production)

Local dev defaults to SQLite (`backend/freeme.db`). Set `DATABASE_URL` to a hosted Postgres and every account, wishlist, search, and interaction persists there instead — this deployment runs on **[Neon](https://neon.tech)** (free, serverless). Supabase or Render Postgres work identically. Paste the connection string into `DATABASE_URL`; tables auto-create on startup (no migrations). Legacy `postgres://` URLs are normalized automatically, and `pool_pre_ping` revives the connections Neon drops when idle. Do **not** enable Neon Auth — Nexus uses its own JWT auth.

---

## 🔌 API

| Method | Route | Notes |
|---|---|---|
| `POST` | `/recommend` | `{text, top_k, model, category?, min_rating?}` — `model` = `internal` (hybrid) or `api` (grounded RAG) |
| `POST` | `/recommend/personalized` | Auth — logs search history, returns **pure** results |
| `POST` | `/similar` | `{id}` — vector neighbours of one item |
| `POST` | `/login` · `/signup` · `GET /me` | JWT auth |
| `GET` | `/feed` · `/discover` | personalized home feed / anonymous landing feed |
| `GET` | `/recommendations/foryou` | taste-based recommendations |
| `GET/POST/DELETE` | `/wishlist...` | Auth — favourites |
| `POST` | `/interactions` | Auth — `view` / `like` / `dismiss` |
| `GET` | `/history/search` | Auth — recent searches |
| `GET` | `/title/{id}` | full detail: trailer, cast, crew, providers, backdrop |
| `GET` | `/facets/{namespace}` | tag counts for the filter bar — `theme`, `mood`, `genre`, `era`, `origin`, `people`, `studio`, `franchise`, `where`, `audience` |
| `GET` | `/random` | one well-known title at random ("surprise me") |
| `GET` | `/` | health: rerank/store/model state and the real title count; **503** when the catalogue is empty |

---

## ⚙️ Hosting notes

Three services, one of them optional: **Vercel** (frontend) · **Render** (API) · **Neon** (Postgres — catalogue *and* users).

Query-time embedding runs in-process on ONNX, not torch. Measured resident cost: **186 MB** for the encoder and a
further **106 MB** if the cross-encoder is enabled, against ~600 MB for the equivalent torch build. `ONNX_THREADS`
defaults to 1 because ONNX Runtime gives every intra-op thread its own memory arena and a container reports the
host's core count — left alone, that multiplies resident memory until the worker is OOM-killed mid-request.

### Why one database

The catalogue used to live in a Qdrant Cloud cluster. Free-tier vector clusters are suspended and eventually **deleted** after a period of inactivity, and a portfolio site that sits idle between visits is exactly the profile that gets reaped — it happened twice, and each time the API stayed up and cheerfully returned zero results for everything.

Consolidating onto Postgres removed that failure mode rather than working around it:

- Neon persists storage even when compute auto-suspends, and `pool_pre_ping` handles the wake.
- One set of credentials, one thing to back up, one thing that can break.
- The dense cosine survives fusion, which is what makes an honest %MATCH possible at all.
- Tag facets become a GIN-indexed array with real `COUNT`s — awkward in Qdrant, native here.

### Storage

Neon's free branch is 0.5 GB. The first build measured **6.5 KB per title** — 332 MB for 50,000 — which put 100k on a paid plan. Four changes brought that down without dropping a single feature:

| change | measured on 20 real titles |
|---|---|
| detail blob holds only what the detail view renders (cast, providers), gzipped | 2,859 B → **768 B** |
| posters stored as bare TMDB paths, both sizes derived on read | 190 B → **64 B** |
| `original_title` stored only when it differs from the title | empties the column *and* most of its trigram index |
| `types` / `forms` GIN indexes dropped — never in a `WHERE` clause | two indexes gone |

Crew, alternative titles and the IMDb/AniList ids were being stored for nobody: they are consumed at build time, into the embedded document, and never read back.

The on-disk saving is smaller than the raw one, because Postgres was already compressing the old blob once its row crossed 2 KB — the new one is small enough to sit inline. Expect roughly **5.5 KB per title**, so about **90k titles free**.

You do not have to guess, and you must not overshoot: an over-quota database stops accepting writes. `catalogue.build index` loads best-scoring titles first and stops on `STORAGE_BUDGET_MB` (default 440, leaving room for the user tables and the write-ahead log). Set `--target` high and the catalogue fills the disk with the best of what qualified.

Run `python bench_storage.py` against a scratch database to re-measure — below a few thousand rows the fixed costs dominate and any estimate is meaningless.

### Cold starts

Render's free tier spins the API down after 15 minutes idle, and the cold boot takes ~40 s. Three things absorb that:

- `.github/workflows/keepalive.yml` pings `/` every 5 minutes so the instance stays resident. Actions can skip scheduled runs under load — pair it with a free pinger at [cron-job.org](https://cron-job.org) or [UptimeRobot](https://uptimerobot.com) if you want a hard guarantee.
- The API client retries once with a 75 s budget, and the UI says **WAKING** instead of spinning forever.
- The health probe re-runs on tab focus and `visibilitychange`, so the **OFFLINE** badge clears itself when you come back rather than sticking until a reload.

### Required env in production

`SECRET_KEY` is **mandatory** whenever `DATABASE_URL` points at anything other than local SQLite — the app refuses to boot without it rather than falling back to a shared default that would make every token forgeable.

Access tokens last 30 days (`ACCESS_TOKEN_EXPIRE_MINUTES`). Set `CORS_ORIGINS` to your real frontend origin in production; while it stays `*`, credentialed CORS is disabled because `*` plus credentials is a combination browsers reject.

---

## 📜 License

MIT.
