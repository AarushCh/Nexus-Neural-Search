# 🦅 Nexus Neural Search

> **A hybrid AI discovery engine for movies, TV, anime & documentaries — it nails the exact title *and* the vibe.** Semantic "mood" search and precise keywords are fused, reranked, and (optionally) reasoned over by an LLM that can only speak about *real* results.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-9cf?style=for-the-badge)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

# [Live Website](https://nexus-neural-search-pi.vercel.app/)

---

## 📖 Overview

Ask Nexus for *"cyberpunk anime about identity"* **or** an exact title like *"Cyberpunk: Edgerunners"* — it handles both. Every query runs **hybrid retrieval** (dense semantic vectors + BM25 keyword matching), fuses the two with Reciprocal Rank Fusion, and **reranks with a cross-encoder** for final accuracy. On top of that sits a full product: user accounts, a personalized home feed, wishlist/favourites, search history, view tracking, and flagship detail pages with trailers, cast, and where-to-watch.

The catalogue targets **100,000+ titles**, built from TMDB's daily id exports and quality-gated on artwork, overview and vote count, then enriched with AniList's rank-weighted tags and IMDb ratings. Every card has a poster that passed a real resolution and aspect-ratio bar, so artwork is correct by construction and detail pages load instantly.

---

## ✨ How retrieval works

Every `internal` search:

1. **Dense embedding** — the query is embedded with `BAAI/bge-small-en-v1.5` (384-dim), the *same* model used for the corpus.
2. **Sparse embedding** — a BM25 sparse vector (FastEmbed, IDF modifier) captures exact keyword/title hits.
3. **Fusion** — dense and sparse are queried **separately** and fused with **weighted Reciprocal Rank Fusion** in `backend/ranking.py`. Qdrant's built-in RRF returns only a fused rank, which discards the dense cosine — and that cosine is the one absolute relevance signal available without a reranker. Doing it client-side also buys per-channel weights, which Qdrant's fusion does not expose.
4. **Cross-encoder rerank** — `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores the candidates. **Off by default** (`ENABLE_RERANK=false`): it needs `torch` + `sentence-transformers` (~600 MB) and OOMs on a 512 MB host. Turn it on wherever you have the RAM.
5. **Title pinning** — token-set similarity, not string prefix, so `fellowship of the ring` and `star wars a new hope` resolve to the right title.
6. **Ranking** — relevance is blended with a **vote-shrunk Bayesian quality prior** (the IMDb weighted-rating formula), log-linearly, so a popular title that *doesn't* match can never climb over one that does, while quality breaks ties between equally relevant results.

### The %MATCH number

The badge is a monotone function of genuine, content-based relevance — the cross-encoder logit when the reranker is on, the calibrated dense cosine otherwise — so **90% means the same strength of match in every query**.

It is deliberately *not* derived from list position or from RRF. A rank-based score is content-blind: rank 3 of a pool of masterpieces and rank 3 of a pool of junk are the same number, which is why the old badge gave the top result 99% no matter how badly the query had gone. A hopeless query now honestly reads in the twenties all the way down.

Every constant (`RRF_K`, channel weights, `QUALITY_WEIGHT`, the Bayesian prior, the cosine calibration band) is an env override, and `test_ranking.py` asserts the properties that must survive any retuning.

Everything runs **locally** — no per-query external inference API. Qdrant itself is Qdrant Cloud.

`GET /` reports which of these are actually live (`rerank`, `sparse`, `dense_model`, and the real collection **point count**), and returns **503** when the index is missing or empty rather than reporting a healthy service that silently answers every search with `[]`.

### 🧠 Grounded RAG ("API" mode)

The Nemotron mode (`model: "api"`, via OpenRouter `nvidia/nemotron-nano-12b-v2-vl`) is **grounded**: the LLM only reorders and explains the **real** hits from hybrid retrieval — it can't invent titles or fake posters. If the call fails, it transparently falls back to the hybrid ranking.

### 🎯 Search is pure; recommendations are personal

Explicit search returns **pure query relevance** — favouriting horror never bleeds into a "cyberpunk anime" query. Taste-based recommendation lives only in the home feed:

- **Your Favourites** — your saved wishlist.
- **For You** — Qdrant `recommend` using your wishlist + likes + recent views as positive vectors (minus anything you dismissed).
- **Because you liked X** — items similar to your top signal.
- **Recently Viewed** — your view history.
- **Top Movies / TV / Anime / Documentaries** — most-voted mainstream titles per category.

Logged-out visitors get the four "Top …" rows.

---

## 🛠️ Tech Stack

**Backend:** FastAPI · Uvicorn · **Qdrant Cloud** (named dense + BM25 sparse vectors) · SQLAlchemy over **SQLite (local) / Postgres (prod)** · `sentence-transformers` (bge-small + ms-marco cross-encoder) · `fastembed` (BM25) · `torch` (cpu) · JWT (python-jose) + passlib/bcrypt · OpenAI SDK → OpenRouter (Nemotron).

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
│   └── models.py        # User, WishlistItem, SearchHistory, Interaction, MediaDetail
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
├── test_nexus.py        # Engine regression + smoke tests
├── debug.py             # One-shot engine sanity check
├── .github/workflows/   # CI (tests + frontend build) and the keep-alive ping
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
QDRANT_URL=...              # cloud.qdrant.io cluster URL
QDRANT_API_KEY=...
SECRET_KEY=...              # python -c "import secrets; print(secrets.token_hex(32))"
OPENROUTER_API_KEY=...      # optional, enables Nemotron "API" mode
# DATABASE_URL=postgresql://user:pass@host/db?sslmode=require   # optional; omit for local SQLite
```

### 2. Build the catalogue

Four resumable stages. Each writes to `data/`, so the expensive one is paid once.

```bash
python -m catalogue.build all --target 120000 --recreate

# or stage by stage
python -m catalogue.build ids         # enumerate every TMDB id from the daily export
python -m catalogue.build fetch       # download details (resumable — safe to interrupt)
python -m catalogue.build normalise   # join AniList + IMDb, tag, quality-gate
python -m catalogue.build index       # embed + upsert into Qdrant
```

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

**Tags** come from TMDB `/keywords` (a curated theme vocabulary the old build never
requested), AniList's **rank-weighted** tags for anime (`Cyberpunk 95, Tragedy 85` —
the weight decides what reaches the embedding), and derived facets for era, origin,
language, studio, franchise, certification and streaming provider. Everything is
namespaced (`theme:dystopia`, `people:denis-villeneuve`, `studio:a24`) and indexed
as a Qdrant keyword facet, so it is filterable and clickable rather than prose glued
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
| `POST` | `/similar` | `{id}` — Qdrant recommend from one item |
| `POST` | `/login` · `/signup` · `GET /me` | JWT auth |
| `GET` | `/feed` · `/discover` | personalized home feed / anonymous landing feed |
| `GET` | `/recommendations/foryou` | taste-based recommendations |
| `GET/POST/DELETE` | `/wishlist...` | Auth — favourites |
| `POST` | `/interactions` | Auth — `view` / `like` / `dismiss` |
| `GET` | `/history/search` | Auth — recent searches |
| `GET` | `/title/{id}` | full detail: trailer, cast, providers, backdrop (served from payload) |

---

## ⚙️ Hosting notes

Query-time embedding + reranking run in-process, so the host needs RAM for `torch` + bge-small + the cross-encoder (~600 MB resident). Set `ENABLE_RERANK=false` on tiny instances to run dense + BM25 only. Recommended split: **Vercel** (frontend) · **Render** (API) · **Qdrant Cloud** (vectors) · **Neon** (Postgres).

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
