# 🦅 Nexus Neural Search

> **A hybrid AI discovery engine for movies, TV, anime & documentaries — it nails the exact title *and* the vibe.** Semantic "mood" search and precise keywords are fused, reranked, and (optionally) reasoned over by an LLM that can only speak about *real* results.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=for-the-badge&logo=nextdotjs)](https://nextjs.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-9cf?style=for-the-badge)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

---

## 📖 Overview

Ask Nexus for *"cyberpunk anime about identity"* **or** an exact title like *"Cyberpunk: Edgerunners"* — it handles both. Every query runs **hybrid retrieval** (dense semantic vectors + BM25 keyword matching), fuses the two with Reciprocal Rank Fusion, and **reranks with a cross-encoder** for final accuracy. On top of that sits a full product: user accounts, a personalized home feed, wishlist/favourites, search history, view tracking, and flagship detail pages with trailers, cast, and where-to-watch.

The catalogue is **~6,850 mainstream titles**, each validated and enriched directly from TMDB (correct poster + backdrop, real overview/rating/genres, runtime, YouTube trailer, top cast, and streaming providers) — all stored in the Qdrant payload, so cards are correct by construction and detail pages load instantly.

---

## ✨ How retrieval works

Every `internal` search:

1. **Dense embedding** — the query is embedded with `BAAI/bge-small-en-v1.5` (384-dim), the *same* model used for the corpus.
2. **Sparse embedding** — a BM25 sparse vector (FastEmbed, IDF modifier) captures exact keyword/title hits.
3. **Fusion** — Qdrant's Query API fuses dense + sparse with **Reciprocal Rank Fusion (RRF)**.
4. **Cross-encoder rerank** — `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores the fused candidates.
5. **Exact-title pin** — a normalized exact title match is pinned to the top.
6. **Calibrated match %** — the cross-encoder relevance logit is temperature-scaled through a sigmoid into an honest 0–99 match score (pins floored, everything else capped), so the number reflects real relevance instead of raw list rank.

Everything runs **locally** — no per-query external inference API. Qdrant itself is Qdrant Cloud.

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

**Data:** TMDB (harvest + validation + enrichment).

---

## 📂 Project Structure

```text
nexus-neural-search/
├── backend/
│   ├── main.py          # FastAPI routes: search, feed, auth, wishlist, history, detail
│   ├── engine.py        # Retrieval core: embeddings, hybrid search, rerank, recommend, RAG
│   ├── auth.py          # JWT auth (SECRET_KEY from env)
│   ├── database.py      # Env-driven DB: SQLite local, Postgres via DATABASE_URL
│   └── models.py        # User, WishlistItem, SearchHistory, Interaction, MediaDetail
├── frontend/            # Next.js app (App Router)
│   ├── app/             # pages: / (search+feed), /title/[id] (detail), layout, globals.css
│   ├── components/      # Feed, Card, DetailModal, AuthModal, Chrome, NeuralBg
│   └── lib/             # api client, store (Context), types, image helpers
├── build_catalogue.py   # Authoritative TMDB pipeline: harvest → validate → enrich → Qdrant
├── ingest.py            # stable_id helper + legacy CSV ingester
├── dataset.csv          # Legacy seed data
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

```bash
python build_catalogue.py                 # deep harvest, validate + enrich, upsert into Qdrant
python build_catalogue.py --recreate      # wipe + rebuild the collection clean
python build_catalogue.py --pages 120 --min-votes 80 --workers 20   # go wider
```

This harvests mainstream titles across movies / TV / anime (genre 16 + Japanese origin, so Netflix anime like Edgerunners are caught) / documentaries, validates each (must have a poster + real overview), enriches with backdrop, trailer, top cast, and providers, and writes everything into the Qdrant payload.

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

---

## 📜 License

MIT.
