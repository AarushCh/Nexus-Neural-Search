# 🦅 Nexus Neural Search

## 🔗 Live Website

➡️ [https://aarushch.github.io/Nexus-Neural-Search/](https://aarushch.github.io/Nexus-Neural-Search/)

![Project Banner](https://placehold.co/1200x400/050505/00f3ff?text=NEXUS+INTELLIGENCE+ENGINE)

> **A hybrid AI search engine that nails the exact match *and* the close ones — semantic "vibes" and precise keywords, fused and reranked.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-9cf?style=for-the-badge)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

## 📖 Overview

**Nexus** is an AI recommendation engine for movies/anime. Ask for *"cyberpunk anime with
philosophical themes about identity"* **or** an exact title — the engine handles both by
running **hybrid retrieval** (dense semantic vectors + BM25 keyword matching), fusing the two,
and then **reranking with a cross-encoder** for accuracy.

## ✨ How the retrieval works

The pipeline for every `internal` search:

1. **Dense embedding** — the query is embedded with `BAAI/bge-small-en-v1.5` (384-dim), the
   *same* model used to embed the corpus. (The previous version embedded the corpus and the
   query with two different models — a silent bug that made scores meaningless. Fixed.)
2. **Sparse embedding** — a BM25 sparse vector (via FastEmbed) captures exact keyword/title hits.
3. **Fusion** — Qdrant's Query API fuses dense + sparse results with **Reciprocal Rank Fusion (RRF)**.
4. **Cross-encoder rerank** — `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores the fused
   candidates for final ordering.
5. **Exact-title pin** — a normalized exact title match is pinned to the top.

Everything runs **locally** (no per-query external inference API). Qdrant itself is **Qdrant Cloud**.

### 🧠 Grounded RAG ("API" mode)

The Nemotron mode (`model: "api"`) is **grounded**: the LLM only reorders and explains the
**real** hits returned by hybrid retrieval — it can't invent titles or fake posters. If the LLM
call fails, it transparently falls back to the hybrid results.

### ❤️ Real personalization

`/recommend/personalized` uses your **wishlist vectors** as positive examples in Qdrant's
`recommend` query, then interleaves those with your text-query results. The old version
claimed this but did nothing.

### 💻 Frontend

Unchanged: zero-framework Vanilla JS/CSS, reactive HTML5 Canvas neural background, dark/light
themes, served from `/docs` on GitHub Pages.

---

## 🛠️ Tech Stack

**Backend:** FastAPI · Uvicorn · **Qdrant Cloud** (dense + sparse vectors) · SQLite/SQLAlchemy
(auth + wishlist) · `sentence-transformers` (bge-small + ms-marco cross-encoder) · `fastembed`
(BM25) · `torch` (cpu) · OpenAI SDK → OpenRouter (Nemotron).

**Frontend:** HTML5, CSS3, JS (ES6+), Canvas API. GitHub Pages from `/docs`.

---

## 📂 Project Structure

```text
nexus-neural-search/
├── backend/
│   ├── main.py        # Thin FastAPI routes
│   ├── engine.py      # Retrieval core: embeddings, hybrid search, rerank, recommend, RAG
│   ├── auth.py        # JWT auth (SECRET_KEY from env)
│   ├── database.py    # SQLite connection
│   └── models.py      # User + WishlistItem (media_id is a uuid string)
├── docs/              # Frontend (GitHub Pages root)
├── ingest.py          # Single unified ingester (dense + BM25 -> Qdrant)
├── expand_dataset.py  # Pull more titles (movies/TV) from TMDB into dataset.csv
├── clean_dataset.py   # Blank duplicated/corrupt descriptions (+ --push to re-embed)
├── enrich_data.py     # Optional TMDB/Jikan metadata enrichment
├── debug.py           # Health check for the collection + a test search
├── dataset.csv        # Source data (~21k items)
├── requirements.txt
├── .env.example       # Copy to .env and fill in
└── README.md
```

---

## 🚀 Setup

### 1. Install
```bash
git clone https://github.com/aarushch/Nexus-Neural-Search.git
cd Nexus-Neural-Search
python -m venv .venv && .venv\Scripts\activate      # (Windows)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Create a Qdrant Cloud cluster
1. Sign in at [cloud.qdrant.io](https://cloud.qdrant.io) and create a **free cluster**.
2. Copy the **cluster URL** and an **API key**.
3. `cp .env.example .env` and fill in `QDRANT_URL`, `QDRANT_API_KEY`, and a `SECRET_KEY`
   (`python -c "import secrets; print(secrets.token_hex(32))"`). Optionally add `OPENROUTER_API_KEY`.

### 3. Ingest the data
```bash
python ingest.py            # creates the collection + uploads all items
python debug.py             # verify: prints point count + a test search
```

### 4. Run the API
```bash
uvicorn backend.main:app --reload --port 10000
```
The frontend (`docs/script.js`) points at the deployed API URL — change `API_URL` there for local testing.

---

## 🔌 API

| Method | Route | Notes |
|---|---|---|
| `POST` | `/recommend` | `{text, top_k, model}` — `model` = `internal` (hybrid) or `api` (grounded RAG) |
| `POST` | `/recommend/personalized` | Auth required; blends wishlist recommendations |
| `POST` | `/similar` | `{id}` — Qdrant recommend from one item |
| `POST` | `/login` · `/signup` | JWT auth |
| `GET/POST/DELETE` | `/wishlist...` | Auth required |

---

## ⚙️ Notes on hosting

Query-time embedding + reranking now run in-process, so the host needs enough RAM for
`torch` + bge-small + the cross-encoder (~600 MB resident). On a very small instance set
`ENABLE_RERANK=false` (dense+BM25 only) to cut memory. See `.env.example` for all toggles.
