# 🦅 FreeMe Neural Search (Nexus Engine)
## 🔗 Live Website

➡️ [https://aarushch.github.io/Nexus-Neural-Search/](https://aarushch.github.io/Nexus-Neural-Search/) 


![Project Banner](https://placehold.co/1200x400/050505/00f3ff?text=NEXUS+INTELLIGENCE+ENGINE)

> **A next-generation, multimodal AI search engine that understands "vibes" and semantic context rather than just keywords.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-9cf?style=for-the-badge)](https://qdrant.tech/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

## 📖 Overview

**FreeMe Neural Search** (codenamed *Nexus*) is a local AI-powered recommendation engine designed to break free from rigid, keyword-based search algorithms. Instead of matching exact titles, it uses **vector embeddings** and **neural networks** to understand the *meaning* behind a query.

You can ask for *"movies that feel like a rainy Tuesday in Tokyo"* or *"cyberpunk anime with philosophical themes about identity"*, and the engine will "think" about your request to find the closest semantic matches.

## ✨ Key Features

### 🧠 **Core Intelligence**
* **Semantic Vector Search:** Powered by the `all-MiniLM-L6-v2` transformer model, converting text into 384-dimensional vectors.
* **Hybrid Re-Ranking:** Uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to double-check and re-score vector results for maximum accuracy.
* **LLM Integration:** Optional connection to **Trinity (Thinking)** via OpenRouter for complex reasoning queries.

### 💻 **The Nexus Interface (Frontend)**
* **Reactive Neural Network:** A custom-built HTML5 Canvas background that visualizes neural connections, reacting dynamically to cursor movement with a "synapse" effect.
* **Dynamic Theming:** Seamless toggle between **Cyber-Dark Mode** (Neon/Glassmorphism) and **Clean-Light Mode**.
* **Zero-Framework:** Built with pure Vanilla JS and CSS3 for maximum performance and zero bloat.

### ⚙️ **System Capabilities**
* **Secure Authentication:** Full JWT-based user login and registration system with Bcrypt password hashing.
* **Personalized Wishlist:** Save movies/anime to your profile. The system learns from your wishlist to adjust future recommendations.
* **Data Enrichment:** Automated scripts to fetch high-quality metadata (posters, ratings) from **TMDB** and **Jikan (MyAnimeList)** APIs.

---

## 🛠️ Tech Stack

### **Backend (Python)**
* **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (High-performance async API)
* **Server:** Uvicorn
* **Vector Database:** [Qdrant](https://qdrant.tech/) (Local file-based instance)
* **Relational Database:** SQLite (via SQLAlchemy)
* **ML Libraries:** `sentence-transformers`, `numpy`, `torch` (cpu)

### **Frontend (Web)**
* **Core:** HTML5, CSS3, JavaScript (ES6+)
* **Hosting:** GitHub Pages compatible (served from `/docs`)
* **Visuals:** Custom Canvas API animations

---

## 📂 Project Structure

```text
freeme-neural-search/
├── backend/                 # FastAPI Application Source
│   ├── main.py              # API Entry Point & Routes
│   ├── auth.py              # JWT Authentication Logic
│   ├── database.py          # Database Connection (SQLite)
│   └── models.py            # SQLAlchemy Database Models
│
├── docs/                    # Frontend UI (GitHub Pages Root)
│   ├── index.html           # Main Interface
│   ├── script.js            # UI Logic & Animation Engine
│   └── style.css            # Cyberpunk/Light Theme Styling
│
├── qdrant_storage/          # Local Vector Database Files (GitIgnored)
├── freeme.db                # User/Auth Database (GitIgnored)
│
├── ingest.py                # Script to vectorise data -> Qdrant
├── enrich_data.py           # Script to fetch metadata from APIs
├── requirements.txt         # Python Dependencies
├── .env                     # API Keys & Secrets (GitIgnored)
└── README.md                # Documentation
```
---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/nexus-neural-search.git](https://github.com/YOUR_USERNAME/nexus-neural-search.git)

cd nexus-neural-search
```
