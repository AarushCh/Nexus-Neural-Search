"""
Nexus API.

Routes are thin; all retrieval logic lives in backend/engine.py.

Search modes (frontend sends `model`):
  * "internal" -> hybrid vector search (dense + BM25 + cross-encoder rerank)
  * "api"      -> grounded RAG: same hybrid retrieval, then Nemotron reranks/explains
                  REAL results (no hallucinated titles).
"""

import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth import get_current_user_db, hash_password, login_user
from backend.database import Base, SessionLocal, engine
from backend.engine import (
    COLLECTION_NAME,
    get_qdrant,
    ground_with_llm,
    hybrid_search,
    recommend,
    similar_items,
)
from backend.models import User, WishlistItem

app = FastAPI(title="Nexus Neural Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Schemas ------------------------------------------------------------------

class SearchRequest(BaseModel):
    text: str
    top_k: int = 12
    model: str = "internal"


class AuthRequest(BaseModel):
    username: str
    email: str
    password: str


class SimilarRequest(BaseModel):
    id: str


# --- Health / Auth ------------------------------------------------------------

@app.get("/")
def health_check():
    return {"status": "online", "engine": "hybrid+rerank"}


@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    return login_user(form, db)


@app.post("/signup")
def signup(data: AuthRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username taken")
    user = User(username=data.username, email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    return {"status": "created"}


# --- Search -------------------------------------------------------------------

def _search(text: str, top_k: int, model: str) -> list[dict]:
    # Always retrieve real hits from the DB first.
    hits = hybrid_search(text, top_k=top_k if model != "api" else max(top_k, 20))
    if model == "api":
        # Grounded RAG: Nemotron only reorders/explains these real hits.
        return ground_with_llm(text, hits, top_k=top_k)
    return hits[:top_k]


@app.post("/recommend")
def recommend_route(req: SearchRequest):
    return _search(req.text, req.top_k, req.model)


@app.post("/recommend/personalized")
def personalized(req: SearchRequest, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    """Hybrid search, then blend in items similar to the user's wishlist."""
    base = _search(req.text, req.top_k, req.model)

    wishlist_ids = [i.media_id for i in db.query(WishlistItem).filter_by(user_id=user.id).all()]
    if not wishlist_ids or req.model == "api":
        return base

    # Real personalization: recommend from wishlist, then interleave with the query results.
    liked = recommend(positive_ids=wishlist_ids, top_k=req.top_k, exclude_ids=set(wishlist_ids))
    return _interleave(base, liked, req.top_k)


def _interleave(primary: list[dict], secondary: list[dict], limit: int) -> list[dict]:
    out, seen = [], set()
    for a, b in zip(primary, secondary + [None] * len(primary)):
        for item in (a, b):
            if item and str(item.get("id")) not in seen:
                seen.add(str(item["id"]))
                out.append(item)
    for item in primary + secondary:
        if str(item.get("id")) not in seen:
            seen.add(str(item["id"]))
            out.append(item)
    return out[:limit]


@app.post("/similar")
def similar(req: SimilarRequest):
    return similar_items(req.id, top_k=12)


# --- Wishlist -----------------------------------------------------------------

@app.post("/wishlist/add/{mid}")
def add_wishlist(mid: str, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    if mid.startswith("ai-"):
        raise HTTPException(status_code=400, detail="Cannot save AI items.")
    if not db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).first():
        db.add(WishlistItem(user_id=u.id, media_id=mid))
        db.commit()
    return {"status": "ok"}


@app.delete("/wishlist/remove/{mid}")
def remove_wishlist(mid: str, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).delete()
    db.commit()
    return {"status": "ok"}


@app.get("/wishlist")
def get_wishlist(u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    ids = [i.media_id for i in db.query(WishlistItem).filter_by(user_id=u.id).all()]
    if not ids:
        return []
    try:
        points = get_qdrant().retrieve(COLLECTION_NAME, ids=ids, with_payload=True)
    except Exception:
        return []
    results = []
    for p in points:
        item = dict(p.payload or {})
        item["id"] = p.id
        results.append(item)
    return results


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)
