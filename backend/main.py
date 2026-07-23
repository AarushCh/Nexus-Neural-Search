"""
Nexus API.

Routes are thin; all retrieval logic lives in backend/engine.py.

Search modes (frontend sends `model`):
  * "internal" -> hybrid vector search (dense + BM25 + cross-encoder rerank)
  * "api"      -> grounded RAG: same hybrid retrieval, then Nemotron reranks/explains
                  REAL results (no hallucinated titles).

User system: JWT auth, wishlist, server-side search history, interactions
(view/like/dismiss), history-driven "For You" recommendations, a home feed,
and a TMDB-enriched title detail endpoint.
"""

import json
import os
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth import get_current_user_db, hash_password, login_user
from backend.database import Base, SessionLocal, engine
from backend.engine import (
    build_filter,
    enrich_detail,
    for_you,
    get_by_ids,
    ground_with_llm,
    hybrid_search,
    similar_items,
    top_popular,
)
from backend.models import Interaction, MediaDetail, SearchHistory, User, WishlistItem

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
    category: Optional[str] = None
    min_rating: Optional[float] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None


class AuthRequest(BaseModel):
    username: str
    email: str
    password: str


class SimilarRequest(BaseModel):
    id: str


class InteractionRequest(BaseModel):
    media_id: str
    kind: str = "view"  # view | like | dismiss


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
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(username=data.username, email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    return {"status": "created"}


@app.get("/me")
def me(user=Depends(get_current_user_db)):
    return {"username": user.username, "email": user.email}


# --- Search -------------------------------------------------------------------

def _search(req: SearchRequest) -> list[dict]:
    qfilter = build_filter(req.category, req.min_rating, req.year_min, req.year_max)
    hits = hybrid_search(
        req.text,
        # Give the LLM a wider pool in api mode so mainstream titles are in reach.
        top_k=req.top_k if req.model != "api" else max(req.top_k, 40),
        qfilter=qfilter,
    )
    if req.model == "api":
        return _patch_posters(ground_with_llm(req.text, hits, top_k=req.top_k))
    return _patch_posters(hits[:req.top_k])


@app.post("/recommend")
def recommend_route(req: SearchRequest):
    return _search(req)


@app.post("/recommend/personalized")
def personalized(req: SearchRequest, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    """Log the search for history, then return PURE query relevance.

    Explicit search is never personalized — favouriting horror must not bleed
    into a 'cyberpunk anime' query. Taste-based recommendations live only in the
    For You feed (`/feed`, `/recommendations/foryou`), not here."""
    db.add(SearchHistory(user_id=user.id, query=req.text, model=req.model))
    db.commit()
    return _search(req)


@app.post("/similar")
def similar(req: SimilarRequest):
    return similar_items(req.id, top_k=12)


# --- Feed / recommendations ---------------------------------------------------

def _profile_ids(user_id: int, db: Session) -> list[str]:
    """Positive-signal media ids: wishlist + likes + recent views (most recent first)."""
    wl = [w.media_id for w in db.query(WishlistItem).filter_by(user_id=user_id)
          .order_by(WishlistItem.added_at.desc()).all()]
    inter = db.query(Interaction).filter_by(user_id=user_id).filter(
        Interaction.kind.in_(["like", "view"])).order_by(Interaction.created_at.desc()).limit(60).all()
    ids = wl + [i.media_id for i in inter]
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _dismissed_ids(user_id: int, db: Session) -> list[str]:
    return [i.media_id for i in db.query(Interaction).filter_by(user_id=user_id, kind="dismiss").all()]


@app.get("/recommendations/foryou")
def recommendations_foryou(top_k: int = 20, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    profile = _profile_ids(user.id, db)
    if not profile:
        return []
    return for_you(profile, negative_ids=_dismissed_ids(user.id, db),
                   top_k=top_k, exclude_ids=set(profile))


@app.get("/feed")
def feed(user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    """Personalized home feed: For You, Because you liked X, Recently Viewed + Top Rated rows."""
    rows = []
    profile = _profile_ids(user.id, db)

    # Your Favourites — the user's saved wishlist, most recently added first.
    wl_ids = [w.media_id for w in db.query(WishlistItem).filter_by(user_id=user.id)
              .order_by(WishlistItem.added_at.desc()).limit(40).all()]
    favs = get_by_ids(wl_ids)
    if favs:
        rows.append({"title": "Your Favourites", "items": _patch_posters(_dedupe(favs))})

    if profile:
        fy = for_you(profile, negative_ids=_dismissed_ids(user.id, db), top_k=20, exclude_ids=set(profile))
        if fy:
            rows.append({"title": "For You", "items": fy})

        anchor = get_by_ids([profile[0]])
        if anchor:
            sim = similar_items(profile[0], top_k=20)
            if sim:
                rows.append({"title": f"Because you liked {anchor[0].get('title','')}", "items": sim})

    views = db.query(Interaction).filter_by(user_id=user.id, kind="view").order_by(
        Interaction.created_at.desc()).limit(40).all()
    recent = get_by_ids([v.media_id for v in views])
    if recent:
        rows.append({"title": "Recently Viewed", "items": _patch_posters(_dedupe(recent))})

    rows += _popular_rows()
    return rows


@app.get("/discover")
def discover():
    """Anonymous landing feed (no auth) = most-voted (mainstream) rows per category."""
    return _popular_rows()


def _patch_posters(items: list[dict]) -> list[dict]:
    """Override stored (often wrong/missing) images with corrected TMDB posters
    from the MediaDetail cache. Fixes Recently Viewed / feed / search posters
    (e.g. 'Another Life' no longer shows the Re:Zero art) once a title is enriched."""
    if not items:
        return items
    ids = [str(it.get("id")) for it in items]
    db = SessionLocal()
    try:
        rows = db.query(MediaDetail.media_id, MediaDetail.poster).filter(
            MediaDetail.media_id.in_(ids), MediaDetail.poster.isnot(None)).all()
    finally:
        db.close()
    posters = {mid: p for mid, p in rows if p}
    for it in items:
        p = posters.get(str(it.get("id")))
        if p:
            it["image"] = p
    return items


def _dedupe(items: list[dict]) -> list[dict]:
    """Drop repeated ids and repeated posters (bad TMDB data reuses posters)."""
    out, seen_id, seen_img = [], set(), set()
    for it in items:
        iid = str(it.get("id"))
        img = it.get("image") or ""
        if iid in seen_id or (img and img in seen_img):
            continue
        seen_id.add(iid)
        if img:
            seen_img.add(img)
        out.append(it)
    return out


def _popular_rows() -> list[dict]:
    rows = []
    for cat, label in [("MOVIE", "Top Movies"), ("TV", "Top TV"),
                       ("ANIME", "Top Anime"), ("DOCUMENTARY", "Top Documentaries")]:
        items = _patch_posters(_dedupe(top_popular(top_k=24, qfilter=build_filter(category=cat))))
        if items:
            rows.append({"title": label, "items": items[:20]})
    return rows


# --- Interactions / history ---------------------------------------------------

@app.post("/interactions")
def add_interaction(req: InteractionRequest, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    if req.kind not in ("view", "like", "dismiss"):
        raise HTTPException(status_code=400, detail="Invalid kind")
    if req.media_id.startswith("ai-"):
        return {"status": "skipped"}
    db.add(Interaction(user_id=user.id, media_id=req.media_id, kind=req.kind))
    db.commit()
    return {"status": "ok"}


@app.get("/history/search")
def get_search_history(limit: int = 50, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    rows = db.query(SearchHistory).filter_by(user_id=user.id).order_by(
        SearchHistory.created_at.desc()).limit(limit).all()
    return [{"query": r.query, "model": r.model, "at": r.created_at.isoformat()} for r in rows]


@app.delete("/history/search")
def clear_search_history(user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    db.query(SearchHistory).filter_by(user_id=user.id).delete()
    db.commit()
    return {"status": "cleared"}


@app.get("/history/views")
def get_view_history(limit: int = 30, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    views = db.query(Interaction).filter_by(user_id=user.id, kind="view").order_by(
        Interaction.created_at.desc()).limit(limit).all()
    return get_by_ids([v.media_id for v in views])


# --- Title detail (TMDB enriched, cached) -------------------------------------

@app.get("/title/{media_id}")
def title_detail(media_id: str, db: Session = Depends(get_db)):
    cards = get_by_ids([media_id])
    if not cards:
        raise HTTPException(status_code=404, detail="Not found")
    card = cards[0]

    # Fast path: build_catalogue.py already embedded validated enrichment in the
    # Qdrant payload (trailer/cast/providers/backdrop). Serve it with no TMDB call.
    if card.get("trailer_key") or card.get("cast") or card.get("providers"):
        return card

    cached = db.query(MediaDetail).filter_by(media_id=media_id).first()
    if cached:
        enrichment = {
            "tmdb_id": cached.tmdb_id,
            "poster": cached.poster,
            "trailer_key": cached.trailer_key,
            "cast": json.loads(cached.cast_json) if cached.cast_json else [],
            "providers": json.loads(cached.providers_json) if cached.providers_json else [],
            "backdrop": cached.backdrop,
            "runtime": cached.runtime,
        }
    else:
        enrichment = enrich_detail(media_id, card)
        db.merge(MediaDetail(
            media_id=media_id,
            tmdb_id=enrichment.get("tmdb_id"),
            poster=enrichment.get("poster"),
            trailer_key=enrichment.get("trailer_key"),
            cast_json=json.dumps(enrichment.get("cast", [])),
            providers_json=json.dumps(enrichment.get("providers", [])),
            backdrop=enrichment.get("backdrop"),
            runtime=enrichment.get("runtime"),
            fetched_at=datetime.utcnow(),
        ))
        db.commit()

    # Prefer the corrected TMDB poster (fixes wrong/low-quality stored images).
    if enrichment.get("poster"):
        card["image"] = enrichment["poster"]
    card.update(enrichment)
    return card


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
    ids = [i.media_id for i in db.query(WishlistItem).filter_by(user_id=u.id)
           .order_by(WishlistItem.added_at.desc()).all()]
    return get_by_ids(ids)


# --- helpers ------------------------------------------------------------------

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 10000)), reload=True)
