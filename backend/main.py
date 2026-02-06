from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import Base, engine, SessionLocal
from backend.models import User, WishlistItem
from backend.auth import get_current_user_db, login_user, hash_password
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
import uvicorn
import math
import requests
import json
import re
import os
from dotenv import load_dotenv

# --- SECURITY: LOAD ENV VARIABLES ---
load_dotenv()  # This loads variables from your .env file

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Fallback model if env var is missing, though you should put this in .env too if it changes
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")

if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY not found in .env file!")

app = FastAPI(title="FreeMe Engine v3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

print("🦅 FreeMe Neural Core Starting...")
model = SentenceTransformer("all-MiniLM-L6-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Ensure qdrant_storage is in .gitignore
client = QdrantClient(path="qdrant_storage", force_disable_check_same_thread=True)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def normalize_score(logit):
    return int((1 / (1 + math.exp(-logit))) * 100)

def get_safe_rating(payload):
    try: val = payload.get('rating', 0); return float(val) if str(val).lower() not in ['nan', 'n/a', ''] else 5.0
    except: return 5.0

def safe_vector_search(vector, limit=50):
    try: return client.query_points(collection_name="freeme_collection", query=vector, limit=limit).points
    except: return []

def get_llm_recommendations(query):
    try:
        if not OPENROUTER_API_KEY: return None
        
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "HTTP-Referer": "http://localhost"},
            data=json.dumps({
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": f"Recommend 10 movies strictly matching '{query}'. Return ONLY JSON list of strings."}]
            }), timeout=15
        )
        if resp.status_code != 200: return None
        content = resp.json()['choices'][0]['message']['content']
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if not match: return None
        titles = json.loads(match.group())
        results = []
        for t in titles:
            hits = safe_vector_search(model.encode(str(t)).tolist(), limit=1)
            if hits:
                item = hits[0].payload
                item["id"] = hits[0].id
                item["score"] = 99
                results.append(item)
        return results
    except: return None

class UserRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class PersonalizedRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class AuthRequest(BaseModel): username: str; email: str; password: str
class SimilarRequest(BaseModel): id: int

@app.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    return login_user(form, db)

@app.post("/signup")
def signup(data: AuthRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    
    user = User(username=data.username, email=data.email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    return {"status": "created"}

@app.post("/recommend")
async def recommend(req: UserRequest):
    if req.model == 'api': return get_llm_recommendations(req.text) or []
    hits = safe_vector_search(model.encode(req.text).tolist(), limit=50)
    if not hits: return []
    pairs = [[req.text, f"{h.payload.get('title','')} {h.payload.get('description','')}"] for h in hits]
    scores = cross_encoder.predict(pairs)
    results = []
    for i, h in enumerate(hits):
        item = h.payload
        item["id"] = h.id
        rel = float(scores[i])
        rat = get_safe_rating(item)
        if rat >= 8.0: rel += 2.0
        elif rat < 5.0 and rat > 0: rel -= 2.0
        item["score"] = normalize_score(rel)
        results.append(item)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:req.top_k]

@app.post("/recommend/personalized")
async def personalized(req: PersonalizedRequest, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    if req.model == 'api': return await recommend(UserRequest(text=req.text, top_k=req.top_k, model='api'))
    w_items = db.query(WishlistItem).filter_by(user_id=user.id).all()
    if not w_items: return await recommend(UserRequest(text=req.text, top_k=req.top_k))
    try:
        points = client.retrieve("freeme_collection", ids=[i.media_id for i in w_items], with_vectors=True)
        vectors = [p.vector for p in points if p.vector]
        if not vectors: raise Exception
        avg_taste = [sum(col)/len(col) for col in zip(*vectors)]
        mood = model.encode(req.text).tolist()
        final_vector = [0.2*t + 0.8*m for t, m in zip(avg_taste, mood)]
    except: final_vector = model.encode(req.text).tolist()
    hits = safe_vector_search(final_vector, limit=50)
    pairs = [[req.text, f"{h.payload.get('title','')} {h.payload.get('description','')}"] for h in hits]
    scores = cross_encoder.predict(pairs)
    results = []
    for i, h in enumerate(hits):
        item = h.payload
        item["id"] = h.id
        item["score"] = normalize_score(float(scores[i]))
        results.append(item)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:req.top_k]

@app.post("/similar")
def similar(req: SimilarRequest):
    tgt = client.retrieve("freeme_collection", ids=[req.id], with_vectors=True)
    if not tgt: return []
    hits = safe_vector_search(tgt[0].vector, limit=15)
    return [{**h.payload, 'id': h.id, 'score': 95} for h in hits if h.id!=req.id][:4]

@app.post("/wishlist/add/{mid}")
def add_w(mid: int, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    if not db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).first():
        db.add(WishlistItem(user_id=u.id, media_id=mid))
        db.commit()
    return {"status": "ok"}

@app.delete("/wishlist/remove/{mid}")
def rem_w(mid: int, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).delete()
    db.commit()
    return {"status": "ok"}

@app.get("/wishlist")
def get_w(u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    ids = [i.media_id for i in db.query(WishlistItem).filter_by(user_id=u.id).all()]
    if not ids: return []
    try: return [{**p.payload, 'id': p.id} for p in client.retrieve("freeme_collection", ids=ids)]
    except: return []

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)