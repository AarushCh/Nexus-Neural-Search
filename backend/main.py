from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import Base, engine, SessionLocal
from backend.models import User, WishlistItem
from backend.auth import get_current_user_db, login_user, hash_password
import requests
import json
import re
import os
import time
from dotenv import load_dotenv
from qdrant_client import models

# --- CONFIGURATION ---
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Model: BAAI/bge-small-en-v1.5
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "arcee-ai/trinity-large-preview:free")

app = FastAPI(title="FreeMe Engine (Hybrid Fill)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

# --- 🧠 CORE AI FUNCTIONS ---

def get_embedding(text):
    if not HF_TOKEN:
        print("❌ CRITICAL: HF_TOKEN missing.")
        return None

    # Input must be a list [text]
    payload = {"inputs": [text], "options": {"wait_for_model": True}}
    
    for attempt in range(3):
        try:
            response = requests.post(
                HF_API_URL,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json=payload,
                timeout=15 
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                        return data[0]
                    return data
            
            if response.status_code == 503:
                time.sleep(3)
                continue
            break
        except:
            break
    return None

def get_qdrant():
    from qdrant_client import QdrantClient
    url = QDRANT_URL
    if url and url.startswith("ttps://"): url = url.replace("ttps://", "https://")
    return QdrantClient(url=url, api_key=QDRANT_API_KEY)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def keyword_search(query, limit=1):
    """Finds exact movie by title text"""
    try:
        client = get_qdrant()
        results = client.scroll(
            collection_name="freeme_collection",
            scroll_filter=models.Filter(
                should=[
                    models.FieldCondition(key="title", match=models.MatchText(text=query)),
                ]
            ),
            limit=limit
        )[0]
        return results
    except:
        return []

def safe_vector_search(vector, limit=50):
    try: 
        q_client = get_qdrant()
        return q_client.query_points(collection_name="freeme_collection", query=vector, limit=limit).points
    except:
        return []

def get_llm_recommendations(query):
    final_results = []
    seen_ids = set()

    print(f"🧠 TRINITY: Thinking about '{query}'...") # DEBUG LOG

    # --- STEP 1: TRINITY THINKING ---
    try:
        if not OPENROUTER_API_KEY:
            print("❌ TRINITY ERROR: No API Key found in Environment!")
        else:
            # We use a simple prompt to ensure JSON format
            resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
                    "HTTP-Referer": "http://nexus-search.com",
                    "Content-Type": "application/json"
                },
                data=json.dumps({
                    "model": OPENROUTER_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a movie expert. Return ONLY a raw JSON list of 10 movie titles. No text, no markdown. Example: [\"Matrix\", \"Inception\"]"},
                        {"role": "user", "content": f"Recommend 10 movies strictly matching '{query}'."}
                    ]
                }), 
                timeout=10
            )
            
            if resp.status_code == 200:
                content = resp.json()['choices'][0]['message']['content']
                print(f"🤖 TRINITY SAID: {content}") # DEBUG LOG

                # Clean up the response (sometimes AI adds ```json ... ```)
                clean_content = re.sub(r'```json|```', '', content).strip()
                match = re.search(r'\[.*\]', clean_content, re.DOTALL)
                
                if match:
                    titles = json.loads(match.group())
                    print(f"🎯 PARSED TITLES: {titles}") # DEBUG LOG
                    
                    for t in titles:
                        # Search for the exact movie in our DB
                        hits = keyword_search(t, limit=1)
                        if hits:
                            h = hits[0]
                            if h.id not in seen_ids:
                                item = h.payload
                                item["id"] = h.id
                                item["score"] = 99 # 🟢 FORCE 99% FOR TRINITY
                                final_results.append(item)
                                seen_ids.add(h.id)
                else:
                    print("⚠️ TRINITY ERROR: Could not find JSON list in response.")
            else:
                print(f"❌ TRINITY API ERROR: {resp.status_code} - {resp.text}")

    except Exception as e:
        print(f"❌ TRINITY CRASH: {e}")

    # --- STEP 2: THE BACKFILL (Ensure 12 Tiles) ---
    slots_needed = 12 - len(final_results)
    if slots_needed > 0:
        print(f"⚠️ Trinity found {len(final_results)}. Backfilling {slots_needed} from Vector DB.")
        vector = get_embedding(query)
        if vector:
            hits = safe_vector_search(vector, limit=slots_needed + 5)
            for h in hits:
                if h.id not in seen_ids:
                    item = h.payload
                    item["id"] = h.id
                    item["score"] = int(h.score * 100) if h.score else 65
                    final_results.append(item)
                    seen_ids.add(h.id)
                    if len(final_results) >= 12: break

    return final_results

# --- ROUTES ---

class UserRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class PersonalizedRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class AuthRequest(BaseModel): username: str; email: str; password: str
class SimilarRequest(BaseModel): id: str # Changed to str to support UUIDs

@app.get("/")
def health_check():
    return {"status": "online", "message": "Nexus Hybrid Engine v7.0"}

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
def recommend(req: UserRequest):
    # If model is API (Trinity), use the smart hybrid function
    if req.model == 'api': 
        return get_llm_recommendations(req.text)
    
    # Standard Vector Search
    vector = get_embedding(req.text)
    if not vector: 
        # Emergency keyword fallback if embedding fails completely
        hits = keyword_search(req.text, limit=12)
        results = []
        for h in hits:
            item = h.payload
            item["id"] = h.id
            item["score"] = 80
            results.append(item)
        return results

    hits = safe_vector_search(vector, limit=req.top_k)
    results = []
    for h in hits:
        item = h.payload
        item["id"] = h.id
        item["score"] = int(h.score * 100) if h.score else 0 
        results.append(item)
        
    return results

@app.post("/recommend/personalized")
def personalized(req: PersonalizedRequest, user=Depends(get_current_user_db), db: Session = Depends(get_db)):
    # Personalization is complex with Trinity, so we default to standard Hybrid for now
    if req.model == 'api': 
        return get_llm_recommendations(req.text)
    
    return recommend(UserRequest(text=req.text, top_k=req.top_k, model='internal'))

@app.post("/similar")
def similar(req: SimilarRequest):
    q_client = get_qdrant()
    # Retrieve the vector of the clicked movie
    tgt = q_client.retrieve("freeme_collection", ids=[req.id], with_vectors=True)
    if not tgt: return []
    
    # Search for nearest neighbors
    hits = safe_vector_search(tgt[0].vector, limit=13) # Get 13, remove self
    
    results = []
    for h in hits:
        # Don't show the movie itself in recommendations
        if str(h.id) != str(req.id):
            item = h.payload
            item["id"] = h.id
            item["score"] = 95
            results.append(item)
            
    return results[:12]

@app.post("/wishlist/add/{mid}")
def add_w(mid: str, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    if not db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).first():
        db.add(WishlistItem(user_id=u.id, media_id=mid))
        db.commit()
    return {"status": "ok"}

@app.delete("/wishlist/remove/{mid}")
def rem_w(mid: str, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    db.query(WishlistItem).filter_by(user_id=u.id, media_id=mid).delete()
    db.commit()
    return {"status": "ok"}

@app.get("/wishlist")
def get_w(u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    q_client = get_qdrant()
    ids = [i.media_id for i in db.query(WishlistItem).filter_by(user_id=u.id).all()]
    if not ids: return []
    try: 
        points = q_client.retrieve("freeme_collection", ids=ids)
        results = []
        for p in points:
            item = p.payload
            item["id"] = p.id
            results.append(item)
        return results
    except: return []

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=10000, reload=True)