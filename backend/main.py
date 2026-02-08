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
import uuid  # <--- CRITICAL: Needed for God Mode fake IDs
from dotenv import load_dotenv
from qdrant_client import models

# --- CONFIGURATION ---
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Model: BAAI/bge-small-en-v1.5 (For Vector Search Fallback)
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
# Model: Mistral 7B (For God Mode Generation)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")

app = FastAPI(title="Nexus God Mode Engine")

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
    if not HF_TOKEN: return None
    payload = {"inputs": [text], "options": {"wait_for_model": True}}
    for attempt in range(3):
        try:
            response = requests.post(
                HF_API_URL, headers={"Authorization": f"Bearer {HF_TOKEN}"}, json=payload, timeout=8
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0] if isinstance(data[0], list) else data
            if response.status_code == 503:
                time.sleep(2)
                continue
            break
        except:
            continue
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

def safe_vector_search(vector, limit=50):
    try: 
        q_client = get_qdrant()
        return q_client.query_points(collection_name="freeme_collection", query=vector, limit=limit).points
    except: return []

# --- 🧠 GOD MODE GENERATOR (THE NEW LOGIC) ---
def get_llm_recommendations(query):
    print(f"🧠 TRINITY GOD MODE: Generating fresh data for '{query}'...") 

    try:
        if not OPENROUTER_API_KEY:
            print("❌ ERROR: No API Key.")
            return []

        # 1. Ask Trinity to be the Database (Generates Title, Desc, Rating)
        prompt = f"""
        You are a movie database API. 
        User Request: "{query}"
        
        Generate 12 unique recommendations.
        Return strictly a JSON array of objects. 
        Each object must have:
        - "title": (String) Exact Title
        - "description": (String) 1 sentence plot summary.
        - "rating": (Float) IMDB style rating (e.g. 8.5)
        - "type": (String) One of: MOVIE, TV, ANIME, DOCUMENTARY
        
        Do NOT include markdown formatting (like ```json). Just the raw JSON array.
        """
        
        resp = requests.post(
            "[https://openrouter.ai/api/v1/chat/completions](https://openrouter.ai/api/v1/chat/completions)",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
                "HTTP-Referer": "[http://nexus-search.com](http://nexus-search.com)",
                "Content-Type": "application/json"
            },
            data=json.dumps({
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}]
            }), 
            timeout=25 # Increased timeout for long generation
        )
        
        if resp.status_code == 200:
            content = resp.json()['choices'][0]['message']['content']
            # Clean possible markdown
            clean_content = re.sub(r'```json|```', '', content).strip()
            
            # Find the JSON array
            match = re.search(r'\[.*\]', clean_content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                results = []
                
                for item in data:
                    # ✨ MAGIC: Create a working image link using AI Art (Pollinations)
                    safe_title = re.sub(r'[^a-zA-Z0-9 ]', '', item['title'])
                    image_url = f"[https://image.pollinations.ai/prompt/movie](https://image.pollinations.ai/prompt/movie) poster for {safe_title} minimalist 4k?width=400&height=600&nologo=true"
                    
                    results.append({
                        "id": f"ai-{uuid.uuid4()}", # Fake ID so React doesn't crash
                        "title": item.get('title', 'Unknown'),
                        "description": item.get('description', 'AI Generated.'),
                        "rating": item.get('rating', 0),
                        "type": item.get('type', 'MOVIE').upper(),
                        "image": image_url, # The AI Image
                        "score": 99 # It's a perfect match because AI said so
                    })
                
                print(f"✨ Generated {len(results)} AI tiles.")
                return results

    except Exception as e:
        print(f"❌ TRINITY CRASH: {e}")

    return [] # Fallback will trigger if this returns empty

# --- ROUTES ---

class UserRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class PersonalizedRequest(BaseModel): text: str; top_k: int = 12; model: str = "internal"
class AuthRequest(BaseModel): username: str; email: str; password: str
class SimilarRequest(BaseModel): id: str

@app.get("/")
def health_check(): return {"status": "online", "mode": "GOD_MODE"}

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

@app.post("/recommend")
def recommend(req: UserRequest):
    # 1. If user wants AI (Trinity), use God Mode
    if req.model == 'api':
        results = get_llm_recommendations(req.text)
        if results: return results
        # If AI fails, fall through to vector search
    
    # 2. Standard Vector Search (Fallback or Internal Mode)
    vector = get_embedding(req.text)
    if not vector: return []
    hits = safe_vector_search(vector, limit=req.top_k)
    results = []
    for h in hits:
        item = h.payload
        item["id"] = h.id
        item["score"] = int(h.score * 100) if h.score else 0 
        results.append(item)
    return results

@app.post("/recommend/personalized")
def personalized(req: PersonalizedRequest, user=Depends(get_current_user_db)):
    # Redirect to standard recommend logic for God Mode
    return recommend(UserRequest(text=req.text, top_k=req.top_k, model=req.model))

@app.post("/similar")
def similar(req: SimilarRequest):
    q_client = get_qdrant()
    
    # 🕵️‍♂️ DETECT GHOST ID (AI Generated)
    if str(req.id).startswith("ai-"):
        # We can't look up an AI ID in the database.
        # Simple fix: Return empty list so UI doesn't crash, 
        # or you could trigger another AI generation here if you wanted.
        return [] 

    # Normal Database ID Logic
    try:
        tgt = q_client.retrieve("freeme_collection", ids=[req.id], with_vectors=True)
        if not tgt: return []
        hits = safe_vector_search(tgt[0].vector, limit=13)
        results = []
        for h in hits:
            if str(h.id) != str(req.id):
                item = h.payload
                item["id"] = h.id
                item["score"] = 95
                results.append(item)
        return results[:12]
    except:
        return []

@app.post("/wishlist/add/{mid}")
def add_w(mid: str, u=Depends(get_current_user_db), db: Session = Depends(get_db)):
    # AI IDs can't be saved to DB because they aren't real records in Qdrant
    if mid.startswith("ai-"):
        raise HTTPException(status_code=400, detail="Cannot save AI-generated dreams yet.")
        
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