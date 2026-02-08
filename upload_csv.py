import os
import pandas as pd
import requests
import time
import uuid
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# --- CONFIGURATION ---
CSV_FILE = "dataset.csv"
COLLECTION_NAME = "freeme_collection"
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH_SIZE = 20  

# Keep this FALSE so you resume where you left off
RESET_COLLECTION = False 

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not QDRANT_URL or not QDRANT_API_KEY or not HF_TOKEN:
    print("❌ Error: Missing credentials in .env file!")
    exit()

if QDRANT_URL.startswith("ttps://"): 
    QDRANT_URL = QDRANT_URL.replace("ttps://", "https://")

print(f"☁️ Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# 1. Handle Collection Reset
if RESET_COLLECTION:
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  WIPED old collection.")
    except:
        pass
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print("✅ Created FRESH collection.")
else:
    print("ℹ️  Resume Mode: Checking for existing items...")

# 2. Load CSV
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: Could not find {CSV_FILE}")
    exit()

print(f"📖 Reading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

def generate_id(title):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, title.lower().strip()))

def get_column_value(row, possible_names, default=""):
    for name in possible_names:
        for col in row.index:
            if col.lower() == name.lower():
                val = row[col]
                if pd.isna(val) or val == "":
                    return default
                return val
    return default

print(f"📊 Found {len(df)} rows. Processing...")

total_uploaded = 0
points_batch = []
skipped_count = 0

for index, row in df.iterrows():
    title = get_column_value(row, ['title', 'original_title', 'Series_Title', 'Name'], "Unknown")
    desc = get_column_value(row, ['overview', 'description', 'summary', 'plot'], "")
    rating = get_column_value(row, ['vote_average', 'rating', 'IMDB_Rating', 'Score'], 0)
    image = get_column_value(row, ['poster_path', 'poster', 'image', 'Poster_Link'], "")
    media_type = get_column_value(row, ['media_type', 'type', 'Genre'], "MOVIE")

    point_id = generate_id(str(title))

    # Check existence
    if not RESET_COLLECTION:
        try:
            # Quick check if ID exists
            existing = client.retrieve(collection_name=COLLECTION_NAME, ids=[point_id])
            if existing:
                print(f"   ⏭️  Skipping existing: {title}")
                skipped_count += 1
                continue
        except:
            pass

    # Clean data
    image = str(image).strip()
    if image.startswith("/"): image = f"https://image.tmdb.org/t/p/w500{image}"
    if not desc or len(str(desc)) < 20 or str(desc).lower() == "no data.": continue 
    if not image or len(image) < 5 or "nan" in image.lower(): continue 

    text = f"{title} {desc}"
    vector = None
    
    # 🛡️ ROBUST RETRY LOOP (5 Attempts)
    for attempt in range(5):
        try:
            response = requests.post(
                HF_API_URL, 
                headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
                json={"inputs": [text]},
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    vector = data[0] if isinstance(data[0], list) else data
                break # Success!
            
            # If server is busy or rate limited, WAIT.
            elif response.status_code in [503, 429, 504]:
                wait_time = (attempt + 1) * 5  # Wait 5s, then 10s, then 15s...
                print(f"   ⏳ API Busy ({response.status_code}). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"   ❌ API Error {response.status_code}: {response.text}")
                break
        except Exception as e:
            print(f"   ❌ Connection Error. Retrying...")
            time.sleep(5)
    
    if vector and len(vector) == VECTOR_SIZE:
        payload = {
            "title": title,
            "description": str(desc)[:500] + "...",
            "rating": float(rating) if rating else 0,
            "type": str(media_type).split(",")[0].upper(),
            "image": image
        }
        points_batch.append(PointStruct(id=point_id, vector=vector, payload=payload))
        print(f"   ✅ Prepared: {title}")
    else:
        print(f"   ⚠️ PERMANENT FAIL: {title}")

    # 🛑 COOL DOWN: Sleep 0.5s after every movie to be polite
    time.sleep(0.5)

    if len(points_batch) >= BATCH_SIZE:
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
            total_uploaded += len(points_batch)
            print(f"🚀 Uploaded batch! Total New: {total_uploaded}")
            points_batch = []
            time.sleep(1) # Extra rest after upload
        except Exception as e:
            print(f"❌ Batch Upload Failed: {e}")
            time.sleep(10)

if points_batch:
    client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
    total_uploaded += len(points_batch)

print(f"🎉 DONE! Uploaded: {total_uploaded} | Skipped: {skipped_count}")