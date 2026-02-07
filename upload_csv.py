import os
import pandas as pd
import requests
import time
import math
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# --- CONFIGURATION ---
CSV_FILE = "dataset.csv" 
COLLECTION_NAME = "freeme_collection"
# Using the working BAAI model
HF_API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH_SIZE = 50

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
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# 1. Load CSV & Inspect Columns
if not os.path.exists(CSV_FILE):
    print(f"❌ Error: Could not find {CSV_FILE}")
    exit()

print(f"📖 Reading {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)
print(f"🔍 YOUR CSV COLUMNS: {list(df.columns)}") # <--- READ THIS IN THE LOGS IF IT FAILS!

# 2. Reset Collection (Clean Slate to remove "Ghost Data")
try:
    client.delete_collection(COLLECTION_NAME)
    print("🗑️  Wiped old collection (Goodbye Ghost Data).")
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print("✅ Created fresh collection.")

# --- SMART COLUMN MAPPING ---
def get_column_value(row, possible_names, default=""):
    for name in possible_names:
        # Case-insensitive check
        for col in row.index:
            if col.lower() == name.lower():
                val = row[col]
                if pd.isna(val) or val == "":
                    return default
                return val
    return default

print(f"📊 Processing {len(df)} rows...")

total_uploaded = 0
points_batch = []

for index, row in df.iterrows():
    # Try multiple names for each field
    title = get_column_value(row, ['title', 'original_title', 'Series_Title', 'Name'], "Unknown")
    desc = get_column_value(row, ['overview', 'description', 'summary', 'plot', 'Synopsis'], "")
    rating = get_column_value(row, ['vote_average', 'rating', 'IMDB_Rating', 'Score'], 0)
    image = get_column_value(row, ['poster_path', 'poster', 'image', 'Poster_Link'], "")
    media_type = get_column_value(row, ['media_type', 'type', 'Genre'], "MOVIE")

    # --- 🧹 DATA CLEANING (The Filter) ---
    
    # 1. Fix Image URL
    image = str(image).strip()
    if image.startswith("/"):
        image = f"https://image.tmdb.org/t/p/w500{image}"
    
    # 2. Skip JUNK Data (Fixes "No data" issue)
    if not desc or len(str(desc)) < 20 or str(desc).lower() == "no data.":
        print(f"   ⚠️ Skipped (No Description): {title}")
        continue

    # 3. Skip Missing Images (Fixes black cards)
    if not image or len(image) < 5 or "nan" in image.lower():
        print(f"   ⚠️ Skipped (No Image): {title}")
        continue

    # 4. Construct Search Text
    text = f"{title} {desc}"
    
    # --- GET EMBEDDING ---
    vector = None
    for attempt in range(3):
        try:
            response = requests.post(
                HF_API_URL, 
                headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
                json={"inputs": [text]} 
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    vector = data[0] if isinstance(data[0], list) else data
                break
            elif response.status_code == 503:
                time.sleep(2)
            else:
                break
        except:
            break
    
    if vector and len(vector) == VECTOR_SIZE:
        payload = {
            "title": title,
            "description": str(desc)[:500] + "...", # Truncate long descriptions
            "rating": float(rating) if rating else 0,
            "type": str(media_type).split(",")[0].upper(), # Clean genre/type
            "image": image
        }
        points_batch.append(PointStruct(id=index+1, vector=vector, payload=payload))
        print(f"   ✅ Prepared: {title}")
    else:
        print(f"   ⚠️ Skipped (Embedding Failed): {title}")

    # Upload Batch
    if len(points_batch) >= BATCH_SIZE:
        client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
        total_uploaded += len(points_batch)
        print(f"🚀 Uploaded batch! Total: {total_uploaded}")
        points_batch = [] 
        time.sleep(0.5)

# Upload remaining
if points_batch:
    client.upsert(collection_name=COLLECTION_NAME, points=points_batch)
    total_uploaded += len(points_batch)

print(f"🎉 FINAL SUCCESS! Uploaded {total_uploaded} clean movies.")