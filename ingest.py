import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# --- CONFIGURATION (CLOUD VS LOCAL) ---
load_dotenv() # Load keys from .env

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if QDRANT_URL and QDRANT_API_KEY:
    print(f"☁️ CONNECTING TO QDRANT CLOUD: {QDRANT_URL}")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    print("📁 USING LOCAL STORAGE (qdrant_storage)")
    # Local Mode: Saves to folder so we don't need Docker
    client = QdrantClient(path="qdrant_storage") 

model = SentenceTransformer("all-MiniLM-L6-v2")
COLLECTION_NAME = "freeme_collection"

# --- HELPERS ---
def find_col(df, candidates):
    for col in candidates:
        if col in df.columns: return col
    return None

# --- MAIN ---
if not os.path.exists("dataset.csv"):
    print("❌ ERROR: dataset.csv missing.")
    exit()

print("📊 Reading dataset...")
try:
    df = pd.read_csv("dataset.csv")
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    col_map = {}
    
    # Map standard columns
    t_col = find_col(df, ['title', 'show_title', 'name', 'series_title'])
    if t_col: col_map[t_col] = 'title'
    
    d_col = find_col(df, ['description', 'synopsis', 'plot', 'summary', 'desc'])
    if d_col: col_map[d_col] = 'description'
    
    i_col = find_col(df, ['image', 'img_url', 'poster', 'cover', 'poster_link', 'picture'])
    if i_col: col_map[i_col] = 'image'
    
    r_col = find_col(df, ['rating', 'score', 'imdb_score', 'vote_average'])
    if r_col: col_map[r_col] = 'rating'
    
    y_col = find_col(df, ['year', 'release_year', 'date', 'aired'])
    if y_col: col_map[y_col] = 'year'

    # Important: Capture Genre to detect Docs/Anime
    g_col = find_col(df, ['genre', 'listed_in', 'category', 'genres'])
    if g_col: col_map[g_col] = 'genre'

    ty_col = find_col(df, ['type', 'media_type', 'content_type'])
    if ty_col: col_map[ty_col] = 'type'

    df = df.rename(columns=col_map)
    
    # Fill missing
    for std_col in ['title', 'description', 'image', 'rating', 'year', 'type', 'genre']:
        if std_col not in df.columns: df[std_col] = "N/A"

    df = df.fillna("") 
    print(f"✅ Columns mapped. Processing {len(df)} items...")

except Exception as e:
    print(f"❌ CSV Error: {e}")
    exit()

print(f"⚙️ Resetting DB '{COLLECTION_NAME}'...")
try:
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
except Exception as e:
    # Some cloud instances throw error if checking existence differently, safe to ignore
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)

print("🚀 Embedding Data...")

points = []
BATCH_SIZE = 100 

for idx, row in df.iterrows():
    # [SMART LOGIC] Force correct types based on Genre text
    real_type = str(row['type']).title()
    genre_text = str(row['genre']).lower()
    
    if "documentary" in genre_text or "doc" in genre_text:
        real_type = "Documentary"
    elif "anime" in genre_text:
        real_type = "Anime"
    elif "stand-up" in genre_text:
        real_type = "Stand-Up"

    # Rich Text Embedding
    search_text = f"{row['title']} {row['description']} {row['genre']} {real_type}"
    vector = model.encode(search_text).tolist()
    
    # Update payload
    payload = row.to_dict()
    payload['type'] = real_type # Save the corrected type
    
    points.append(models.PointStruct(id=idx, vector=vector, payload=payload))

    if len(points) >= BATCH_SIZE:
        client.upload_points(collection_name=COLLECTION_NAME, points=points)
        points = []
        if idx % 1000 == 0: print(f"   Uploaded {idx} / {len(df)}")

if points:
    client.upload_points(collection_name=COLLECTION_NAME, points=points)

print(f"✅ DONE! All items ingested into {('CLOUD' if QDRANT_URL else 'LOCAL')}.")