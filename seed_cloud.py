import os
import json
import requests
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

if not QDRANT_URL or not QDRANT_API_KEY or not HF_TOKEN:
    print("❌ Error: Missing credentials in .env file!")
    exit()

if QDRANT_URL.startswith("ttps://"): 
    QDRANT_URL = QDRANT_URL.replace("ttps://", "https://")

print(f"☁️ Connecting to: {QDRANT_URL}...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "freeme_collection"
VECTOR_SIZE = 384 

try:
    client.delete_collection(COLLECTION_NAME)
    print("🗑️  Deleted old collection.")
except:
    pass

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
)
print("Cb  Created fresh collection.")

movies = [
    {"title": "Inception", "description": "A thief who steals corporate secrets through the use of dream-sharing technology.", "type": "MOVIE", "rating": 8.8, "image": "https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg"},
    {"title": "Interstellar", "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "type": "MOVIE", "rating": 8.6, "image": "https://image.tmdb.org/t/p/w500/gEU2QniL6C8z1dY4rer3387P38t.jpg"},
    {"title": "The Dark Knight", "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham.", "type": "MOVIE", "rating": 9.0, "image": "https://image.tmdb.org/t/p/w500/qJ2tW6WMUDux911r6m7haRef0WH.jpg"},
    {"title": "Spirited Away", "description": "A young girl, Chihiro, becomes trapped in a strange new world of spirits.", "type": "ANIME", "rating": 8.6, "image": "https://image.tmdb.org/t/p/w500/39wmItIWsg5sZMyRUKGnSxQbUgZ.jpg"},
    {"title": "Breaking Bad", "description": "A high school chemistry teacher diagnosed with inoperable lung cancer turns to manufacturing and selling methamphetamine.", "type": "TV", "rating": 9.5, "image": "https://image.tmdb.org/t/p/w500/ggFHVNu6YYI5L9pCfOacjizRGt.jpg"},
    {"title": "Attack on Titan", "description": "After his hometown is destroyed and his mother is killed, young Eren Jaeger vows to cleanse the earth of the giant humanoid Titans.", "type": "ANIME", "rating": 9.0, "image": "https://image.tmdb.org/t/p/w500/h8jXav3piw08BZN3RMLy96W9k6t.jpg"},
    {"title": "Planet Earth", "description": "Emmy Award-winning, 11-part series from the makers of The Blue Planet documents the incredible beauty and power of nature.", "type": "DOC", "rating": 9.4, "image": "https://image.tmdb.org/t/p/w500/d95ZlK1F8l7L7Wq6Vk6W5e5Z5.jpg"},
    {"title": "The Office", "description": "A mockumentary on a group of typical office workers, where the workday consists of ego clashes, inappropriate behavior, and tedium.", "type": "TV", "rating": 8.9, "image": "https://image.tmdb.org/t/p/w500/qWnJzyZhyy74gJPo09Bq05d2.jpg"},
    {"title": "Your Name", "description": "Two strangers find themselves linked in a bizarre way. When a connection forms, will distance be the only thing to keep them apart?", "type": "ANIME", "rating": 8.4, "image": "https://image.tmdb.org/t/p/w500/q719jXXEzOoYaps6babgKnONONX.jpg"},
    {"title": "Avengers: Endgame", "description": "After the devastating events of Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more.", "type": "MOVIE", "rating": 8.4, "image": "https://image.tmdb.org/t/p/w500/or06FN3Dka5tukK1e9sl16pB3iy.jpg"}
]

print(f"🚀 Uploading {len(movies)} movies...")

api_url = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"

points = []
for i, movie in enumerate(movies):
    text = f"{movie['title']} {movie['description']}"
    
    # Retry logic
    vector = None
    for attempt in range(3):
        try:
            # 🚨 FIX: Wrap text in a LIST [text]
            response = requests.post(
                api_url, 
                headers={"Authorization": f"Bearer {HF_TOKEN}"}, 
                json={"inputs": [text]} 
            )
            if response.status_code == 200:
                data = response.json()
                # 🚨 FIX: The result is now [[0.1, 0.2...]], so we take index [0]
                if isinstance(data, list):
                    vector = data[0] if isinstance(data[0], list) else data
                break
            elif response.status_code == 503:
                print("   ⏳ Model loading... waiting 3s")
                time.sleep(3)
            else:
                print(f"   ❌ Error {response.status_code}: {response.text}")
                break
        except Exception as e:
            print(f"   ❌ Connection Error: {e}")
            break

    if vector:
        points.append(PointStruct(id=i+1, vector=vector, payload=movie))
        print(f"   ✅ Processed: {movie['title']}")
    else:
        print(f"   ⚠️ SKIPPED: {movie['title']} (Embedding Failed)")

if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"🎉 Success! {len(points)} movies uploaded to the Cloud.")
else:
    print("❌ No movies were uploaded.")