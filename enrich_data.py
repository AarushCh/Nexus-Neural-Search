import pandas as pd
import requests
import time
import os
from dotenv import load_dotenv

# ==========================================
#  CONFIGURATION
# ==========================================
load_dotenv()  # Load keys from .env

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
INPUT_FILE = "dataset.csv"
OUTPUT_FILE = "dataset_enriched.csv"

if not TMDB_API_KEY:
    print("❌ ERROR: TMDB_API_KEY not found in .env file.")
    exit()

# ==========================================
#  API HELPERS
# ==========================================
def fetch_anime_jikan(title):
    """Fetches Anime data from MyAnimeList via Jikan (Free, No Key)"""
    try:
        url = f"https://api.jikan.moe/v4/anime?q={title}&limit=1"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json().get('data', [])
            if data:
                item = data[0]
                return {
                    'image': item['images']['jpg']['large_image_url'],
                    'rating': item.get('score', 0),
                    'description': item.get('synopsis', '').replace('\n', ' ')
                }
        time.sleep(1.2) # Jikan has strict rate limits
    except Exception as e:
        pass
    return None

def fetch_movie_tmdb(title, type_hint="movie"):
    """Fetches Movie/TV data from TMDB (Credible Source)"""
    try:
        search_type = "tv" if "tv" in str(type_hint).lower() else "movie"
        url = f"https://api.themoviedb.org/3/search/{search_type}?api_key={TMDB_API_KEY}&query={title}"
        res = requests.get(url, timeout=5)
        
        if res.status_code == 200:
            results = res.json().get('results', [])
            if results:
                item = results[0]
                poster = item.get('poster_path')
                img_url = f"https://image.tmdb.org/t/p/w500{poster}" if poster else None
                desc = item.get('overview', '')
                rating = item.get('vote_average', 0)
                
                return {'image': img_url, 'rating': rating, 'description': desc}
    except Exception:
        pass
    return None

# ==========================================
#  MAIN LOOP
# ==========================================
print(f"🚀 Starting Deep Scan for {INPUT_FILE}...")
try:
    df = pd.read_csv(INPUT_FILE)
except:
    print("❌ dataset.csv not found!")
    exit()

df = df.fillna("")
fixed_count = 0

for index, row in df.iterrows():
    needs_fix = False
    
    # 1. BAD IMAGE
    curr_img = str(row['image']).lower()
    if not curr_img or "placeholder" in curr_img or "n/a" in curr_img or "nan" in curr_img:
        needs_fix = True
        
    # 2. BAD DESCRIPTION
    curr_desc = str(row['description']).strip().lower()
    if not curr_desc or len(curr_desc) < 20 or "no description" in curr_desc:
        needs_fix = True
        
    # 3. BAD RATING
    try:
        curr_rate = float(row['rating'])
    except:
        curr_rate = 0.0
    if curr_rate <= 0.1:
        needs_fix = True

    if needs_fix:
        title = row['title']
        m_type = row['type']
        
        # Decide which DB to check
        is_anime = "anime" in str(m_type).lower() or "anime" in str(row.get('genre', '')).lower()
        print(f"[{index}/{len(df)}] Scanning: {title[:30]}...", end="\r")
        
        new_data = None
        if is_anime:
            new_data = fetch_anime_jikan(title)
        else:
            new_data = fetch_movie_tmdb(title, m_type)
            
        if new_data:
            if new_data['image']: df.at[index, 'image'] = new_data['image']
            if new_data['description']: df.at[index, 'description'] = new_data['description']
            if new_data['rating']: df.at[index, 'rating'] = new_data['rating']
            fixed_count += 1

    # Autosave every 50 items
    if index > 0 and index % 50 == 0:
        df.to_csv(OUTPUT_FILE, index=False)

# Final Save
df.to_csv(OUTPUT_FILE, index=False)
print(f"\n✅ SCAN COMPLETE! Fixed {fixed_count} items.")
print(f"💾 Saved to '{OUTPUT_FILE}'.")
print("👉 INSTRUCTIONS: Delete 'dataset.csv', rename 'dataset_enriched.csv' to 'dataset.csv', and run 'ingest.py' again.")