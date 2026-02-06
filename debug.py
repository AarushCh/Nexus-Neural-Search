from qdrant_client import QdrantClient

# Connect to Qdrant
client = QdrantClient(path="freeme_qdrant_db")
COLLECTION_NAME = "freeme_collection"

print("---------------------------------------------")
print(f"🔍 Checking Neural Database: {COLLECTION_NAME}")
print("---------------------------------------------")

try:
    # 1. Check if Collection Exists
    if not client.collection_exists(COLLECTION_NAME):
        print(f"❌ ERROR: Collection '{COLLECTION_NAME}' does not exist!")
        print("👉 FIX: You must run 'python ingest.py' again.")
        exit()

    # 2. Check Item Count
    info = client.get_collection(COLLECTION_NAME)
    count = info.points_count
    print(f"📊 Total Memories Found: {count}")

    if count == 0:
        print("⚠️  WARNING: Collection exists but is EMPTY.")
        print("👉 FIX: Run 'python ingest.py' to load the data.")
    else:
        print("✅ SUCCESS: The Brain is loaded and ready.")
        
        # 3. Test a Search manually to verify connections
        from sentence_transformers import SentenceTransformer
        print("\n🧪 Running Test Search for 'Action Movie'...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        vector = model.encode("Action Movie").tolist()
        
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vector,
            limit=1
        )
        
        if results:
            print(f"🎉 Search Works! Found: {results[0].payload['title']}")
        else:
            print("❌ Search returned 0 results (This shouldn't happen if count > 0).")

except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not connect to Qdrant.")
    print(f"Details: {e}")
    print("👉 Ensure Docker is running: 'docker ps'")

print("---------------------------------------------")