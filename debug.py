"""
Quick sanity check for the Nexus hybrid engine.

Run after creating a new Qdrant cluster + ingesting:
    python debug.py
"""

from backend.engine import COLLECTION_NAME, get_qdrant, hybrid_search

print("---------------------------------------------")
print(f"🔍 Checking collection: {COLLECTION_NAME}")
print("---------------------------------------------")

client = get_qdrant()

if not client.collection_exists(COLLECTION_NAME):
    print("❌ Collection missing. Run: python ingest.py")
    raise SystemExit(1)

count = client.count(COLLECTION_NAME).count
print(f"📊 Points stored: {count}")
if count == 0:
    print("⚠️  Empty collection. Run: python ingest.py")
    raise SystemExit(1)

print("\n🧪 Hybrid test search: 'cyberpunk anime about identity'")
for i, hit in enumerate(hybrid_search("cyberpunk anime about identity", top_k=5), 1):
    print(f"   {i}. [{hit.get('score')}%] {hit.get('title')}  ({hit.get('type')})")

print("\n✅ Engine is live.")
