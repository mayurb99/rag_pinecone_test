import os
import hashlib
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# ----------------------------
# CONFIG
# ----------------------------
DOCS_FOLDER = "docs"
CHUNK_SIZE = 100
OVERLAP = 20

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def generate_id(text):
    return hashlib.md5(text.encode()).hexdigest()

def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - OVERLAP

    return chunks

# ----------------------------
# MAIN INGESTION
# ----------------------------

# Init model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index")

# Process all files in docs/
for filename in os.listdir(DOCS_FOLDER):

    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(DOCS_FOLDER, filename)

    print(f"\n📄 Processing file: {filename}")

    # Load file
    with open(file_path) as f:
        text = f.read()

    # Step 1: DELETE old chunks for this file
    
    try:
        print("⚡ Deleting old data from Pinecone...")
        index.delete(filter={"source": filename})
    except Exception as e:
        print("⚠️ No existing data to delete (first run)")
        
    # Step 2: Chunk
    chunks = chunk_text(text)
    print(f"🔹 Created {len(chunks)} chunks")

    # Step 3: Embed
    embeddings = model.encode(chunks)

    # Step 4: Prepare vectors
    vectors = []

    for chunk, embedding in zip(chunks, embeddings):
        vector_id = generate_id(chunk)

        vectors.append((
            vector_id,
            embedding.tolist(),
            {
                "text": chunk,
                "source": filename
            }
        ))

    # Step 5: Upsert
    index.upsert(vectors)

    print(f"✅ Ingested {len(vectors)} chunks for {filename}")

print("\n🎉 Ingestion complete!")