import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# Load docs
with open("docs/sample_document.txt") as f:
    text = f.read()

# Chunking
chunks = []
chunk_size = 100
overlap = 20

start = 0
while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start = end - overlap

# Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("rag-index")

# Upload
vectors = []
for i, chunk in enumerate(chunks):
    vectors.append((
        str(i),
        embeddings[i].tolist(),
        {"text": chunk}
    ))

index.upsert(vectors)

print("✅ Data ingested to Pinecone")