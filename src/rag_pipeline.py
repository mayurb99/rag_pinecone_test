import os
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from huggingface_hub import InferenceClient

def get_rag_answer(query):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("rag-index")

    query_embedding = model.encode([query])[0]

    results = index.query(
        vector=query_embedding.tolist(),
        top_k=2,
        include_metadata=True
    )

    context = "\n".join([m["metadata"]["text"] for m in results["matches"]])

    prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{query}
"""

    llm = InferenceClient(
        model="allenai/Olmo-3-7B-Instruct:publicai",
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    response = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content