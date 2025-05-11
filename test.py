import pickle
from sklearn.preprocessing import normalize

def debug_retrieval(retriever, query, top_k=5):
    print(f"\n🔍 Query: {query}\n{'='*60}")
    
    # Embed query and search manually
    query_vec = normalize(retriever.model.encode([query], convert_to_numpy=True))
    D, I = retriever.index.search(query_vec, top_k)
    
    for rank, (dist, idx) in enumerate(zip(D[0], I[0])):
        chunk = retriever.chunks[idx]
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        
        print(f"\n--- Result {rank+1} ---")
        print(f"Distance: {dist:.4f}")
        print(f"Source: {source}, Page: {page}")
        print(f"Text: {chunk['text'][:300].strip()}...")

with open("corpus/chunk_corpus.pkl", "rb") as f:
    chunk_corpus = pickle.load(f)

from retriever.Retriever import Retriever

retriever = Retriever(chunk_corpus, model_name='all-mpnet-base-v2')

debug_retrieval(retriever, "how does diffraction grating work with white light as opposed to a laser?")

