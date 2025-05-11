from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import normalize

class Retriever:
    def __init__(self, chunks, model_name='all-MiniLM-L6-v2'):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)

        # Embed chunks
        self.embeddings = normalize(self.model.encode(chunks, convert_to_numpy=True))
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])  # IP = inner product
        self.index.add(self.embeddings)

    def retrieve(self, query, top_k=3):
        query_vec = normalize(self.model.encode([query], convert_to_numpy=True))
        D, I = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in I[0]]