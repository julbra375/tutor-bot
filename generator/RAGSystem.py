class RAGSystem:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question, top_k=3):
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        context = "\n\n".join(f"[Source: {chunk['metadata']['source']}, Page {chunk['metadata']['page']}]\n{chunk['text']}" for chunk in retrieved)
        return self.generator.generate(question, context)
