from chunker.TextChunker import TextChunker
from chunker.PDFTextExtractor import PDFTextExtractor
from retriever.Retriever import Retriever
from generator.Generator import OpenAIGenerator
from generator.RAGSystem import RAGSystem
from dotenv import load_dotenv
import os
import pickle
import nltk
nltk.data.path.insert(0, "nltk_data")

# Load API key
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]

# Load chunked corpus
with open("corpus/chunk_corpus.pkl", "rb") as f:
    chunks = pickle.load(f)

# Instantiate retriever and generator
retriever = Retriever(chunks, model_name='all-mpnet-base-v2')
generator = OpenAIGenerator(api_key=api_key, model='gpt-3.5-turbo')

# Create RAG system and answer question
rag_system = RAGSystem(retriever, generator)
question = "how does diffraction grating work with white light as opposed to a laser?"
answer = rag_system.answer(question)
print(answer)