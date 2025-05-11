from chunker.PDFTextExtractor import PDFTextExtractor
from chunker.TextChunker import TextChunker

def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    return f"{source.replace(' ', '')}_p{page:03d}_c{chunk_index:02d}"

topics = ["Electricity", "ODU", "ODU", "PW", "PW"]

chunk_data = []
for i, source in enumerate(["corpus/electricity.pdf", "corpus/ODU_1.pdf", "corpus/ODU_2.pdf", 
                            "corpus/PW_1.pdf", "corpus/PW_2.pdf"]):
    extractor = PDFTextExtractor(source)
    total_pages = extractor.get_num_pages()
    chunker = TextChunker(max_tokens=200, overlap=10)

    for page_num in range(total_pages):
        text = extractor.extract_page(page_num)

        chunks = chunker.chunk(text)
        
        for j, chunk in enumerate(chunks):
            chunk_id = make_chunk_id(source, page_num, j)
            chunk_data.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "metadata": {
                    "source": source,
                    "page": page_num,
                    "topic": topics[i],
                    "chunk_index": j
                }
            })

import pickle
with open("chunk_corpus.pkl", "wb") as f:
    pickle.dump(chunk_data, f)
