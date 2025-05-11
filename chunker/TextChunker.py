from nltk.tokenize import sent_tokenize

class TextChunker:
    def __init__(self, max_tokens=200, overlap=10):
        self.max_tokens = max_tokens
        self.overlap = overlap

    def chunk(self, text):
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current_len + sentence_len > self.max_tokens:
                chunks.append(" ".join(current_chunk))
                if self.overlap > 0:
                    current_chunk = current_chunk[-self.overlap:].copy()
                else:
                    current_chunk = []
                current_len = sum(len(s.split()) for s in current_chunk)

            current_chunk.append(sentence)
            current_len += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
