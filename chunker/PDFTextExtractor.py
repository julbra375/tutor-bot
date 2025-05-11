import pdfplumber

class PDFTextExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf = pdfplumber.open(pdf_path)
        self.total_pages = len(self.pdf.pages)

    def extract_page(self, page_num: int) -> str:
        if 0 <= page_num < self.total_pages:
            return self.pdf.pages[page_num].extract_text()
        else:
            raise IndexError("Page number out of range")

    def extract_range(self, start: int, end: int) -> str:
        if start < 0 or end > self.total_pages or start >= end:
            raise ValueError("Invalid page range")
        return "\n".join(self.pdf.pages[i].extract_text() for i in range(start, end))

    def extract_all(self) -> str:
        return "\n".join(page.extract_text() for page in self.pdf.pages)
    
    def get_num_pages(self):
        return self.total_pages

    def close(self):
        self.pdf.close()
