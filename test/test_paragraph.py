from src.ingestion.pdf_load import PDFLoader
from src.ingestion.paragraph import ParagraphSplitter

loader = PDFLoader()
splitter = ParagraphSplitter()

# ✅ อ่านไฟล์เป็น bytes ก่อน
with open("data/samples/l8.pdf", "rb") as f:
    pdf_bytes = f.read()

pages = loader.load_from_bytes(pdf_bytes)
paragraphs = splitter.split(pages)

for p in paragraphs:
    print(f"\n[Page {p.page_number} | #{p.index}]")
    print(p.text)
