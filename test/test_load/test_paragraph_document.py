from src.ingestion.document_load import DocumentLoader
from src.ingestion.paragraph import ParagraphSplitter

loader = DocumentLoader()
splitter = ParagraphSplitter()

# ✅ อ่านไฟล์เป็น bytes ก่อน
with open("data/samples/l1.docx", "rb") as f:
    pdf_bytes = f.read()

pages = loader.load_from_bytes(pdf_bytes)
paragraphs = splitter.split(pages)

for p in paragraphs:
    print(f"\n[Page {p.page_number} | #{p.index}]")
    print(p.text)
