from src.ingestion.pdf_load import PDFLoader

loader = PDFLoader()

# ✅ อ่านไฟล์เป็น bytes ก่อน
with open("data/samples/l1.pdf", "rb") as f:
    pdf_bytes = f.read()

pages = loader.load_from_bytes(pdf_bytes)

print(len(pages), "หน้า")

for p in pages:
    print(f"\n===== PAGE {p.page_number} =====")
    print(p.text)
