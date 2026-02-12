from src.ingestion.word_load import WordLoader

loader = WordLoader()

with open("data/samples/l1.docx", "rb") as f:
    docx_bytes = f.read()

pages = loader.load_from_bytes(docx_bytes)

print(len(pages), "หน้า")

for p in pages:
    print(f"\n===== PAGE {p.page_number} =====")
    print(p.text)  # พิมพ์แค่ 500 ตัวแรกเพื่อดู preview
