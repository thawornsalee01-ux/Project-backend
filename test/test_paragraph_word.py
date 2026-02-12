from src.ingestion.word_load import WordLoader
from src.ingestion.paragraph_word import ParagraphSplitter

# ---------------------------
# 1) โหลดไฟล์ Word เป็น bytes
# ---------------------------
loader = WordLoader()

with open("data/samples/l1.docx", "rb") as f:
    docx_bytes = f.read()

pages = loader.load_from_bytes(docx_bytes)

print("===== RAW WORD BLOCKS =====")
print("จำนวน block ทั้งหมด:", len(pages))

for p in pages:
    print("\n----------------------------")
    print("PAGE:", p.page_number)
    print("STYLE:", p.style)
    print("IS_HEADING:", p.is_heading)
    print("IS_LIST:", p.is_list)
    print("TEXT:")
    print(p.text)

# ---------------------------
# 2) Split เป็น Paragraph จริง
# ---------------------------
splitter = ParagraphSplitter()
paragraphs = splitter.split(pages)

print("\n\n===== AFTER PARAGRAPH SPLIT =====")
print("จำนวน paragraph หลัง split:", len(paragraphs))

for para in paragraphs:
    print("\n==============================")
    print(f"PARAGRAPH #{para.index}")
    print("PAGE:", para.page_number)
    print("TEXT:")
    print(para.text)
