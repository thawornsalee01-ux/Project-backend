from src.service.rag_upload import UploadRAG_service

# โหลด PDF ตัวอย่าง
pdf_path = "data/samples/13.pdf"

with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

# เรียกใช้งาน service
UploadRAG_service(pdf_bytes, pdf_name="กฏหมาย.pdf")
