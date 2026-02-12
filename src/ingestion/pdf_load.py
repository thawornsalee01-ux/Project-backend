# src/ingestion/pdf_loader.py

import fitz  # PyMuPDF
from dataclasses import dataclass
from typing import List


@dataclass
class PageText:
    page_number: int
    text: str


class PDFLoader:
    """
    โหลดไฟล์ PDF แล้วดึงข้อความออกมาเป็นรายหน้า
    (จัดบรรทัดใหม่ให้เหมาะกับเอกสารราชการ / TOR)
    """

    def load_from_bytes(self, pdf_bytes: bytes) -> List[PageText]:
        from io import BytesIO

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as e:
            raise RuntimeError(f"ไม่สามารถเปิดไฟล์ PDF จาก bytes ได้ ({e})")

        pages: List[PageText] = []

        for i in range(len(doc)):
            page = doc.load_page(i)

            blocks = page.get_text("blocks")
            lines: List[str] = []

            # blocks: (x0, y0, x1, y1, text, block_no, block_type)
            for b in blocks:
                text = b[4]
                if not text:
                    continue

                # แยกเป็นบรรทัดย่อย
                for line in text.split("\n"):
                    clean = line.strip()
                    if clean:
                        lines.append(clean)

            page_text = "\n".join(lines)

            pages.append(
                PageText(
                    page_number=i + 1,
                    text=page_text,
                )
            )

        doc.close()
        return pages
