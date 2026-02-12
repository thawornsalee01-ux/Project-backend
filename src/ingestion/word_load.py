from dataclasses import dataclass
from typing import List
from io import BytesIO
from docx import Document


@dataclass
class PageText:
    page_number: int
    text: str


class WordLoader:
    """
    โหลดไฟล์ Word (.docx)
    - ตรวจจับ Page Break จริง (w:br w:type="page")
    - แปลงข้อความเป็น Markdown-like
    """

    def load_from_bytes(self, docx_bytes: bytes) -> List[PageText]:
        try:
            doc = Document(BytesIO(docx_bytes))
        except Exception as e:
            raise RuntimeError(f"ไม่สามารถเปิดไฟล์ Word จาก bytes ได้ ({e})")

        pages: List[PageText] = []
        current_lines: List[str] = []
        page_number = 1

        for p in doc.paragraphs:
            # ตรวจจับ Page Break ภายใน paragraph
            runs = p._p.xpath(".//w:br")
            has_page_break = any(r.get("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}type") == "page"
                                 for r in runs)

            text = p.text.strip()
            if text:
                style = p.style.name if p.style else "Normal"

                # 🔹 Heading → Markdown #
                if style.startswith("Heading"):
                    level = style.replace("Heading", "").strip()
                    if level.isdigit():
                        markdown_line = f"{'#' * int(level)} {text}"
                    else:
                        markdown_line = f"# {text}"

                # 🔹 List → Markdown -
                elif p._p.pPr is not None and p._p.pPr.numPr is not None:
                    markdown_line = f"- {text}"

                # 🔹 Normal paragraph
                else:
                    markdown_line = text

                current_lines.append(markdown_line)

            # 🔹 ถ้ามี page break → สร้างหน้าใหม่
            if has_page_break:
                if current_lines:
                    pages.append(
                        PageText(
                            page_number=page_number,
                            text="\n".join(current_lines).strip(),
                        )
                    )
                    page_number += 1
                    current_lines = []

        # 🔹 บันทึกหน้าสุดท้าย
        if current_lines:
            pages.append(
                PageText(
                    page_number=page_number,
                    text="\n".join(current_lines).strip(),
                )
            )

        return pages
