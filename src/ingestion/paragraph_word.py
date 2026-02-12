import re
from dataclasses import dataclass
from typing import List, Optional
from src.ingestion.word_load import PageText


@dataclass
class Paragraph:
    page_number: int
    index: int
    text: str
    embedding: Optional[List[float]] = None


class ParagraphSplitter:
    """
    Hybrid Paragraph Splitter สำหรับ Word
    - ใช้โครงสร้าง Word ถ้ามี (Heading / List)
    - fallback ด้วย regex (เลขไทย/อารบิก) ถ้าไม่มี
    """

    THAI = "๐๑๒๓๔๕๖๗๘๙"

    # 1. / ๑. / 1)
    MAIN_NUMBER = re.compile(
        rf"^\s*([{THAI}0-9]+)[\.\)]\s+"
    )

    # 1.1 / ๑.๑ → ไม่ split
    SUB_NUMBER = re.compile(
        rf"^\s*([{THAI}0-9]+)\.([{THAI}0-9]+)"
    )

    def split(self, pages: List[PageText]) -> List[Paragraph]:
        paragraphs: List[Paragraph] = []
        buffer: List[str] = []

        index = 0
        current_page: Optional[int] = None

        def flush():
            nonlocal index, current_page
            if buffer:
                paragraphs.append(
                    Paragraph(
                        page_number=current_page or 0,
                        index=index,
                        text="\n".join(buffer),
                    )
                )
                buffer.clear()
                index += 1
                current_page = None

        for p in pages:
            text = (p.text or "").strip()
            if not text:
                continue

            # ==============================
            # 🥇 ใช้โครงสร้าง Word ก่อน
            # ==============================
            if p.is_heading:
                flush()
                current_page = p.page_number
                buffer.append(text)
                continue

            if p.is_list:
                flush()
                paragraphs.append(
                    Paragraph(
                        page_number=p.page_number,
                        index=index,
                        text=text,
                    )
                )
                index += 1
                continue

            # ==============================
            # 🥉 fallback: regex เลขข้อ
            # ==============================
            is_main = self.MAIN_NUMBER.match(text)
            is_sub = self.SUB_NUMBER.match(text)

            if is_main and not is_sub:
                flush()
                current_page = p.page_number

            if current_page is None:
                current_page = p.page_number

            buffer.append(text)

        flush()
        return paragraphs
