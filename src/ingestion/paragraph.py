# src/ingestion/paragraph.py

import re
from dataclasses import dataclass
from typing import List, Optional
from src.ingestion.pdf_load import PageText


@dataclass
class Paragraph:
    page_number: int
    index: int
    text: str
    embedding: Optional[List[float]] = None


class ParagraphSplitter:
    """
    Paragraph splitter สำหรับ TOR / สัญญา / ระเบียบราชการ
    """

    THAI_DIGITS = "๑๒๓๔๕๖๗๘๙๐"

    # ๑. หัวข้อระดับหลัก (ต้อง split)
    MAIN_HEADING = re.compile(
        rf"^\s*([{THAI_DIGITS}]+|\d+)\.\s+[^\d]"
    )

    # ๑.๑ ห้าม split
    SUB_HEADING = re.compile(r"^\s*\d+\.\d+")

    # bullet / list
    BULLET = re.compile(r"^[-•]")

    def split(self, pages: List[PageText]) -> List[Paragraph]:
        paragraphs: List[Paragraph] = []
        buffer: List[str] = []

        global_index = 0
        current_page: Optional[int] = None

        def flush():
            nonlocal global_index, current_page
            if buffer:
                paragraphs.append(
                    Paragraph(
                        page_number=current_page or 0,
                        index=global_index,
                        text="\n".join(buffer),
                    )
                )
                buffer.clear()
                global_index += 1
                current_page = None

        for page in pages:
            page_no = page.page_number
            lines = [l.strip() for l in (page.text or "").split("\n") if l.strip()]

            for line in lines:
                is_main = bool(self.MAIN_HEADING.match(line))
                is_sub = bool(self.SUB_HEADING.match(line))

                # 🔴 เจอหัวข้อระดับเดียวกัน → ปิด paragraph เก่า
                if is_main and not is_sub:
                    flush()
                    current_page = page_no

                if current_page is None:
                    current_page = page_no

                buffer.append(line)

        flush()
        return paragraphs
