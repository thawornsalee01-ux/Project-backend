from dataclasses import dataclass
from typing import List, Optional
from src.match.paragraph_match import MatchResult


# ==================================================
# Change model (ใช้ downstream: DB / Report / Frontend)
# ==================================================
@dataclass
class Change:
    """
    ตัวแทน 1 จุดที่มีการเปลี่ยนแปลงระหว่าง V1 ↔ V2
    """
    change_type: str                # ADDED / REMOVED / MODIFIED
    section_label: str              # เช่น "page 3"
    old_text: Optional[str]
    new_text: Optional[str]

    # --- verdict (มาจาก MatchResolver โดยตรง) ---
    edit_severity: Optional[str] = None   # LIGHT / MEDIUM / HEAVY
    status: Optional[str] = None          # added / removed / modified

    # --- metrics ---
    similarity: Optional[float] = None
    coverage: Optional[float] = None
    mean_similarity: Optional[float] = None


# ==================================================
# Diff Engine
# ==================================================
class DiffEngine:
    """
    แปลง MatchResult → Change

    หลักการ:
    - ❗ ไม่วิเคราะห์ใหม่
    - ❗ ไม่แตะ logic semantic
    - ✅ เชื่อผลจาก MatchResolver 100%
    """

    def build_changes(self, matches: List[MatchResult]) -> List[Change]:
        changes: List[Change] = []

        for m in matches:
            ct = m.change_type

            # --------------------------------------------------
            # UNCHANGED → ข้าม
            # --------------------------------------------------
            if ct == "UNCHANGED":
                continue

            # --------------------------------------------------
            # MODIFIED (ใช้ edit_severity จาก resolver ตรง ๆ)
            # --------------------------------------------------
            if ct == "MODIFIED":
                page = m.new_page

                changes.append(
                    Change(
                        change_type="MODIFIED",
                        status="modified",
                        edit_severity=getattr(m, "edit_severity", "LIGHT"),

                        section_label=f"page {page}" if page is not None else "unknown",
                        old_text=m.old_text,
                        new_text=m.new_text,

                        similarity=m.similarity,
                        coverage=m.chunk_coverage,
                        mean_similarity=m.mean_chunk_similarity,
                    )
                )

            # --------------------------------------------------
            # REMOVED → ถือว่า HEAVY เสมอ
            # --------------------------------------------------
            elif ct == "REMOVED":
                page = m.old_page
                changes.append(
                    Change(
                        change_type="REMOVED",
                        status="removed",
                        edit_severity="HEAVY",

                        section_label=f"page {page}" if page is not None else "unknown",
                        old_text=m.old_text,
                        new_text=None,
                    )
                )

            # --------------------------------------------------
            # ADDED → ถือว่า MEDIUM
            # --------------------------------------------------
            elif ct == "ADDED":
                page = m.new_page
                changes.append(
                    Change(
                        change_type="ADDED",
                        status="added",
                        edit_severity="MEDIUM",

                        section_label=f"page {page}" if page is not None else "unknown",
                        old_text=None,
                        new_text=m.new_text,
                    )
                )

            # --------------------------------------------------
            # CANDIDATE → ไม่ควรหลงเหลือ
            # --------------------------------------------------
            elif ct == "CANDIDATE":
                continue

        return changes

    # ==================================================
    # Document-level edit intensity
    # ==================================================
    def compute_edit_intensity(self, changes: List[Change]) -> str:
        """
        สรุปว่าเอกสารถูกแก้มากแค่ไหน (ภาพรวม)
        ใช้ edit_severity ที่ resolver ตัดสินไว้แล้ว
        """

        if not changes:
            return "NONE"

        modified = [c for c in changes if c.change_type == "MODIFIED"]
        if not modified:
            return "NONE"

        # ให้ MEDIUM มีน้ำหนักครึ่งหนึ่ง
        score = 0.0
        for c in modified:
            if c.edit_severity == "HEAVY":
                score += 1.0
            elif c.edit_severity == "MEDIUM":
                score += 0.5

        ratio = score / max(len(modified), 1)

        if ratio >= 0.5:
            return "HIGH"
        elif ratio >= 0.2:
            return "MEDIUM"
        else:
            return "LOW"
