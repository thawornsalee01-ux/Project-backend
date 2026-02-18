from typing import List
from sqlalchemy.orm import Session, joinedload
from langchain.tools import tool

from src.db.session import SessionLocal
from src.db.models import ChangeItem, Comparison


@tool
def get_change_paragraph(change_id: int) -> str:
    """
    ดึงข้อมูล paragraph เป้าหมายจากฐานข้อมูล
    รวมถึง AI comment, AI suggestion
    และบริบท change อื่นในเอกสารเดียวกัน
    """

    db: Session = SessionLocal()

    try:
        # =====================================================
        # 1️⃣ โหลด change หลัก
        # =====================================================
        change = (
            db.query(ChangeItem)
            .options(joinedload(ChangeItem.comparison).joinedload(Comparison.document))
            .filter(ChangeItem.id == change_id)
            .first()
        )

        if not change:
            return "❌ ไม่พบข้อมูล change_id นี้ในระบบ"

        comparison = change.comparison
        document = comparison.document if comparison else None
        doc_name = document.name if document else "unknown"

        # =====================================================
        # 2️⃣ โหลด change อื่นในเอกสารเดียวกัน (context เสริม)
        # =====================================================
        related_changes: List[ChangeItem] = (
            db.query(ChangeItem)
            .filter(
                ChangeItem.comparison_id == comparison.id,
                ChangeItem.id != change_id
            )
            .limit(5)
            .all()
        )

        related_context = ""
        if related_changes:
            for rc in related_changes:
                related_context += f"""
- Section: {rc.section_label or "-"}
  ประเภท: {rc.change_type}
  AI Comment: {rc.ai_comment or "-"}
"""
        else:
            related_context = "ไม่มีบริบท change อื่นในเอกสารนี้"

        # =====================================================
        # 3️⃣ สร้างข้อความส่งกลับให้ LLM
        # =====================================================
        result = f"""
==============================
ข้อมูลเอกสาร
==============================
ชื่อเอกสาร: {doc_name}

==============================
PARAGRAPH เป้าหมาย
==============================
Section: {change.section_label or "-"}
ประเภทการเปลี่ยนแปลง: {change.change_type}

------------------------------
ข้อความเดิม:
{change.old_text or "-"}

------------------------------
ข้อความใหม่:
{change.new_text or "-"}

==============================
AI วิเคราะห์เดิม
==============================
AI Comment:
{change.ai_comment or "ไม่มีความคิดเห็นจาก AI"}

AI Suggestion:
{change.ai_suggestion or "ไม่มีข้อเสนอแนะจาก AI"}

==============================
บริบทจากส่วนอื่นในเอกสารเดียวกัน
==============================
{related_context}
"""

        return result.strip()

    except Exception as e:
        return f"[GET_CHANGE_TOOL_ERROR] {str(e)}"

    finally:
        db.close()
