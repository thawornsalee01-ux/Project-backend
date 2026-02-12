# src/db/ops.py

from typing import Optional, List
from sqlalchemy.orm import Session
from fastapi import HTTPException
import logging
from src.diff.diff import Change
from .models import Document, DocumentVersion, Comparison, ChangeItem, DocumentPageText, DocumentVersionText

logger = logging.getLogger(__name__)

def get_or_create_document(db: Session, name: str, category: Optional[str] = None) -> Document:
    doc = db.query(Document).filter(Document.name == name).first()
    if doc:
        return doc
    doc = Document(name=name, category=category)
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def create_document_version(
    db: Session,
    document: Document,
    version_label: str,
    file_path: str,
    uploaded_by: Optional[str] = None,
) -> DocumentVersion:
    ver = DocumentVersion(
        document_id=document.id,
        version_label=version_label,
        file_path=file_path,
        uploaded_by=uploaded_by,
    )
    db.add(ver)
    db.commit()
    db.refresh(ver)
    return ver


def create_comparison(
    db: Session,
    document: Document,
    version_old: DocumentVersion,
    version_new: DocumentVersion,
    overall_risk_level: Optional[str],
    summary_text: Optional[str],
) -> Comparison:
    comp = Comparison(
        document_id=document.id,
        version_old_id=version_old.id,
        version_new_id=version_new.id,
        overall_risk_level=overall_risk_level,
        summary_text=summary_text,
    )
    db.add(comp)
    db.commit()
    db.refresh(comp)
    return comp


def bulk_insert_changes(
    db: Session,
    comparison: Comparison,
    changes: List[dict],
):
    items = []

    for ch in changes:
        item = ChangeItem(
            comparison_id=comparison.id,

            change_type=ch["change_type"],
            section_label=ch.get("section_label"),

            old_text=ch.get("old_text"),
            new_text=ch.get("new_text"),

            # semantic
            edit_severity=ch.get("edit_severity"),

            ai_comment=ch.get("ai_comment"),
            ai_suggestion=ch.get("ai_suggestion"),
        )

        db.add(item)
        items.append(item)

    db.commit()
    return items
# src/db/ops.py  (ต่อจากฟังก์ชันเดิม)

from sqlalchemy import desc
from .models import Document, DocumentVersion, Comparison, ChangeItem

# ...

def list_comparisons(
    db: Session,
    doc_name: Optional[str] = None,
    limit: int = 50,
) -> List[Comparison]:
    """
    ดึงรายการ comparison ย้อนหลัง
    - ถ้ามี doc_name จะ filter ตามชื่อเอกสาร
    - คืนค่าเรียงจากใหม่สุด → เก่าสุด
    """
    q = db.query(Comparison).join(Document, Comparison.document_id == Document.id)

    if doc_name:
        q = q.filter(Document.name == doc_name)

    q = q.order_by(desc(Comparison.created_at))
    return q.limit(limit).all()


def get_comparison_with_changes(
    db: Session,
    comparison_id: int,
) -> Optional[Comparison]:
    """
    ดึง comparison + changes ทั้งหมดของ run นั้น
    """
    comp = (
        db.query(Comparison)
        .filter(Comparison.id == comparison_id)
        .first()
    )
    # แค่ access .changes เพื่อให้ ORM load มา
    if comp:
        _ = comp.changes  # trigger lazy load
    return comp

def delete_comparison_by_id(db: Session, comparison_id: int) -> None:
    comp = db.query(Comparison).filter(Comparison.id == comparison_id).first()

    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")

    document_id = comp.document_id

    try:
        # 🔥 1) ลบ Changes ที่ผูกกับ comparison นี้
        db.query(ChangeItem).filter(
            ChangeItem.comparison_id == comparison_id
        ).delete(synchronize_session=False)

        # 🔥 2) ลบ DocumentVersionText ที่ผูกกับ document นี้
        db.query(DocumentVersionText).filter(
            DocumentVersionText.document_version_id.in_(
                db.query(DocumentVersion.id).filter(
                    DocumentVersion.document_id == document_id
                )
            )
        ).delete(synchronize_session=False)

        # 🔥 3) ลบ DocumentVersion ทั้งหมดของ document นี้
        db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).delete(synchronize_session=False)

        # 🔥 4) ลบหน้า PDF (document_page_texts) ทั้งหมดของ document นี้
        db.query(DocumentPageText).filter(
            DocumentPageText.document_id == document_id
        ).delete(synchronize_session=False)

        # 🔥 5) ลบ Comparison ตัวนี้
        db.delete(comp)

        db.commit()

    except Exception as e:
        logger.exception("Delete comparison failed: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete comparison")
    
def save_document_pages(
    db: Session,
    document_id: int,
    pages: list,
    pdf_name: str,
    version_pdf: str,   # "old" หรือ "new"
):
    """
    บันทึก text ทีละหน้าเข้า DocumentPageText
    """
    page_records = []

    for i, p in enumerate(pages, start=1):
        if not p.text:
            continue

        page_records.append(
            DocumentPageText(
                document_id=document_id,
                pdf_name=pdf_name,
                page=i,
                text_page=p.text,
                version_pdf=version_pdf,
            )
        )

    db.bulk_save_objects(page_records)
    db.commit()