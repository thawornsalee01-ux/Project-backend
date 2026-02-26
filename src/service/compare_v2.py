from pathlib import Path
import logging
import time
import hashlib
from typing import Optional, List

from src.db.models import (
    Document,
    DocumentVersionText,
    DocumentVersion,
    DocumentPageText,
    Comparison,
)
from src.db.session import SessionLocal
from src.db.ops import (
    create_document_version,
    create_comparison,
    bulk_insert_changes,
    save_document_pages,
)
from src.ingestion.document_load import DocumentLoader
from src.ingestion.paragraph import ParagraphSplitter
from src.embedding.embed import EmbeddingService
from src.match.paragraph_match import ParagraphMatcher
from src.match.match_resolver import MatchResolver
from src.diff.diff import DiffEngine, Change as DiffChange

from src.report.report_builder import ReportBuilder

from src.AI.ai_comment import run_generate_ai_comment_parallel
from src.AI.ai_suggestion import run_generate_ai_suggestion_parallel
from src.AI.ai_sum import build_summary_text

logger = logging.getLogger(__name__)


async def run_compare_v2(
    document_id: int,
    v2_file_bytes: Optional[bytes] = None,
    v2_label: str = "v2",
    progress_callback=None,
) -> dict:

    def update(step: str, progress: int | None = None):
        logger.info(step)
        print(step)
        if progress_callback:
            try:
                progress_callback(step, progress)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")

    start_time = time.perf_counter()
    update("🚀 เริ่มการทำงาน...", 1)

    if v2_file_bytes is None:
        raise RuntimeError("No file uploaded (v2_file_bytes is None)")

    # ==================================================
    # 1) LOAD BASELINE
    # ==================================================
    update("📄 กำลังอ่านเอกสาร...", 5)

    db = SessionLocal()
    try:
        latest_comp = (
            db.query(Comparison)
            .filter(Comparison.document_id == document_id)
            .order_by(Comparison.created_at.desc())
            .first()
        )

        if not latest_comp:
            raise RuntimeError("No comparison found for this document")

        baseline_version_id = latest_comp.version_new_id

        ver1 = (
            db.query(DocumentVersion)
            .filter(DocumentVersion.id == baseline_version_id)
            .first()
        )

        if not ver1:
            raise RuntimeError("Baseline DocumentVersion not found")

        old_document = ver1.document
        doc_name = old_document.name
        v1_label = ver1.version_label

        old_page_records = (
            db.query(DocumentPageText)
            .filter(
                DocumentPageText.document_id == document_id,
                DocumentPageText.version_pdf == "new",
            )
            .order_by(DocumentPageText.page)
            .all()
        )

        if not old_page_records:
            raise RuntimeError("No NEW page-level data found")

        pages_old = [
            type("Page", (), {"text": r.text_page, "page_number": r.page})
            for r in old_page_records
        ]

    finally:
        db.close()

    # ==================================================
    # 2) LOAD NEW FILE
    # ==================================================
    update("📄 กำลังอ่านเอกสาร...", 10)

    loader = DocumentLoader()
    pages_new = loader.load_from_bytes(v2_file_bytes)
    full_text_new = "\n".join(p.text for p in pages_new if p.text)

    # ==================================================
    # 3) CREATE DOCUMENT
    # ==================================================
    update("🆕 กำลังสร้างเอกสารใหม่...", 15)

    db = SessionLocal()
    try:
        last_id = db.query(Document.id).order_by(Document.id.desc()).first()
        new_doc_id = 1 if last_id is None else last_id[0] + 1

        new_doc = Document(id=new_doc_id, name=doc_name)
        db.add(new_doc)
        db.flush()

        save_document_pages(
            db=db,
            document_id=new_doc_id,
            pages=pages_new,
            pdf_name=f"{doc_name}_{v2_label}.pdf".replace(" ", "_"),
            version_pdf="new",
        )

        db.commit()
    finally:
        db.close()

    # ==================================================
    # 4) SPLIT
    # ==================================================
    update("✂ กำลังอ่านเอกสาร...", 25)

    splitter = ParagraphSplitter()
    old_paragraphs = splitter.split(pages_old)
    new_paragraphs = splitter.split(pages_new)

    pages_v1_count = len(pages_old)
    pages_v2_count = len(pages_new)
    paragraphs_v1_count = len(old_paragraphs)
    paragraphs_v2_count = len(new_paragraphs)

    # ==================================================
    # 5) EMBEDDING
    # ==================================================
    update("🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 40)

    embedder = EmbeddingService()
    embedder.embed_paragraphs(old_paragraphs)
    embedder.embed_paragraphs(new_paragraphs)

    # ==================================================
    # 6) MATCH
    # ==================================================
    update("🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 55)

    matcher = ParagraphMatcher(threshold=0.75)
    stage1_matches = matcher.match(old_paragraphs, new_paragraphs)

    # ==================================================
    # 7) RESOLVE
    # ==================================================
    update("🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 65)

    resolver = MatchResolver(chunk_threshold=0.85)
    resolved_matches = resolver.resolve(
        stage1_matches, old_paragraphs, new_paragraphs
    )

    # ==================================================
    # 8) DIFF
    # ==================================================
    update("📝 🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 75)

    diff_engine = DiffEngine()
    changes: List[DiffChange] = diff_engine.build_changes(resolved_matches)
    edit_intensity = diff_engine.compute_edit_intensity(changes)

    # ==================================================
    # 9) AI
    # ==================================================
    update("🤖 กำลังวิเคราะห์การเปลี่ยนแปลงที่เกิดขึ้นด้วย AI ...", 80)

    await run_generate_ai_comment_parallel(changes)
    update("🤖 กำลังวิเคราะห์และให้คำแนะนำด้วย AI ...", 85)
    await run_generate_ai_suggestion_parallel(changes)

    update("📊 กำลังสรุปผลการเปรียบเทียบ...", 88)

    summary_result = build_summary_text(changes)
    summary_text = summary_result.get("summary_text", "")
    overall_risk_level = summary_result.get("overall_risk_level", "LOW")
    impact_scores = summary_result.get("impact_scores", {})
    risk_comment = summary_result.get("risk_comment", "")

    # ==================================================
    # 10) SAVE DB
    # ==================================================
    update("💾 กำลังบันทึกข้อมูล...", 92)

    db = SessionLocal()
    try:
        new_doc = db.query(Document).filter(Document.id == new_doc_id).first()
        new_pdf_name = f"{doc_name}_{v2_label}.pdf".replace(" ", "_")

        ver2 = create_document_version(db, new_doc, v2_label, new_pdf_name)
        db.flush()

        db.add(
            DocumentVersionText(
                document_version_id=ver2.id,
                original_filename=new_pdf_name,
                full_text=full_text_new,
                extractor="DocumentLoader",
            )
        )

        ver1 = db.query(DocumentVersion).filter(
            DocumentVersion.id == baseline_version_id
        ).first()

        comp = create_comparison(
            db, new_doc, ver1, ver2, overall_risk_level, summary_text
        )

        comp.id = new_doc_id
        db.flush()

        # ⭐ FIX: ทำเหมือน v1
        comp.edit_intensity = edit_intensity
        comp.scope_impact_score = impact_scores.get("scope_impact_score")
        comp.timeline_impact_score = impact_scores.get("timeline_impact_score")
        comp.cost_impact_score = impact_scores.get("cost_impact_score")
        comp.resource_impact_score = impact_scores.get("resource_impact_score")
        comp.risk_impact_score = impact_scores.get("risk_impact_score")
        comp.contract_impact_score = impact_scores.get("contract_impact_score")
        comp.stakeholder_impact_score = impact_scores.get("stakeholder_impact_score")
        comp.architecture_impact_score = impact_scores.get("architecture_impact_score")
        comp.risk_comment = risk_comment
        comp.overall_risk_level = overall_risk_level

        bulk_insert_changes(
            db,
            comp,
            [
                {
                    "change_type": c.change_type,
                    "section_label": c.section_label,
                    "old_text": c.old_text,
                    "new_text": c.new_text,
                    "ai_comment": getattr(c, "ai_comment", None),
                    "ai_suggestion": getattr(c, "ai_suggestion", None),
                }
                for c in changes
            ],
        )

        db.commit()
        run_id = comp.id
    finally:
        db.close()

    # ==================================================
    # 11) REPORT
    # ==================================================
    update("📁 กำลังสร้างรายงาน...", 97)

    reporter = ReportBuilder()
    json_path = reporter.save_json(doc_name, v1_label, v2_label, changes)
    html_path = reporter.save_html(
        doc_name,
        v1_label,
        v2_label,
        changes,
        summary_text,
        overall_risk_level,
        edit_intensity,
    )

    end_time = time.perf_counter()
    runtime = end_time - start_time
    minutes = int(runtime // 60)
    seconds = round(runtime % 60, 2)

    update(f"⏱️ เวลาการทำงาน: {minutes} นาที {seconds} วินาที", 100)

    return {
        "doc_name": doc_name,
        "v1_label": v1_label,
        "v2_label": v2_label,
        "pages_v1": pages_v1_count,
        "pages_v2": pages_v2_count,
        "paragraphs_v1": paragraphs_v1_count,
        "paragraphs_v2": paragraphs_v2_count,
        "changes_count": len(changes),
        "edit_intensity": edit_intensity,
        "summary_text": summary_text,
        "overall_risk_level": overall_risk_level,
        "impact_scores": impact_scores,
        "risk_comment": risk_comment,
        "json_report_path": str(json_path),
        "html_report_path": str(html_path),
        "json_report_url": f"/reports/{Path(json_path).name}",
        "html_report_url": f"/reports/{Path(html_path).name}",
        "run_id": run_id,
        "runtime_minutes": minutes,
        "runtime_seconds": seconds,
    }
