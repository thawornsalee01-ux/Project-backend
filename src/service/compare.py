from pathlib import Path
import logging
import time
from typing import Optional, List
import json

from src.db.models import  Document
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
import asyncio
from src.AI.ai_comment import (
    run_generate_ai_comment_parallel,
)
from src.AI.ai_suggestion import (
    run_generate_ai_suggestion_parallel,
)
from src.AI.ai_sum import build_summary_text

logger = logging.getLogger(__name__)



async def run_compare(
    doc_name: str,
    v1_file_bytes: Optional[bytes] = None,
    v2_file_bytes: Optional[bytes] = None,
    v1_filename: Optional[str] = None,
    v2_filename: Optional[str] = None,
    v1_label: str = "v1",
    v2_label: str = "v2",
    progress_callback=None,   # ⭐ เพิ่มอันเดียว
) -> dict:

    # ⭐ helper log (ไม่กระทบของเดิม)
    def log(msg, progress=None):
        print(msg)
        if progress_callback:
            progress_callback(msg, progress)

    start_time = time.perf_counter()
    log("\n🕒 เริ่มจับเวลาการทำงาน ...", 0)

    # ==================================================
    # Load PDFs
    # ==================================================
    log("📄 กำลังอ่านเอกสาร...", 5)
    loader = DocumentLoader()
    try:
        pages_old = loader.load_from_bytes(v1_file_bytes)
        pages_new = loader.load_from_bytes(v2_file_bytes)

        v1_filename = v1_filename or f"{doc_name}_{v1_label}.pdf"
        v2_filename = v2_filename or f"{doc_name}_{v2_label}.pdf"

        log(
            f"  Loaded {len(pages_old)} pages from V1, {len(pages_new)} pages from V2"
        )

    except Exception as e:
        logger.exception("❌ Failed to load PDFs")
        raise RuntimeError(f"Failed to load PDFs: {e}")

    # ==================================================
    # Split paragraphs
    # ==================================================
    log("📄 กำลังอ่านเอกสาร...", 15)
    splitter = ParagraphSplitter()
    old_paragraphs = splitter.split(pages_old)
    new_paragraphs = splitter.split(pages_new)

    # ==================================================
    # Embedding
    # ==================================================
    log("🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 30)
    embedder = EmbeddingService()
    embedder.embed_paragraphs(old_paragraphs)
    embedder.embed_paragraphs(new_paragraphs)

    # ==================================================
    # Matching
    # ==================================================
    log("🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 45)
    matcher = ParagraphMatcher(threshold=0.75)
    stage1_matches = matcher.match(old_paragraphs, new_paragraphs)

    # ==================================================
    # Resolve
    # ==================================================
    log("🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 60)
    resolver = MatchResolver(chunk_threshold=0.85)
    resolved_matches = resolver.resolve(
        stage1_matches, old_paragraphs, new_paragraphs
    )

    # ==================================================
    # Diff
    # ==================================================
    log("🔍 กำลังตรวจสอบการเปลี่ยนแปลงของเอกสาร...", 70)
    diff_engine = DiffEngine()
    changes: List[DiffChange] = diff_engine.build_changes(resolved_matches)

    edit_intensity = diff_engine.compute_edit_intensity(changes)
    log(f"✏️ ระดับของการเปลี่ยนแปลง: {edit_intensity}")

    # ==================================================
    # AI summary + Risk
    # ==================================================
    log("🤖 กำลังวิเคราะห์การเปลี่ยนแปลงที่เกิดขึ้นด้วย AI ...", 75)
    await run_generate_ai_comment_parallel(changes)
    log("🤖 กำลังวิเคราะห์และให้คำแนะนำด้วย AI ...", 80)
    await run_generate_ai_suggestion_parallel(changes)

    log("📊 กำลังสรุปผลการเปรียบเทียบ...", 85)
    summary_result = build_summary_text(changes)

    summary_text = summary_result["summary_text"]
    overall_risk_level = summary_result["overall_risk_level"]
    impact_scores = summary_result["impact_scores"]
    risk_comment = summary_result["risk_comment"]

    # ==================================================
    # DB SAVE
    # ==================================================
    log("💾 บันทึกข้อมูลลง...", 90)
    db = SessionLocal()
    try:
        last_id = db.query(Document.id).order_by(Document.id.desc()).first()
        if last_id is None:
            new_id = 1
        else:
            new_id = last_id[0] + 1

        log(f"🆔 Using document_id = {new_id}")

        real_doc = Document(id=new_id, name=doc_name)
        db.add(real_doc)
        db.flush()

        save_document_pages(
            db=db,
            document_id=new_id,
            pages=pages_old,
            pdf_name=v1_filename,
            version_pdf="old",
        )

        save_document_pages(
            db=db,
            document_id=new_id,
            pages=pages_new,
            pdf_name=v2_filename,
            version_pdf="new",
        )

        ver1 = create_document_version(db, real_doc, v1_label, v1_filename)
        ver2 = create_document_version(db, real_doc, v2_label, v2_filename)

        comp = create_comparison(
            db, real_doc, ver1, ver2, overall_risk_level, summary_text
        )

        comp.edit_intensity = edit_intensity
        comp.scope_impact_score = impact_scores["scope_impact_score"]
        comp.timeline_impact_score = impact_scores["timeline_impact_score"]
        comp.cost_impact_score = impact_scores["cost_impact_score"]
        comp.resource_impact_score = impact_scores["resource_impact_score"]
        comp.risk_impact_score = impact_scores["risk_impact_score"]
        comp.contract_impact_score = impact_scores["contract_impact_score"]
        comp.stakeholder_impact_score = impact_scores["stakeholder_impact_score"]
        comp.architecture_impact_score = impact_scores["architecture_impact_score"]
        comp.risk_comment = risk_comment
        comp.overall_risk_level = overall_risk_level

        change_dicts: List[dict] = []
        for c in changes:
            change_dicts.append(
                {
                    "change_type": c.change_type,
                    "section_label": c.section_label,
                    "old_text": c.old_text,
                    "new_text": c.new_text,
                    "edit_severity": getattr(c, "edit_severity", None),
                    "ai_comment": getattr(c, "ai_comment", None),
                    "ai_suggestion": getattr(c, "ai_suggestion", None),
                    "paragraph_topic": getattr(c, "paragraph_topic", None),
                    "change_category": getattr(c, "change_category", None),
                    "change_details": json.dumps(
                    getattr(c, "change_details", []),
                    ensure_ascii=False
                    ),
                }
            )

        bulk_insert_changes(db, comp, change_dicts)
        db.commit()
        run_id = comp.id

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # ==================================================
    # Build reports
    # ==================================================
    log("📁 กำลังสร้างรายงาน...", 95)
    reporter = ReportBuilder()
    json_path = Path(reporter.save_json(doc_name, v1_label, v2_label, changes))
    html_path = Path(
        reporter.save_html(
            doc_name,
            v1_label,
            v2_label,
            changes,
            summary_text,
            overall_risk_level,
            edit_intensity,
        )
    )

    end_time = time.perf_counter()
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60

    log(
        f"\n⏱️ เวลาการทำงาน: {round(elapsed_seconds, 2)} วินาที "
        f"({minutes} นาที {round(seconds, 2)} วินาที)",
        100
    )

    return {
        "doc_name": doc_name,
        "v1_label": v1_label,
        "v2_label": v2_label,
        "pages_v1": len(pages_old),
        "pages_v2": len(pages_new),
        "paragraphs_v1": len(old_paragraphs),
        "paragraphs_v2": len(new_paragraphs),
        "changes_count": len(changes),
        "edit_intensity": edit_intensity,
        "summary_text": summary_text,
        "overall_risk_level": overall_risk_level,
        "impact_scores": impact_scores,
        "risk_comment": risk_comment,
        "json_report_path": str(json_path),
        "html_report_path": str(html_path),
        "run_id": run_id,
        "html_report_url": f"/reports/{html_path.name}",
        "json_report_url": f"/reports/{json_path.name}",
        "runtime_minutes": minutes,
        "runtime_seconds": round(seconds, 2),
    }
