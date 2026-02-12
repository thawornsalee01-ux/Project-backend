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

# AI (เหมือน run_compare ใหม่)
from src.AI.ai_comment import run_generate_ai_comment_parallel
from src.AI.ai_suggestion import run_generate_ai_suggestion_parallel
from src.AI.ai_sum import build_summary_text

logger = logging.getLogger(__name__)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def run_compare_v2(
    document_id: int,
    v2_file_bytes: Optional[bytes] = None,
    v2_label: str = "v2",
) -> dict:
    """
    Compare document (ไฟล์เดียว)
    -> ใช้ Logic เดียวกับ run_compare ใหม่ทุกอย่าง
    -> ต่างกันแค่ขั้นตอนโหลด Baseline + V2
    """

    start_time = time.perf_counter()
    print("\n🕒 เริ่มจับเวลา run_compare_v2 ...")

    if v2_file_bytes is None:
        raise RuntimeError("No file uploaded (v2_file_bytes is None)")

    # ==================================================
    # 1) โหลด BASELINE (V1 = previous NEW) จาก DB
    # ==================================================
    print("📄 Loading BASELINE (previous NEW) from DB...")

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
            raise RuntimeError("No NEW page-level data found for this document")

        pages_old = [
            type("Page", (), {"text": r.text_page, "page_number": r.page})
            for r in old_page_records
        ]

        print(
            f"  Loaded {len(pages_old)} pages from BASELINE NEW file "
            f"({old_page_records[0].pdf_name})"
        )

    finally:
        db.close()

    # ==================================================
    # 2) โหลด V2 จาก bytes (ไฟล์ใหม่)
    # ==================================================
    print("📄 Loading NEW document from bytes...")
    loader = DocumentLoader()
    pages_new = loader.load_from_bytes(v2_file_bytes)

    print(
        f"  Loaded {len(pages_old)} pages from BASELINE, "
        f"{len(pages_new)} pages from UPLOADED V2"
    )

    full_text_new = "\n".join(p.text for p in pages_new if p.text)

    # ==================================================
    # 3) สร้าง Document ใหม่เสมอ (เหมือน run_compare)
    # ==================================================
    print("🔍 Creating new document entry...")

    db = SessionLocal()
    try:
        last_id = db.query(Document.id).order_by(Document.id.desc()).first()
        new_doc_id = 1 if last_id is None else last_id[0] + 1

        print(f"🆔 Using document_id = {new_doc_id}")

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

    except Exception as e:
        db.rollback()
        logger.exception("❌ Failed to create new document or save pages")
        raise RuntimeError(f"Failed to process document: {e}")

    finally:
        db.close()

    # ==================================================
    # 4) Split paragraphs (เหมือน run_compare)
    # ==================================================
    print("✂ Splitting paragraphs...")
    splitter = ParagraphSplitter()
    old_paragraphs = splitter.split(pages_old)
    new_paragraphs = splitter.split(pages_new)

    pages_v1_count = len(pages_old)
    pages_v2_count = len(pages_new)
    paragraphs_v1_count = len(old_paragraphs)
    paragraphs_v2_count = len(new_paragraphs)

    # ==================================================
    # 5) Embedding
    # ==================================================
    print("🔗 Embedding paragraphs...")
    embedder = EmbeddingService()
    embedder.embed_paragraphs(old_paragraphs)
    embedder.embed_paragraphs(new_paragraphs)

    # ==================================================
    # 6) Matching
    # ==================================================
    print("🔍 Matching paragraphs...")
    matcher = ParagraphMatcher(threshold=0.75)
    stage1_matches = matcher.match(old_paragraphs, new_paragraphs)

    # ==================================================
    # 7) Resolve
    # ==================================================
    print("🧠 Resolving semantic changes...")
    resolver = MatchResolver(chunk_threshold=0.85)
    resolved_matches = resolver.resolve(
        stage1_matches, old_paragraphs, new_paragraphs
    )

    # ==================================================
    # 8) Diff
    # ==================================================
    print("📝 Building diff...")
    diff_engine = DiffEngine()
    changes: List[DiffChange] = diff_engine.build_changes(resolved_matches)

    edit_intensity = diff_engine.compute_edit_intensity(changes)
    print("✏️ Edit Intensity:", edit_intensity)

    # ==================================================
    # 9) AI (เหมือน run_compare ใหม่)
    # ==================================================
    print("🤖 LLM per-change analysis (parallel) ...")
    await run_generate_ai_comment_parallel(changes)
    await run_generate_ai_suggestion_parallel(changes)

    print("🤖 Building summary + impact analysis ...")
    summary_result = build_summary_text(changes)

    summary_text = summary_result.get("summary_text", "")
    overall_risk_level = summary_result.get("overall_risk_level", "LOW")
    impact_scores = summary_result.get(
        "impact_scores",
        {
            "scope_impact_score": 0,
            "timeline_impact_score": 0,
            "cost_impact_score": 0,
            "resource_impact_score": 0,
            "risk_impact_score": 0,
            "contract_impact_score": 0,
            "stakeholder_impact_score": 0,
            "architecture_impact_score": 0,
        },
    )
    risk_comment = summary_result.get(
        "risk_comment",
        "ไม่พบความเสี่ยงที่มีนัยสำคัญจากภาพรวมการเปลี่ยนแปลง",
    )

    # ==================================================
    # 10) บันทึก Version + Comparison (เหมือน run_compare)
    # ==================================================
    db = SessionLocal()
    try:
        new_doc = db.query(Document).filter(Document.id == new_doc_id).first()
        new_pdf_name = f"{doc_name}_{v2_label}.pdf".replace(" ", "_")

        ver2 = create_document_version(
            db, new_doc, v2_label, new_pdf_name
        )
        db.flush()

        db.add(
            DocumentVersionText(
                document_version_id=ver2.id,
                original_filename=new_pdf_name,
                full_text=full_text_new,
                extractor="DocumentLoader",
            )
        )

        # ดึง ver1 ใหม่ใน session นี้
        ver1 = (
            db.query(DocumentVersion)
            .filter(DocumentVersion.id == baseline_version_id)
            .first()
        )

        comp = create_comparison(
            db, new_doc, ver1, ver2, overall_risk_level, summary_text
        )

        comp.id = new_doc_id
        db.flush()

        comp.edit_intensity = edit_intensity

        # บันทึก impact scores
        comp.scope_impact_score = float(impact_scores.get("scope_impact_score", 0))
        comp.timeline_impact_score = float(impact_scores.get("timeline_impact_score", 0))
        comp.cost_impact_score = float(impact_scores.get("cost_impact_score", 0))
        comp.resource_impact_score = float(impact_scores.get("resource_impact_score", 0))
        comp.risk_impact_score = float(impact_scores.get("risk_impact_score", 0))
        comp.contract_impact_score = float(impact_scores.get("contract_impact_score", 0))
        comp.stakeholder_impact_score = float(impact_scores.get("stakeholder_impact_score", 0))
        comp.architecture_impact_score = float(impact_scores.get("architecture_impact_score", 0))

        comp.risk_comment = risk_comment
        comp.overall_risk_level = overall_risk_level or "LOW"

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

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()

    # ==================================================
    # 11) สร้าง Report (เหมือน run_compare)
    # ==================================================
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
    elapsed_seconds = end_time - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = elapsed_seconds % 60

    print(f"\n⏱️ TOTAL RUNTIME: {round(elapsed_seconds, 2)} seconds "
          f"({minutes} minutes {round(seconds, 2)} seconds)")

    # ==================================================
    # ✅ RETURN — เหมือน run_compare ใหม่
    # ==================================================
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
        "runtime_seconds": round(seconds, 2),
    }
