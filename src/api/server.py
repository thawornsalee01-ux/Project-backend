import sys
import os
import time
from pathlib import Path
import threading
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any
import logging
from pydantic import BaseModel

from service.compare import run_compare
from service.compare_v2 import run_compare_v2
from db.session import SessionLocal, engine, Base
from db.models import Comparison
from db.ops import (
    get_comparison_with_changes,
)
from AI.agent_rewrite import generate_rewrite_suggestion_for_row
from sqlalchemy.orm import Session
from src.report.report_builder import ReportBuilder
from src.diff.diff import Change as DiffChange

from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# ----------------- Config -----------------
LOCALMODEL_API_KEY = os.getenv("LOCALMODEL_API_KEY")
LOCALMODEL_BASE_URL = os.getenv("LOCALMODEL_BASE_URL", "http://localhost:11434/v1")
LOCALMODEL_MODEL = os.getenv("LOCALMODEL_MODEL", "openai/gpt-oss-120b")

print("🔧 LOCALMODEL_BASE_URL =", LOCALMODEL_BASE_URL)
print("🔧 LOCALMODEL_MODEL    =", LOCALMODEL_MODEL)
print("🔧 HAS_API_KEY         =", bool(LOCALMODEL_API_KEY))

_effective_api_key = LOCALMODEL_API_KEY or "local-dev-key"

local_client = OpenAI(
    api_key=_effective_api_key,
    base_url=LOCALMODEL_BASE_URL,
    timeout=300,
    max_retries=0,
)

logger = logging.getLogger(__name__)

# ----------------- FastAPI -----------------
app = FastAPI(title="Document Versioning Compare API")

@app.on_event("startup")
def on_startup():
    Path("data").mkdir(parents=True, exist_ok=True)
    Path("data/outputs").mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("✅ DB tables ensured (create_all)")

# ----------------- Middleware -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/reports", StaticFiles(directory=str(Path("data/outputs"))), name="reports")

# ----------------- DB Dependency -----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================================================
# 🔥🔥🔥  เพิ่มส่วน JOB + POLLING ตรงนี้  🔥🔥🔥
# ======================================================

# เก็บสถานะงาน (In-Memory)
jobs: Dict[str, Dict[str, Any]] = {}

def process_compare_job(job_id: str, doc_name: str, v1_label: str, v2_label: str,
                        file_v1_bytes: bytes, file_v2_bytes: bytes):
    try:
        # 🔹 ต้องรัน async function ใน Thread ด้วย asyncio.run
        import asyncio

        result = asyncio.run(
            run_compare(
                doc_name=doc_name,
                v1_file_bytes=file_v1_bytes,
                v2_file_bytes=file_v2_bytes,
                v1_label=v1_label,
                v2_label=v2_label,
            )
        )

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result

    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

# --------- (ใหม่) API #1: เริ่มงาน → ได้ job_id ---------
@app.post("/compare")
async def start_compare_job(
    doc_name: str = Form(...),
    v1_label: str = Form("v1"),
    v2_label: str = Form("v2"),
    file_v1: UploadFile = File(...),
    file_v2: UploadFile = File(...),
):
    job_id = str(uuid.uuid4())

    v1_bytes = await file_v1.read()
    v2_bytes = await file_v2.read()

    jobs[job_id] = {
        "status": "processing",
        "result": None,
        "error": None,
    }

    # รันงานใน Thread แยก
    threading.Thread(
        target=process_compare_job,
        args=(job_id, doc_name, v1_label, v2_label, v1_bytes, v2_bytes),
        daemon=True,
    ).start()

    return {"job_id": job_id}

# --------- (ใหม่) API #2: เช็คสถานะงาน ---------
@app.get("/compare/status/{job_id}")
def check_status(job_id: str):
    if job_id not in jobs:
        return {"status": "not_found"}

    return {"status": jobs[job_id]["status"]}

# --------- (ใหม่) API #3: ดึงผลลัพธ์ ---------
@app.get("/compare/result/{job_id}")
def get_result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != "done":
        raise HTTPException(status_code=400, detail="Job not ready")

    return jobs[job_id]["result"]

# ======================================================
# 🔥🔥🔥  จบส่วน JOB + POLLING  🔥🔥🔥
# ======================================================

# ----------------- Pydantic Models -----------------
class CompareResultModel(BaseModel):
    doc_name: str
    v1_label: str
    v2_label: str
    pages_v1: int
    pages_v2: int
    paragraphs_v1: int
    paragraphs_v2: int
    changes_count: int
    overall_risk_level: str
    summary_text: str
    json_report_path: str
    html_report_path: str
    run_id: int
    html_report_url: str
    json_report_url: str

class ComparisonItemModel(BaseModel):
    id: int
    document_name: str
    version_old_label: str
    version_new_label: str
    created_at: str
    overall_risk_level: Optional[str] = None
    legal_score: float
    financial_score: float
    operational_score: float

class ChangeItemModel(BaseModel):
    id: int
    change_type: str
    section_label: Optional[str]
    old_text: Optional[str]
    new_text: Optional[str]
    risk_level: Optional[str] = None
    ai_comment: Optional[str] = None
    ai_suggestion: Optional[str] = None
    status: Optional[str] = None

class ComparisonDetailModel(BaseModel):
    id: int
    document_name: str
    version_old_label: str
    version_new_label: str
    created_at: str
    overall_risk_level: Optional[str]
    summary_text: Optional[str]
    changes: List[ChangeItemModel]

class AnnotateResultModel(BaseModel):
    comparison_id: int
    updated_count: int
    message: str

class AIChatRequest(BaseModel):
    question: str

class AIChatResponse(BaseModel):
    answer: str

# ----------------- AI / LLM Helpers -----------------
def call_llm(prompt: str) -> str:
    if not LOCALMODEL_API_KEY:
        raise RuntimeError("LOCALMODEL_API_KEY is not set in environment variables")

    print("🚀 CALL_LLM → model =", LOCALMODEL_MODEL, "base =", LOCALMODEL_BASE_URL)

    resp = local_client.chat.completions.create(
        model=LOCALMODEL_MODEL,
        messages=[
            {"role": "system", "content": "..."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    content = resp.choices[0].message.content
    if isinstance(content, list):
        text = "".join(part.get("text", "") for part in content)
    else:
        text = str(content)

    return text.strip()

def safe_call_llm(prompt: str, retries: int = 2, delay: float = 1.0) -> str:
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return call_llm(prompt)
        except Exception as e:
            last_exc = e
            logger.warning("call_llm attempt %d failed: %s", attempt + 1, str(e))
            time.sleep(delay)
    logger.exception("LLM unreachable after retries: %s", last_exc)
    return ""

# ----------------- AI Annotate Endpoint -----------------
@app.post("/comparisons/{comparison_id}/annotate", response_model=AnnotateResultModel)
async def annotate_comparison(comparison_id: int, db=Depends(get_db)):
    comp = get_comparison_with_changes(db, comparison_id=comparison_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")

    updated = 0
    for ch in comp.changes:
        try:
            impact_comment, rewrite_suggestion, scores = generate_rewrite_suggestion_for_row(
                change_type=ch.change_type,
                section_label=ch.section_label,
                old_text=ch.old_text,
                new_text=ch.new_text,
                risk_level=ch.risk_level,
                call_llm=safe_call_llm,
            )
        except Exception as e:
            logger.exception("annotate error for change id=%s: %s", ch.id, e)
            impact_comment = "การวิเคราะห์ล้มเหลว ช่วยตรวจสอบด้วยตนเอง"
            rewrite_suggestion = ""
            scores = {"legal": 0, "financial": 0, "operational": 0}

        ch.ai_comment = impact_comment
        ch.ai_suggestion = rewrite_suggestion

        ch.legal_score = scores.get("legal", 0)
        ch.financial_score = scores.get("financial", 0)
        ch.operational_score = scores.get("operational", 0)

        avg = (ch.legal_score + ch.financial_score + ch.operational_score) / 3
        if avg >= 3.6:
            ch.risk_level = "HIGH"
        elif avg >= 2:
            ch.risk_level = "MEDIUM"
        else:
            ch.risk_level = "LOW"

        updated += 1

    legal_scores = [c.legal_score for c in comp.changes]
    financial_scores = [c.financial_score for c in comp.changes]
    operational_scores = [c.operational_score for c in comp.changes]

    def avg_list(xs):
        return sum(xs) / len(xs) if xs else 0

    comp.legal_score_avg = round(avg_list(legal_scores), 2)
    comp.financial_score_avg = round(avg_list(financial_scores), 2)
    comp.operational_score_avg = round(avg_list(operational_scores), 2)

    overall_avg = (
        comp.legal_score_avg
        + comp.financial_score_avg
        + comp.operational_score_avg
    ) / 3

    if overall_avg >= 3.6:
        comp.overall_risk_level = "HIGH"
    elif overall_avg >= 2:
        comp.overall_risk_level = "MEDIUM"
    else:
        comp.overall_risk_level = "LOW"

    try:
        db.commit()
    except Exception as e:
        logger.exception("DB commit failed: %s", e)
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to commit AI annotations")

    return AnnotateResultModel(
        comparison_id=comparison_id,
        updated_count=updated,
        message=f"annotated {updated} changes with AI comments, suggestions & scores",
    )

# ----------------- Generate Report from DB -----------------
@app.get("/comparisons/{comparison_id}/report")
def generate_report_from_db(comparison_id: int, db: Session = Depends(get_db)):
    comp = get_comparison_with_changes(db, comparison_id=comparison_id)
    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")

    changes = []
    for ch in comp.changes:
        c = DiffChange(
            change_type=ch.change_type,
            section_label=ch.section_label,
            old_text=ch.old_text,
            new_text=ch.new_text,
        )
        c.risk_level = ch.risk_level
        c.ai_comment = ch.ai_comment
        c.ai_suggestion = ch.ai_suggestion
        changes.append(c)

    builder = ReportBuilder(output_dir="data/outputs")

    html_path = builder.save_html(
        doc_name=comp.document.name if comp.document else "document",
        v1_label=comp.version_old.version_label if comp.version_old else "v1",
        v2_label=comp.version_new.version_label if comp.version_new else "v2",
        changes=changes,
        summary_text=comp.summary_text,
        overall_risk_level=comp.overall_risk_level,
    )

    return {
        "html_report_url": f"/reports/{html_path.name}"
    }
