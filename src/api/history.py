from fastapi import FastAPI, Depends, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
import uuid
import threading
import asyncio
from enum import Enum

from src.db.session import SessionLocal
from src.db.models import Comparison
from src.db.ops import delete_comparison_by_id
from src.service.compare_v2 import run_compare_v2

# =========================
# ✅ FastAPI App
# =========================
app = FastAPI(title="History API")

# =========================
# 🔹 POLLING: Job Store (เหมือน service compare)
# =========================
class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"

# In-memory job store สำหรับ /compare/continue
continue_jobs: Dict[str, Dict[str, Any]] = {}

# =========================
# DB Dependency
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Pydantic Models
# =========================
class ChangeItemModel(BaseModel):
    id: int
    change_type: str
    section_label: Optional[str]
    old_text: Optional[str]
    new_text: Optional[str]
    risk_level: Optional[str] = None
    ai_comment: Optional[str] = None
    ai_suggestion: Optional[str] = None
    

class ComparisonDetailModel(BaseModel):
    id: int
    document_name: str
    version_old_label: str
    version_new_label: str
    created_at: str
    overall_risk_level: Optional[str]
    summary_text: Optional[str]
    changes: List[ChangeItemModel]

class ComparisonItemModel(BaseModel):
    id: int
    document_name: str
    version_old_label: str
    version_new_label: str
    created_at: str
    overall_risk_level: Optional[str] = None
    scope_impact_score: Optional[float] = None
    timeline_impact_score: Optional[float] = None
    cost_impact_score: Optional[float] = None
    resource_impact_score: Optional[float] = None
    risk_impact_score: Optional[float] = None
    contract_impact_score: Optional[float] = None
    stakeholder_impact_score: Optional[float] = None
    architecture_impact_score: Optional[float] = None

class ImpactScoresModel(BaseModel):
    scope_impact_score: float
    timeline_impact_score: float
    cost_impact_score: float
    resource_impact_score: float
    risk_impact_score: float
    contract_impact_score: float
    stakeholder_impact_score: float
    architecture_impact_score: float  

class CompareResultModel(BaseModel):
    doc_name: str
    v1_label: str
    v2_label: str
    pages_v1: int
    pages_v2: int
    paragraphs_v1: int
    paragraphs_v2: int
    changes_count: int
    edit_intensity: str

    summary_text: str
    overall_risk_level: str
    impact_scores: ImpactScoresModel
    risk_comment: str

    json_report_path: str
    html_report_path: str
    json_report_url: str
    html_report_url: str

    run_id: int
    runtime_minutes: int
    runtime_seconds: float
    

# =========================
# GET /comparisons/{id}
# =========================
@app.get("/comparisons/{comparison_id}", response_model=ComparisonDetailModel)
async def get_comparison_detail(
    comparison_id: int,
    db: Session = Depends(get_db),
):
    comp = db.query(Comparison).filter(Comparison.id == comparison_id).first()
    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")

    changes_models = [
        ChangeItemModel(
            id=ch.id,
            change_type=ch.change_type,
            section_label=ch.section_label,
            old_text=ch.old_text,
            new_text=ch.new_text,
            ai_comment=ch.ai_comment,
            ai_suggestion=ch.ai_suggestion,
            
        )
        for ch in comp.changes
    ]

    old_label = comp.version_old.version_label if comp.version_old else ""
    new_label = comp.version_new.version_label if comp.version_new else ""
    doc_name = comp.document.name if comp.document else ""

    return ComparisonDetailModel(
        id=comp.id,
        document_name=doc_name,
        version_old_label=old_label,
        version_new_label=new_label,
        created_at=comp.created_at.isoformat() if comp.created_at else "",
        overall_risk_level=comp.overall_risk_level,
        summary_text=comp.summary_text,
        changes=changes_models,
    )

# =========================
# GET /comparisons
# =========================
@app.get("/comparisons", response_model=List[ComparisonItemModel])
async def list_comparison_runs(limit: int = 50, db=Depends(get_db)):
    comps = (
        db.query(Comparison)
        .order_by(Comparison.created_at.desc())
        .limit(limit)
        .all()
    )

    items = []
    for c in comps:
        old_label = c.version_old.version_label if c.version_old else ""
        new_label = c.version_new.version_label if c.version_new else ""
        doc_name = c.document.name if c.document else ""

        items.append(
            ComparisonItemModel(
                id=c.id,
                document_name=doc_name,
                version_old_label=old_label,
                version_new_label=new_label,
                created_at=c.created_at.isoformat() if c.created_at else "",
                overall_risk_level=c.overall_risk_level,
                scope_impact_score=c.scope_impact_score,
                timeline_impact_score=c.timeline_impact_score,
                cost_impact_score=c.cost_impact_score,
                resource_impact_score=c.resource_impact_score,
                risk_impact_score=c.risk_impact_score,
                contract_impact_score=c.contract_impact_score,
                stakeholder_impact_score=c.stakeholder_impact_score,
                architecture_impact_score=c.architecture_impact_score,
            )
        )
    return items

# =========================
# DELETE /comparisons/{id}
# =========================
@app.delete("/comparisons/{comparison_id}")
def delete_comparison(comparison_id: int, db=Depends(get_db)):
    comp = (
        db.query(Comparison)
        .filter(Comparison.id == comparison_id)
        .first()
    )

    if not comp:
        raise HTTPException(status_code=404, detail="Comparison not found")

    document_id = comp.document_id

    delete_comparison_by_id(db, comparison_id)

    print(f"✅ DELETE comparison_id={comparison_id} success (document_id={document_id})")

    return {
        "success": True,
        "comparison_id": comparison_id,
        "deleted_document_id": document_id
    }

# ======================================================
# 🔹 THREAD-BASED WORKER (เหมือน service compare)
# ======================================================

def process_continue_job(
    job_id: str,
    document_id: int,
    v2_bytes: bytes,
    v2_label: str,
):
    try:
        continue_jobs[job_id]["status"] = JobStatus.running

        result = asyncio.run(
            run_compare_v2(
                document_id=document_id,
                v2_file_bytes=v2_bytes,
                v2_label=v2_label,
            )
        )

        continue_jobs[job_id]["status"] = JobStatus.done
        continue_jobs[job_id]["result"] = result

    except Exception as e:
        print(f"❌ Job {job_id} failed: {e}")
        continue_jobs[job_id]["status"] = JobStatus.error
        continue_jobs[job_id]["error"] = str(e)

# ======================================================
# 🔹 POST /compare/continue/start
# ======================================================

@app.post("/compare/continue/start")
async def compare_continue_start(
    document_id: int = Form(...),
    v2_label: str = Form("v2"),
    file_v2: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    latest_comp = (
        db.query(Comparison)
        .filter(Comparison.document_id == document_id)
        .order_by(Comparison.created_at.desc())
        .first()
    )

    if not latest_comp:
        raise HTTPException(
            status_code=400,
            detail=f"No comparison found for document_id={document_id}",
        )

    v2_bytes = await file_v2.read()
    if not v2_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    job_id = str(uuid.uuid4())

    continue_jobs[job_id] = {
        "status": JobStatus.pending,
        "result": None,
        "error": None,
    }

    threading.Thread(
        target=process_continue_job,
        args=(job_id, document_id, v2_bytes, v2_label),
        daemon=True,
    ).start()

    return {"job_id": job_id}

# ======================================================
# 🔹 GET /compare/continue/status/{job_id}
# ======================================================

@app.get("/compare/continue/status/{job_id}")
def get_continue_status(job_id: str):
    if job_id not in continue_jobs:
        return {"status": "not_found"}

    return {
        "job_id": job_id,
        "status": continue_jobs[job_id]["status"],
    }

# ======================================================
# 🔹 GET /compare/continue/result/{job_id}
# ======================================================

@app.get("/compare/continue/result/{job_id}", response_model=CompareResultModel)
def get_continue_result(job_id: str):
    if job_id not in continue_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = continue_jobs[job_id]

    if job["status"] in [JobStatus.pending, JobStatus.running]:
        raise HTTPException(status_code=202, detail="Job still processing")

    if job["status"] == JobStatus.error:
        raise HTTPException(status_code=500, detail=job["error"])

    r = job["result"]

    # === บังคับให้ตรงกับ Model เสมอ (กันพัง) ===
    safe_result = {
        "doc_name": r["doc_name"],
        "v1_label": r["v1_label"],
        "v2_label": r["v2_label"],
        "pages_v1": r["pages_v1"],
        "pages_v2": r["pages_v2"],
        "paragraphs_v1": r["paragraphs_v1"],
        "paragraphs_v2": r["paragraphs_v2"],
        "changes_count": r["changes_count"],
        "edit_intensity": r["edit_intensity"],

        "summary_text": r["summary_text"],
        "overall_risk_level": r["overall_risk_level"],
        "impact_scores": r["impact_scores"],
        "risk_comment": r["risk_comment"],

        "json_report_path": r["json_report_path"],
        "html_report_path": r["html_report_path"],
        "json_report_url": r["json_report_url"],
        "html_report_url": r["html_report_url"],

        "run_id": r["run_id"],
        "runtime_minutes": r["runtime_minutes"],
        "runtime_seconds": r["runtime_seconds"],
    }

    return safe_result