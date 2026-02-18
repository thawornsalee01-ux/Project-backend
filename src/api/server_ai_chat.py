import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
from src.db.session import SessionLocal
from src.AI.ai_chat.ai_chat_pipeline import run_ai_chat
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="AI Chat API", description="AI chat endpoint for document change discussion")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ปรับตาม security policy
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- DB Dependency -----------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------- Pydantic Models -----------------
class AIChatRequest(BaseModel):
    question: str

class AIChatResponse(BaseModel):
    answer: str

# ----------------- Health Check -----------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "ai_chat_api"}

# ----------------- AI Chat Endpoint -----------------
@app.post("/changes/{change_id}/chat", response_model=AIChatResponse)
def ai_chat(
    change_id: int,
    req: AIChatRequest,
    db: Session = Depends(get_db)
):
    try:
        answer = run_ai_chat(
            change_id=change_id,
            user_message=req.question,
            db=db
        )
    except Exception as e:
        logger.exception("AI Chat error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    return AIChatResponse(answer=answer)
