import logging
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel

from src.db.session import SessionLocal
from src.AI.ai_chat.ai_chat_pipeline import stream_ai_chat, STATUS_PREFIX

logger = logging.getLogger(__name__)

# ----------------- FastAPI Setup -----------------
app = FastAPI(
    title="AI Chat API",
    description="AI streaming chat endpoint for document change discussion"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

# ----------------- Request Model -----------------
class AIChatRequest(BaseModel):
    question: str

# ----------------- Health Check -----------------
@app.get("/health")
def health():
    return {"status": "ok", "service": "ai_chat_streaming"}

# ----------------- STREAMING CHAT -----------------
@app.post("/changes/{change_id}/chat")
def ai_chat(
    change_id: int,
    req: AIChatRequest,
    db: Session = Depends(get_db)
):
    try:

        def _to_sse_event(event_name: str, payload: str) -> str:
            normalized = payload.replace("\r\n", "\n").replace("\r", "\n")
            lines = normalized.split("\n")
            data_lines = "".join(f"data: {line}\n" for line in lines)
            return f"event: {event_name}\n{data_lines}\n"

        def generator():
            for token in stream_ai_chat(
                change_id=change_id,
                user_message=req.question,
                db=db
            ):
                if token.startswith(STATUS_PREFIX):
                    status_text = token[len(STATUS_PREFIX):].strip()
                    yield _to_sse_event("status", status_text)
                else:
                    yield _to_sse_event("chunk", token)

        return StreamingResponse(
            generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.exception("AI Streaming error")
        raise HTTPException(status_code=500, detail=str(e))
