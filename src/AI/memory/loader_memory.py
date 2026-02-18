from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.db.models import ChatMessage, ChatSummary


def load_memory(db, conversation_id, recent_limit: int = 5):

    # ------------------------------------------------------
    # 1️⃣ โหลด summary
    # ------------------------------------------------------
    summary = (
        db.query(ChatSummary)
        .filter(ChatSummary.conversation_id == conversation_id)
        .first()
    )

    # ------------------------------------------------------
    # 2️⃣ โหลด recent messages เท่านั้น
    # ------------------------------------------------------
    recent_messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(recent_limit)
        .all()
    )

    lc_messages = []

    # ------------------------------------------------------
    # 3️⃣ ใส่ summary เป็น context
    # ------------------------------------------------------
    if summary and summary.summary:
        lc_messages.append(
            SystemMessage(
                content=f"""
สรุปบทสนทนาก่อนหน้า (ใช้เป็นบริบทต่อเนื่อง):

{summary.summary}
"""
            )
        )

    # ------------------------------------------------------
    # 4️⃣ ใส่ recent messages (เรียงเวลาเก่า → ใหม่)
    # ------------------------------------------------------
    for m in reversed(recent_messages):
        if m.role == "user":
            lc_messages.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            lc_messages.append(AIMessage(content=m.content))
        elif m.role == "system":
            lc_messages.append(SystemMessage(content=m.content))

    return lc_messages
