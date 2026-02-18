from src.db.models import ChatConversation, ChatMessage, ChatSummary
from src.AI.memory.sum_memory import summarize_conversation

def get_or_create_conversation(db, change_id: int):
    conv = (
        db.query(ChatConversation)
        .filter(ChatConversation.change_id == change_id)
        .first()
    )

    if not conv:
        conv = ChatConversation(change_id=change_id)
        db.add(conv)
        db.commit()
        db.refresh(conv)

    return conv

def save_message(db, conversation_id, role, content):
    msg = ChatMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(msg)
    db.commit()

def delete_old_messages(db, conversation_id: int, keep_last: int = 5):
    """
    ลบข้อความเก่าใน conversation
    เก็บไว้เฉพาะข้อความล่าสุด keep_last รายการ
    """

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at.desc())
        .all()
    )

    # ถ้ามีน้อยกว่า keep_last → ไม่ต้องลบ
    if len(messages) <= keep_last:
        return

    # แยกข้อความที่ต้องลบ
    to_delete = messages[keep_last:]

    for msg in to_delete:
        db.delete(msg)

    db.commit()

def save_summary(db, conversation_id, summary_text):

    existing = (
        db.query(ChatSummary)
        .filter(ChatSummary.conversation_id == conversation_id)
        .first()
    )

    if existing:
        existing.summary = summary_text
    else:
        new_summary = ChatSummary(
            conversation_id=conversation_id,
            summary=summary_text
        )
        db.add(new_summary)

    db.commit()

def auto_summarize_if_needed(db, conversation_id, llm):

    count = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .count()
    )

    if count < 15:
        return

    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.conversation_id == conversation_id)
        .order_by(ChatMessage.created_at)
        .all()
    )

    text = "\n".join([m.content for m in messages])

    summary = summarize_conversation(llm, text)

    save_summary(db, conversation_id, summary)

    # ลบ message เก่า (เหลือ 5 ล่าสุด)
    delete_old_messages(db, conversation_id)
