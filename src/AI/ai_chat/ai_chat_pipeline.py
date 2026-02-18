import os
from typing import List
import logging

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from src.AI.Tools.tavily_search import get_web_tools
from src.AI.Tools.get_change import get_change_paragraph

# memory
from src.AI.memory.loader_memory import load_memory
from src.AI.memory.service_memory import (
    get_or_create_conversation,
    save_message,
    auto_summarize_if_needed,
)


# ==========================================================
# CONFIG
# ==========================================================
load_dotenv()
logger = logging.getLogger(__name__)


# ==========================================================
# LLM
# ==========================================================
llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY"),
    model=os.getenv("LOCALMODEL_MODEL_SUM", "openai/gpt-oss-120b"),
    temperature=0.3,
)


# ==========================================================
# TOOLS
# ==========================================================
tools = [
    get_change_paragraph,
    *get_web_tools(),
]

llm_with_tools = llm.bind_tools(tools)


# ==========================================================
# TOOL EXECUTION LOOP
# ==========================================================
def execute_tools(messages: List, max_iterations: int = 5):

    logger.info("=== TOOL EXECUTION START ===")

    for i in range(max_iterations):

        logger.info(f"TOOL ITERATION {i+1}")

        response = llm_with_tools.invoke(messages)

        if not response.tool_calls:
            logger.info("NO TOOL CALL → FINAL ANSWER")
            return response

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logger.info(f"CALL TOOL → {tool_name}")

            selected_tool = next(
                (t for t in tools if t.name == tool_name),
                None
            )

            if not selected_tool:
                messages.append(
                    ToolMessage(
                        content=f"[ERROR] ไม่พบ tool: {tool_name}",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            try:
                result = selected_tool.invoke(tool_args)
            except Exception as e:
                logger.exception("TOOL ERROR")
                result = f"[TOOL ERROR] {e}"

            messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"],
                )
            )

    return AIMessage(content="⚠️ เกินจำนวน tool iteration ที่กำหนด")


# ==========================================================
# MAIN CHAT FUNCTION
# ==========================================================
def run_ai_chat(change_id: int, user_message: str, db) -> str:

    logger.info("=== RUN AI CHAT ===")

    # ------------------------------------------------------
    # 1️⃣ conversation
    # ------------------------------------------------------
    conversation = get_or_create_conversation(db, change_id)
    logger.info(f"CONVERSATION ID = {conversation.id}")

    # ------------------------------------------------------
    # 2️⃣ load memory
    # ------------------------------------------------------
    memory_messages = load_memory(db, conversation.id)
    logger.info(f"LOADED MEMORY MSG COUNT = {len(memory_messages)}")

    # ------------------------------------------------------
    # 3️⃣ extract summary memory
    # ------------------------------------------------------
    memory_context = ""
    remaining_messages = []

    for msg in memory_messages:
        if isinstance(msg, SystemMessage) and "สรุปบทสนทนา" in msg.content:
            memory_context += msg.content + "\n"
        else:
            remaining_messages.append(msg)

    # ------------------------------------------------------
    # 4️⃣ build SYSTEM PROMPT
    # ------------------------------------------------------
    system_prompt = f"""
คุณคือ AI DOCUMENT COMPARE
เป็นเพศหญิงผู้เชี่ยวชาญด้าน TOR / สัญญา / กฎหมายจัดซื้อจัดจ้าง

TOOLS:
- get_change_paragraph
- web_search

กฎ:
- ตอบตรงคำถามเท่านั้น
- ห้ามเดา
- ห้ามระบุ change_id เด็ดขาด
- ถ้าไม่รู้ให้บอกว่าไม่มีข้อมูล
- ให้ใส่อิโมจิในคำตอบเสมอเพิ่มเพิ่มความเป็นมิตรและเข้าใจง่าย

บริบทการสนทนาก่อนหน้า:
{memory_context if memory_context else "ไม่มี"}
"""

    messages = [SystemMessage(content=system_prompt)]

    # ------------------------------------------------------
    # 5️⃣ add recent conversation
    # ------------------------------------------------------
    messages.extend(remaining_messages)

    # ------------------------------------------------------
    # 6️⃣ user message
    # ------------------------------------------------------
    user_content = f"""
change_id: {change_id}

คำถาม:
{user_message}
"""

    messages.append(HumanMessage(content=user_content))

    # ------------------------------------------------------
    # 7️⃣ reasoning + tools
    # ------------------------------------------------------
    logger.info("START MODEL REASONING")
    response = execute_tools(messages)

    if not response.content:
        return "⚠️ Model ไม่ส่งข้อความกลับมา"

    final_answer = response.content.strip()

    # ------------------------------------------------------
    # 8️⃣ SAVE MEMORY
    # ------------------------------------------------------
    save_message(db, conversation.id, "user", user_content)
    save_message(db, conversation.id, "assistant", final_answer)

    # ------------------------------------------------------
    # 9️⃣ AUTO SUMMARY
    # ------------------------------------------------------
    auto_summarize_if_needed(
        db=db,
        conversation_id=conversation.id,
        llm=llm
    )

    logger.info("=== CHAT COMPLETE ===")

    return final_answer
