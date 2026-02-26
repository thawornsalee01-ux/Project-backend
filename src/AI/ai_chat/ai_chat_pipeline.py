import os
from typing import List, Generator, Tuple
import logging
import time
import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
    SystemMessage,
)

from src.AI.Tools.tavily_search import get_web_tools
from src.AI.Tools.get_change import get_change_paragraph

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
MODEL_TIMEOUT_SEC = float(os.getenv("LOCALMODEL_TIMEOUT_SEC", "90"))
TOOL_RESULT_MAX_CHARS = int(os.getenv("TOOL_RESULT_MAX_CHARS", "12000"))
INVOKE_HEARTBEAT_SEC = float(os.getenv("INVOKE_HEARTBEAT_SEC", "8"))
STREAM_CHUNK_TARGET = int(os.getenv("STREAM_CHUNK_TARGET", "32"))
STREAM_CHUNK_DELAY_SEC = float(os.getenv("STREAM_CHUNK_DELAY_SEC", "0.02"))
TRUE_TOKEN_STREAM = os.getenv("TRUE_TOKEN_STREAM", "false").strip().lower() in {"1", "true", "yes", "on"}

# ==========================================================
# CLEAN DEBUG (อ่านง่าย ไม่ยาว)
# ==========================================================
DEBUG_ENABLED = True
DEBUG_MAX_LEN = 300

def _truncate(value, max_len=DEBUG_MAX_LEN):
    if value is None:
        return None
    value = str(value)
    if len(value) <= max_len:
        return value
    return value[:max_len] + f"... [truncated {len(value)-max_len} chars]"

def debug(title, value=None):
    if not DEBUG_ENABLED:
        return

    print("\n" + "═" * 70)
    print(f"🧪 DEBUG :: {title}")
    print("─" * 70)

    if value is not None:
        try:
            if isinstance(value, (list, tuple)):
                print(f"type: {type(value).__name__} | len: {len(value)}")
                print(_truncate(value))
            elif isinstance(value, dict):
                print(f"type: dict | keys: {list(value.keys())}")
                print(_truncate(value))
            else:
                print(_truncate(value))
        except Exception as e:
            print("debug print error:", e)

    print("═" * 70 + "\n", flush=True)

# ==========================================================
# LLM
# ==========================================================
llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY"),
    model=os.getenv("LOCALMODEL_MODEL_SUM", "openai/gpt-oss-120b"),
    temperature=0.3,
    timeout=MODEL_TIMEOUT_SEC,
    max_retries=1,
)

# ==========================================================
# TOOLS
# ==========================================================
tools = [
    get_change_paragraph,
    *get_web_tools(),
]

llm_with_tools = llm.bind_tools(tools)

debug("REGISTERED TOOLS", [t.name for t in tools])
STATUS_PREFIX = "__STATUS__:"


def status_event(message: str) -> str:
    return f"{STATUS_PREFIX}{message}"


def _extract_question_and_change_id(messages: List) -> Tuple[str, int | None]:
    text = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = getattr(msg, "content", "")
            text = str(content) if content is not None else ""
            break

    m = re.search(r"change_id\s*:\s*(\d+)", text, flags=re.IGNORECASE)
    change_id = int(m.group(1)) if m else None

    if "คำถาม:" in text:
        question = text.split("คำถาม:", 1)[1].strip()
    else:
        question = text.strip()

    return question, change_id


def _needs_web_search(question: str) -> bool:
    q = question.lower()
    keywords = [
        "ล่าสุด", "ปัจจุบัน", "ตอนนี้", "วันนี้", "ปี ", "ราคา", "ข่าว", "อัปเดต",
        "latest", "current", "today", "news", "price", "update",
    ]
    return any(k in q for k in keywords)


def _needs_change_context(question: str) -> bool:
    q = question.lower()
    keywords = [
        "change", "เอกสาร", "สัญญา", "ข้อกำหนด", "paragraph", "section",
        "tor", "สเปค", "โครงการ", "งบประมาณ", "เปลี่ยนแปลง",
    ]
    return any(k in q for k in keywords)


def _invoke_with_heartbeat(messages: List, heartbeat_message: str) -> Generator[str, None, object | None]:
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as executor:
        fut = executor.submit(llm_with_tools.invoke, messages)
        while True:
            try:
                result = fut.result(timeout=INVOKE_HEARTBEAT_SEC)
                debug("MODEL INVOKE SEC", f"{time.monotonic() - started:.2f}")
                return result
            except FuturesTimeoutError:
                yield status_event(heartbeat_message)
            except Exception:
                raise

# ==========================================================
# TOOL EXECUTION LOOP
# ==========================================================
def execute_tools(
    messages: List, max_iterations: int = 8, max_web_search: int = 3
) -> Generator[str, None, Tuple[List, object | None]]:

    debug("EXECUTE TOOLS START")
    yield status_event("กำลังวิเคราะห์คำถาม...")
    question, change_id = _extract_question_and_change_id(messages)
    must_use_web = _needs_web_search(question)
    must_use_change = _needs_change_context(question) and change_id is not None
    used_tools: set[str] = set()
    forced_change_tool = False
    forced_web_tool = False

    web_search_count = 0

    for i in range(max_iterations):

        debug("ITERATION", i + 1)
        debug("MESSAGE COUNT", len(messages))
        yield status_event(f"กำลังประมวลผล...")

        try:
            invoke_gen = _invoke_with_heartbeat(messages, "กำลังประมวลผล...")
            while True:
                try:
                    heartbeat = next(invoke_gen)
                    if heartbeat:
                        yield heartbeat
                except StopIteration as stop:
                    response = stop.value
                    break
        except Exception as e:
            debug("MODEL INVOKE ERROR", str(e))
            yield status_event("เชื่อมต่อโมเดลไม่สำเร็จ")
            return messages, None

        debug("MODEL CONTENT PREVIEW", getattr(response, "content", None))
        debug("MODEL TOOL CALL COUNT", len(getattr(response, "tool_calls", []) or []))

        if not getattr(response, "tool_calls", None):
            if must_use_change and "get_change_paragraph" not in used_tools and not forced_change_tool:
                forced_change_tool = True
                yield status_event("กำลังดึงข้อมูลเอกสารจาก change ก่อนตอบ...")
                messages.append(
                    SystemMessage(
                        content=(
                            f"Before answering, you MUST call get_change_paragraph with change_id={change_id} "
                            "to ground your answer in document context."
                        )
                    )
                )
                continue

            if must_use_web and "tavily_search_results_json" not in used_tools and not forced_web_tool:
                forced_web_tool = True
                yield status_event("กำลังค้นข้อมูลล่าสุดจากเว็บก่อนตอบ...")
                messages.append(
                    SystemMessage(
                        content=(
                            "Before answering, you MUST call tavily_search_results_json once "
                            "with a focused query, then answer from the retrieved results."
                        )
                    )
                )
                continue

            debug("FINAL ANSWER REACHED")
            yield status_event("ได้คำตอบแล้ว กำลังจัดรูปแบบ...")
            return messages, response

        messages.append(response)

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            used_tools.add(tool_name)

            debug("RUN TOOL", tool_name)
            yield status_event(f"กำลังใช้เครื่องมือ {tool_name}...")

            if tool_name == "tavily_search_results_json":

                web_search_count += 1
                debug("WEB SEARCH COUNT", web_search_count)
                yield status_event(f"กำลังค้นข้อมูลเว็บ...")

                if web_search_count > max_web_search:

                    debug("WEB SEARCH LIMIT REACHED → FORCING ANSWER")

                    messages.append(
                        ToolMessage(
                            content=(
                                "Web search limit reached. "
                                "Provide the best possible final answer "
                                "using ONLY the information already retrieved. "
                                "Do NOT call any more tools."
                            ),
                            tool_call_id=tool_call["id"],
                        )
                    )

                    try:
                        invoke_gen = _invoke_with_heartbeat(messages, "กำลังประมวลผล...")
                        while True:
                            try:
                                heartbeat = next(invoke_gen)
                                if heartbeat:
                                    yield heartbeat
                            except StopIteration as stop:
                                final_response = stop.value
                                break
                    except Exception as e:
                        debug("FORCED INVOKE ERROR", str(e))
                        yield status_event("เชื่อมต่อโมเดลไม่สำเร็จ")
                        return messages, None

                    debug("FORCED RESPONSE CONTENT", final_response.content)

                    return messages, final_response

            selected_tool = next(
                (t for t in tools if t.name == tool_name),
                None
            )

            if not selected_tool:
                debug("TOOL NOT FOUND", tool_name)
                yield status_event(f"ไม่พบเครื่องมือ {tool_name}")

                messages.append(
                    ToolMessage(
                        content=f"[ERROR] tool not found: {tool_name}",
                        tool_call_id=tool_call["id"],
                    )
                )
                continue

            try:
                result = selected_tool.invoke(tool_args)
                debug("TOOL RESULT PREVIEW", str(result)[:300])
                yield status_event(f"ได้รับผลลัพธ์จาก {tool_name} แล้ว")
            except Exception as e:
                debug("TOOL ERROR", str(e))
                result = f"[TOOL ERROR] {e}"
                yield status_event(f"เครื่องมือ {tool_name} มีข้อผิดพลาด")

            # ✅ FIX (เฉพาะตรงนี้)
            if tool_name == "tavily_search_results_json" and isinstance(result, list):
                compact_rows = []
                for row in result[:5]:
                    if not isinstance(row, dict):
                        compact_rows.append({"text": str(row)[:400]})
                        continue
                    compact_rows.append(
                        {
                            "title": str(row.get("title", ""))[:200],
                            "url": str(row.get("url", ""))[:240],
                            "score": row.get("score"),
                            "content": str(row.get("content", ""))[:600],
                        }
                    )
                result_text = json.dumps(compact_rows, ensure_ascii=False, indent=2)
            elif isinstance(result, (list, dict)):
                result_text = json.dumps(result, ensure_ascii=False, indent=2)
            else:
                result_text = str(result)

            if len(result_text) > TOOL_RESULT_MAX_CHARS:
                debug("TOOL RESULT TRUNCATED", f"{len(result_text)} -> {TOOL_RESULT_MAX_CHARS}")
                result_text = result_text[:TOOL_RESULT_MAX_CHARS] + "\n...[truncated]"

            messages.append(
                ToolMessage(
                    content=result_text,
                    tool_call_id=tool_call["id"],
                )
            )

    debug("MAX ITERATIONS REACHED → FORCE FINAL GENERATION")

    messages.append(
        SystemMessage(
            content=(
                "Provide the final answer now using the information available. "
                "Do NOT call any tools."
            )
        )
    )

    try:
        invoke_gen = _invoke_with_heartbeat(messages, "กำลังประมวลผล...")
        while True:
            try:
                heartbeat = next(invoke_gen)
                if heartbeat:
                    yield heartbeat
            except StopIteration as stop:
                final_response = stop.value
                break
    except Exception as e:
        debug("FALLBACK INVOKE ERROR", str(e))
        yield status_event("เชื่อมต่อโมเดลไม่สำเร็จ")
        return messages, None

    debug("FALLBACK FINAL RESPONSE", final_response.content)
    yield status_event("ได้คำตอบแล้ว กำลังจัดรูปแบบ...")

    return messages, final_response



# ==========================================================
# GPT STYLE STREAMER (เพิ่มใหม่)
# ==========================================================
def gpt_style_stream(text: str, chunk_target=40, delay=0.1):
    """
    stream แบบ ChatGPT
    - ไม่ตัดคำ
    - รักษา markdown block
    - ไม่ทำ code block แตก
    """

    import re

    tokens = re.findall(r'\S+|\s+', text)

    buffer = ""
    in_code_block = False

    for token in tokens:

        if "```" in token:
            in_code_block = not in_code_block

        buffer += token

        limit = chunk_target * 2 if in_code_block else chunk_target

        if len(buffer) >= limit:
            yield buffer
            buffer = ""
            if delay:
                time.sleep(delay)

    if buffer:
        yield buffer


def _content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


# ==========================================================
# STREAM CHAT
# ==========================================================
def stream_ai_chat(change_id: int, user_message: str, db) -> Generator[str, None, None]:

    debug("STREAM CHAT START")

    conversation = get_or_create_conversation(db, change_id)
    debug("CONVERSATION ID", conversation.id)

    memory_messages = load_memory(db, conversation.id)
    debug("MEMORY COUNT", len(memory_messages))

    memory_context = ""
    remaining_messages = []

    for msg in memory_messages:
        if isinstance(msg, SystemMessage) and "สรุปบทสนทนา" in msg.content:
            memory_context += msg.content + "\n"
        else:
            remaining_messages.append(msg)

    debug("MEMORY CONTEXT PREVIEW", memory_context[:200])

    system_prompt = f"""
คุณคือ AI DOCUMENT COMPARE
ผู้เชี่ยวชาญ TOR / สัญญา / กฎหมายจัดซื้อจัดจ้าง

TOOLS:
- get_change_paragraph
- web_search

กฎ:
- ตอบตรงคำถาม
- ใช้อีโมจิแบบพอดี (1-3 ตัวต่อคำตอบ) เพื่อช่วยอ่านง่าย
- ใช้เฉพาะอีโมจิที่เกี่ยวข้องกับเนื้อหา เช่น ✅⚠️📌📊🔎
- ห้ามใช้อีโมจิในตารางทุกช่อง และห้ามใส่มากเกินจำเป็น



บริบทก่อนหน้า:
{memory_context if memory_context else "ไม่มี"}
"""

    messages = [SystemMessage(content=system_prompt)]
    messages.extend(remaining_messages)

    user_content = f"""
change_id: {change_id}

คำถาม:
{user_message}
"""
    messages.append(HumanMessage(content=user_content))

    debug("FINAL MESSAGE COUNT", len(messages))

    response = None
    resolved_messages = messages
    try:
        tool_exec = execute_tools(messages)
        while True:
            try:
                status = next(tool_exec)
                if status:
                    yield status
            except StopIteration as stop:
                resolved_messages, response = stop.value if stop.value else (messages, None)
                break
    except Exception as e:
        debug("EXECUTE TOOLS ERROR", str(e))
        yield f"[ERROR] AI service unavailable: {e}"
        return

    debug("FINAL MODEL RESPONSE TYPE", type(response))
    debug("FINAL CONTENT PREVIEW", getattr(response, "content", None))

    if not response:
        yield "[ERROR] AI service unavailable: no response from model"
        return

    if TRUE_TOKEN_STREAM:
        debug("TRUE TOKEN STREAM", True)
        yield status_event("กำลังสร้างคำตอบ")
        stream_messages = [
            *resolved_messages,
            SystemMessage(
                content=(
                    "Answer the user now. "
                    "Do not call any tools. "
                    "Use only the available context."
                )
            ),
        ]
        chunks: List[str] = []
        chunk_count = 0
        try:
            for chunk in llm.stream(stream_messages):
                text = _content_to_text(getattr(chunk, "content", None))
                if not text:
                    continue
                chunks.append(text)
                chunk_count += 1
                yield text
        except GeneratorExit:
            debug("TRUE STREAM INTERRUPTED", f"client disconnected after {chunk_count} chunks")
            return
        except Exception as e:
            debug("TRUE STREAM ERROR", str(e))
            yield status_event("สลับเป็นโหมดสตรีมสำรอง...")
            chunks = []

        final_answer = "".join(chunks).strip()
        debug("TRUE STREAM CHUNK COUNT", chunk_count)
        debug("TRUE STREAM FINAL LENGTH", len(final_answer))

        if final_answer:
            save_message(db, conversation.id, "user", user_content)
            save_message(db, conversation.id, "assistant", final_answer)
            auto_summarize_if_needed(
                db=db,
                conversation_id=conversation.id,
                llm=llm
            )
            return

    final_text = _content_to_text(getattr(response, "content", None)).strip()
    debug("FINAL TEXT LENGTH (PRE-STREAM)", len(final_text))

    if not final_text:
        yield "[ERROR] AI service unavailable: empty response from model"
        return

    debug("START STREAM FROM FINAL CONTENT")

    chunks: List[str] = []
    chunk_count = 0
    try:
        for piece in gpt_style_stream(
            final_text,
            chunk_target=STREAM_CHUNK_TARGET,
            delay=STREAM_CHUNK_DELAY_SEC,
        ):
            chunks.append(piece)
            chunk_count += 1
            yield piece
    except GeneratorExit:
        debug("STREAM INTERRUPTED", f"client disconnected after {chunk_count} chunks")
        return
    except Exception as e:
        debug("STREAM YIELD ERROR", str(e))
        yield f"[ERROR] stream interrupted: {e}"
        return

    debug("STREAM CHUNK COUNT", chunk_count)

    final_answer = "".join(chunks).strip()
    debug("FINAL ANSWER LENGTH", len(final_answer))

    if not final_answer:
        yield "[ERROR] AI service unavailable: failed to stream final answer"
        return

    save_message(db, conversation.id, "user", user_content)
    save_message(db, conversation.id, "assistant", final_answer)

    auto_summarize_if_needed(
        db=db,
        conversation_id=conversation.id,
        llm=llm
    )
