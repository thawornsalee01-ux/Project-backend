import json
import re
import os
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.diff.diff import Change
import asyncio

# === 🔥 IMPORT TOOLS ของคุณ ===
from src.AI.Tools.tavily_search import get_web_tools

load_dotenv()

# ==================================================
# LLM setup
# ==================================================
llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY"),
    model=os.getenv("LOCALMODEL_MODEL_SUGGESTION"),
    temperature=0.2,
)

# ==================================================
# 🔧 REGISTER TOOLS (Tavily)
# ==================================================
tools = get_web_tools()  # ได้ TavilySearchResults เป็น list

# ผูก tools กับ LLM
llm_with_tools = llm.bind_tools(tools)

# ==================================================
# Utility: Safe JSON Parser
# ==================================================
def _safe_parse_json(raw: str) -> dict:
    """พยายามแปลงข้อความจาก LLM ให้เป็น JSON ที่ถูกต้อง"""
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        print("⚠️ [DEBUG] ไม่พบ JSON ในข้อความ LLM")
        return {}
    try:
        return json.loads(match.group(0))
    except Exception as e:
        print(f"⚠️ [DEBUG] JSON parse error: {e}")
        cleaned = re.sub(r"[\x00-\x1f\x7f]", "", match.group(0))
        try:
            return json.loads(cleaned)
        except Exception as e2:
            print(f"⚠️ [DEBUG] Parse ล้มเหลวซ้ำ: {e2}")
            return {}

# ==================================================
# Generate ai_suggestion (🔥 ASYNC + TOOLS)
# ==================================================
async def generate_ai_suggestion(change: Change) -> None:
    prompt = ChatPromptTemplate.from_template("""
บริบทของการเปลี่ยนแปลง:
{ai_comment}

📡 TOOLS ที่คุณสามารถใช้ได้ (ถ้าจำเป็น):
- คุณสามารถค้นหาข้อมูลจากเว็บเพื่อช่วยวิเคราะห์ผลกระทบด้านกฎหมาย, ธุรกิจ หรือมาตรฐานอุตสาหกรรม
- ถ้าคุณต้องอ้างอิงนิยาม, มาตรฐาน หรือแนวปฏิบัติ ให้ใช้เครื่องมือค้นหาแทนการเดาเอง

หน้าที่ของคุณ:
- ถ้าการเปลี่ยนแปลงเป็นเพียงการแก้ไขการพิมพ์และการจัดรูปแบบของข้อความ, การสะกดคำ,
  การแก้ไขเป็นการปรับปรุงการสะกดคำและการพิมพ์เท่านั้น, มีการเปลี่ยนแปลงในส่วนนี้ กรุณาตรวจสอบรายละเอียดเพิ่มเติม
  ไม่ต้องแนะนำอะไรเด็ดขาดให้ตอบไปว่า
  "ไม่จำเป็นต้องดำเนินการใดๆ เพิ่มเติม"
  และไม่ต้องวิเคราะห์ผลกระทบให้จบการตอบได้เลย ***แต่ถ้ามีการเปลี่ยนแปลงอื่นๆ ร่วมด้วยต้องให้คำแนะนำตามด้านล่างนี้*** จะหยุดตอบได้ก็ต่อเมื่อมีการแก้ไขแค่เว้นวรรค การจัดรูปแบบ หรือการสะกดคำอย่างเดียวเท่านั้น

- ถ้าการเปลี่ยนแปลงมีผลกระทบต่อความหมายต้องให้คำแนะนำที่ชัดเจน
- ให้คำแนะนำในการจัดการกับความเสี่ยงที่เกิดขึ้นจากสิ่งทีเปลี่ยนแปลง 2 มุมมองอย่างชัดเจน ได้แก่
  1) มุมมองผู้ได้รับบริการหรือ ลูกค้า
  2) มุมมองผู้ให้บริการหรือ ผู้ขาย
  ใช้ภาษาทางการ กระชับ อธิบายเป็นข้อๆ , ไม่ต้องส่ง markdown อธิบายละเอียดได้แต่ให้อยู่ในขอบเขตที่เหมาะสม

ตอบเฉพาะในรูปแบบ JSON: {{ "ai_suggestion": "..." }}
""")

    chain = prompt | llm_with_tools | StrOutputParser()

    print(f"\n🛠️ [DEBUG] เริ่มวิเคราะห์ ai_suggestion สำหรับ Change: {change.change_type}")

    try:
        raw_output = await chain.ainvoke({
            "ai_comment": change.ai_comment,
            "legal": change.risk_scores.get("legal", 0) if hasattr(change, "risk_scores") else 0,
            "financial": change.risk_scores.get("financial", 0) if hasattr(change, "risk_scores") else 0,
            "operational": change.risk_scores.get("operational", 0) if hasattr(change, "risk_scores") else 0,
            "change_type": change.change_type,
            "old_text": change.old_text,
            "new_text": change.new_text,
        })

        raw_output = raw_output.strip()
        data = _safe_parse_json(raw_output)

        change.ai_suggestion = data.get(
            "ai_suggestion",
            "ควรตรวจสอบความเหมาะสมของการแก้ไขและติดตามผลกระทบที่อาจเกิดขึ้น"
        )

    except Exception as e:
        print(f"❌ [ERROR] ai_suggestion ล่ม: {e}")
        raise  # ให้ wrapper จัดการ retry

# ==================================================
# Async wrapper for generate_ai_suggestion + RETRY
# ==================================================


async def generate_ai_suggestion_async(
    change: Change,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2
):
    """
    Async wrapper สำหรับ generate_ai_suggestion
    - ใช้ semaphore ที่ถูกสร้างใน event loop เดียวกัน
    """

    async with semaphore:
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(
                        f"🔁 [RETRY {attempt}/{max_retries}] "
                        f"ai_suggestion | Change: {change.change_type}"
                    )

                await generate_ai_suggestion(change)
                return  # สำเร็จ → ออกฟังก์ชัน

            except Exception as e:
                last_error = e
                print(f"❌ [ASYNC ERROR] ai_suggestion attempt {attempt+1}: {e}")
                await asyncio.sleep(1.0)

        print(f"❌ [FATAL] ai_suggestion ล้มเหลวหลัง retry {max_retries} ครั้ง")

        # fallback (ปลอดภัยเสมอ)
        change.ai_suggestion = (
            "ควรตรวจสอบความเหมาะสมของการแก้ไขและติดตามผลกระทบที่อาจเกิดขึ้น "
            f"(ระบบลองใหม่หลายครั้งแล้ว แต่ยังเกิดข้อผิดพลาด)"
        )


# ==================================================
# Run ai_suggestion in parallel
# ==================================================
async def run_generate_ai_suggestion_parallel(changes: List[Change]):
    """
    ยิง generate_ai_suggestion พร้อมกันหลาย Change
    - สร้าง semaphore ภายใน event loop เดียวกัน (ป้องกัน error)
    """

    SEMAPHORE_LIMIT = int(os.getenv("LLM_PARALLEL_LIMIT", 8))
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)  # ✅ สร้างตรงนี้ (ถูกต้อง)

    results = await asyncio.gather(
        *[
            generate_ai_suggestion_async(c, semaphore, max_retries=2)
            for c in changes
        ],
        return_exceptions=True
    )

    # Debug: แสดงเฉพาะเคสที่ยัง error จริง ๆ
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"⚠️ [PARALLEL ERROR] ai_suggestion | Change index {i}: {r}")

