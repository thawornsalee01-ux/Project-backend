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

load_dotenv()

# ==================================================
# LLM setup
# ==================================================
llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY"),
    model=os.getenv("LOCALMODEL_MODEL_COMMENT"),
    temperature=0.2,
)

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
# Generate ai_comment (ASYNC) — ตัด risk ออกแล้ว
# ==================================================
async def generate_ai_comment(change: Change) -> None:
    """
    วิเคราะห์ change แบบ independent (async)
    เวอร์ชันนี้: เก็บเฉพาะ ai_comment เท่านั้น (ไม่มี risk)
    """

    prompt = ChatPromptTemplate.from_template(""" 
คุณคือผู้เชี่ยวชาญด้าน **กฎหมาย การเงิน และการบริหารโครงการ**

หน้าที่ของคุณคือ:
- ระบุสิ่งที่เปลี่ยนแปลงอย่างละเอียดระหว่าง ข้อความเดิม และ ข้อความใหม่ 
   ***เน้นย้ำว่าต้องละเอียดตอบยาวได้ตามความเหมาะสมแต่ไม่เกิน800token***
- หากมีการแก้ไขเพียงเล็กน้อย เช่น การแก้ไขการพิมพ์ การจัดรูปแบบ หรือการสะกดคำ หรือทุกอย่างที่อาจเกิดความผิดพลาดจากการพิมพ์
   ไม่ต้องระบุรายละเอียดของการสะกดคำหรือการพิมพ์ 
   แต่ให้ระบุว่า "ไม่พบความเสี่ยงที่มีนัยสำคัญจากการเปลี่ยนแปลงนี้"
- หากการแก้ไขมีการลบหรือเพิ่มข้อมูลหรือเนื้อหาสาระเปลี่ยนแปลง
   ควรระบุรายละเอียดให้ครบถ้วน

เงื่อนไขพิเศษ:
- ห้ามระบุชื่อตัวแปรลงใน ai_comment
- ห้ามมีmarkdown หรือการเน้นข้อความเช่น **ข้อความ**
----------------------------------------
ประเภทการเปลี่ยนแปลง:
{change_type}

ข้อความเดิม:
{old_text}

ข้อความใหม่:
{new_text}
----------------------------------------

ตอบเฉพาะในรูปแบบ JSON เท่านั้น เช่น:
{{
  "ai_comment": "..."
}}
""")

    chain = prompt | llm | StrOutputParser()

    print(f"\n🧠 [DEBUG] วิเคราะห์ ai_comment สำหรับ Change: {change.change_type}")

    try:
        raw_output = await chain.ainvoke({
            "change_type": change.change_type,
            "old_text": change.old_text or "-",
            "new_text": change.new_text or "-",
        })

        raw_output = raw_output.strip()
        data = _safe_parse_json(raw_output)

        change.ai_comment = data.get(
            "ai_comment",
            "มีการเปลี่ยนแปลงในส่วนนี้ กรุณาตรวจสอบรายละเอียดเพิ่มเติม"
        )

    except Exception as e:
        print(f"❌ [ERROR] วิเคราะห์ ai_comment ล่ม: {e}")
        # โยน error ให้ wrapper ทำ retry
        raise

# ==================================================
# Async wrapper (ห่อ generate_ai_comment) + RETRY
# =================================================

async def generate_ai_comment_async(
    change: Change, 
    semaphore: asyncio.Semaphore,
    max_retries: int = 2
):
    """
    Async wrapper สำหรับ generate_ai_comment
    - ใช้ semaphore ที่ถูกสร้างใน event loop เดียวกัน
    """

    async with semaphore:
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔁 [RETRY {attempt}/{max_retries}] Change: {change.change_type}")

                await generate_ai_comment(change)
                return  # สำเร็จ → ออก

            except Exception as e:
                last_error = e
                print(f"❌ [ASYNC ERROR] attempt {attempt+1}: {e}")
                await asyncio.sleep(1.0)

        print(f"❌ [FATAL] วิเคราะห์ Change ไม่สำเร็จหลัง retry {max_retries} ครั้ง")
        change.ai_comment = (
            "มีการเปลี่ยนแปลงในส่วนนี้ "
            "แต่ระบบไม่สามารถวิเคราะห์ได้หลังจากลองหลายครั้ง"
        )


# ==================================================
# Run ai_comment in parallel (เรียกจากไฟล์นี้)
# ==================================================
async def run_generate_ai_comment_parallel(changes: List[Change]):
    """
    ยิง generate_ai_comment พร้อมกันหลาย Change
    - สร้าง semaphore ภายใน event loop เดียวกัน (ป้องกัน error)
    """

    SEMAPHORE_LIMIT = int(os.getenv("LLM_PARALLEL_LIMIT", 8))
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)  # ✅ สร้างตรงนี้ (ถูกต้อง)

    results = await asyncio.gather(
        *[
            generate_ai_comment_async(c, semaphore, max_retries=2)
            for c in changes
        ],
        return_exceptions=True
    )

    # Debug: แสดงเฉพาะเคสที่ยัง error จริง ๆ
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"⚠️ [PARALLEL ERROR] Change index {i}: {r}")

