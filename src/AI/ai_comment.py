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
# VALID CHANGE CATEGORY
# ==================================================
VALID_CATEGORIES = {
    "scope",
    "timeline",
    "cost",
    "resource",
    "risk",
    "contract",
    "stakeholder",
    "architecture"
}

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
# Generate FULL AI Analysis (ASYNC)
# ==================================================
async def generate_ai_comment(change: Change) -> None:
    """
    วิเคราะห์ change แบบ full analysis:
    - paragraph_topic
    - change_category
    - ai_comment
    - change_details ⭐ NEW
    """

    prompt = ChatPromptTemplate.from_template("""
คุณคือผู้เชี่ยวชาญด้าน กฎหมาย การเงิน และการบริหารโครงการ

งานของคุณมี 4 ส่วน:

========================================
1) สรุปหัวข้อหลักของข้อความใหม่ (paragraph_topic)
========================================

========================================
2) จัดประเภทการเปลี่ยนแปลง (change_category)
========================================

IMPORTANT:
ต้องเลือกเพียง 1 หมวดเท่านั้น
ต้องเลือกจากนิยามด้านล่างอย่างเคร่งครัด
ห้ามตีความเกินนิยาม
ห้ามเดา
ถ้าเข้าได้หลายหมวด ให้ใช้ PRIORITY RULE


====================================================
DEFINITION OF CHANGE CATEGORIES (STRICT RULE)
====================================================

1️⃣ scope
ความหมาย:
การเปลี่ยนขอบเขตของงานหรือสิ่งที่ต้องส่งมอบ

รวมถึง:
- จำนวนงาน
- ปริมาณสินค้า
- รายการ deliverable
- feature ที่ต้องทำ
- สิ่งที่รวม / ไม่รวมในโครงการ
- requirement specification

ตัวอย่าง:
เพิ่ม feature ใหม่
ลดจำนวนเครื่อง
เพิ่ม module
ตัดงานบางส่วนออก

ไม่รวม:
ระยะเวลา → timeline
งบประมาณ → cost


----------------------------------

2️⃣ timeline
ความหมาย:
การเปลี่ยนแปลงกำหนดเวลา หรือ schedule

รวมถึง:
- deadline
- milestone
- duration
- phase
- implementation period
- delivery schedule

ตัวอย่าง:
เลื่อนกำหนดส่ง
เพิ่มระยะเวลาดำเนินงาน
เปลี่ยน milestone

ไม่รวม:
ขอบเขตงาน → scope


----------------------------------

3️⃣ cost
ความหมาย:
การเปลี่ยนแปลงด้านการเงินหรือค่าใช้จ่าย

รวมถึง:
- budget
- pricing
- payment
- financial term
- funding
- cost allocation
- penalty ที่เป็นเงิน

ตัวอย่าง:
เพิ่มงบ
ลดราคา
เปลี่ยน payment term


----------------------------------

4️⃣ resource
ความหมาย:
การเปลี่ยนแปลงทรัพยากรที่ใช้ทำงาน

รวมถึง:
- manpower
- staffing
- team structure
- skill requirement
- equipment
- tools

ตัวอย่าง:
เพิ่มคน
ลดทีม
เปลี่ยน role
ใช้เครื่องมือใหม่


----------------------------------

5️⃣ risk
ความหมาย:
การเปลี่ยนแปลงที่ส่งผลต่อระดับความเสี่ยงโดยตรง

รวมถึง:
- uncertainty
- exposure
- mitigation
- contingency
- risk allocation
- risk responsibility

ตัวอย่าง:
เพิ่มเงื่อนไขเสี่ยง
เพิ่มข้อจำกัด
เพิ่ม dependency


----------------------------------

6️⃣ contract
ความหมาย:
การเปลี่ยนข้อกำหนดทางกฎหมายหรือสัญญา

รวมถึง:
- liability
- obligation
- SLA
- warranty
- termination clause
- penalty (เชิงกฎหมาย)
- legal responsibility

ตัวอย่าง:
เพิ่มข้อผูกพัน
เปลี่ยนเงื่อนไขสัญญา


----------------------------------

7️⃣ stakeholder
ความหมาย:
การเปลี่ยนแปลงผู้มีส่วนได้ส่วนเสียหรือบทบาทของพวกเขา

รวมถึง:
- authority
- responsibility
- governance
- approval role
- reporting line
- communication structure

ตัวอย่าง:
เพิ่มหน่วยงานใหม่
เปลี่ยนผู้อนุมัติ


----------------------------------

8️⃣ architecture
ความหมาย:
การเปลี่ยนแปลงโครงสร้างระบบหรือการออกแบบเทคนิค

รวมถึง:
- system design
- technical structure
- platform
- integration
- infrastructure
- data architecture

ตัวอย่าง:
เปลี่ยนระบบ backend
เปลี่ยน platform
เปลี่ยน integration


====================================================
PRIORITY RULE (ถ้าเข้าได้หลายหมวด)
====================================================

contract > cost > scope > timeline > resource > architecture > stakeholder > risk

เลือกตัวที่ priority สูงกว่า


====================================================
3) วิเคราะห์การเปลี่ยนแปลงอย่างละเอียด (ai_comment)
====================================================
- ตอบเป็นภาษาไทย
- ถ้าเป็นการแก้ไขเว้นวรรคหรือสระ, การสะกด ให้ตอบสั้นๆ

====================================================
4) ระบุรายการสิ่งที่เปลี่ยนแปลงจริง (change_details)
====================================================

กฎ:
- ต้องเป็น list JSON
- factual เท่านั้น
- added removed modified
- ถ้าไม่มีสาระสำคัญ → []
- ให้ตอบเป็นภาษาไทย


format:

[
  {{
    "type": "added | removed | modified",
    "description": "..."
  }}
]


----------------------------------------
ประเภทการเปลี่ยนแปลง:
{change_type}

ข้อความเดิม:
{old_text}

ข้อความใหม่:
{new_text}
----------------------------------------


====================================================
OUTPUT FORMAT (JSON ONLY)
====================================================

{{
  "paragraph_topic": "...",
  "change_category": "...",
  "ai_comment": "...",
  "change_details": []
}}

ห้ามข้อความอื่น
ห้าม markdown
ห้ามอธิบายนอก JSON
""")

    chain = prompt | llm | StrOutputParser()

    print(f"\n🧠 [DEBUG] วิเคราะห์ FULL สำหรับ Change: {change.change_type}")

    try:
        raw_output = await chain.ainvoke({
            "change_type": change.change_type,
            "old_text": change.old_text or "-",
            "new_text": change.new_text or "-",
        })

        raw_output = raw_output.strip()
        data = _safe_parse_json(raw_output)

        # =========================
        # ai_comment
        # =========================
        change.ai_comment = data.get(
            "ai_comment",
            "มีการเปลี่ยนแปลงในส่วนนี้ กรุณาตรวจสอบรายละเอียดเพิ่มเติม"
        )

        # =========================
        # paragraph_topic
        # =========================
        change.paragraph_topic = data.get(
            "paragraph_topic",
            "ไม่สามารถสรุปหัวข้อได้"
        )

        # =========================
        # change_category
        # =========================
        cat = str(data.get("change_category", "")).lower().strip()
        if cat not in VALID_CATEGORIES:
            print(f"⚠️ [DEBUG] invalid category: {cat}")
            cat = "unknown"
        change.change_category = cat

        # =========================
        # ⭐ NEW change_details
        # =========================
        change.change_details = data.get("change_details", [])

    except Exception as e:
        print(f"❌ [ERROR] วิเคราะห์ FULL ล่ม: {e}")
        raise


# ==================================================
# Async wrapper + RETRY
# ==================================================
async def generate_ai_comment_async(
    change: Change,
    semaphore: asyncio.Semaphore,
    max_retries: int = 2
):

    async with semaphore:
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"🔁 [RETRY {attempt}/{max_retries}] Change: {change.change_type}")

                await generate_ai_comment(change)
                return

            except Exception as e:
                last_error = e
                print(f"❌ [ASYNC ERROR] attempt {attempt+1}: {e}")
                await asyncio.sleep(1.0)

        print(f"❌ [FATAL] วิเคราะห์ Change ไม่สำเร็จหลัง retry {max_retries} ครั้ง")

        change.ai_comment = (
            "มีการเปลี่ยนแปลงในส่วนนี้ "
            "แต่ระบบไม่สามารถวิเคราะห์ได้หลังจากลองหลายครั้ง"
        )
        change.paragraph_topic = "analysis_failed"
        change.change_category = "unknown"
        change.change_details = []


# ==================================================
# Run parallel
# ==================================================
async def run_generate_ai_comment_parallel(changes: List[Change]):

    SEMAPHORE_LIMIT = int(os.getenv("LLM_PARALLEL_LIMIT", 8))
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    results = await asyncio.gather(
        *[
            generate_ai_comment_async(c, semaphore, max_retries=2)
            for c in changes
        ],
        return_exceptions=True
    )

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"⚠️ [PARALLEL ERROR] Change index {i}: {r}")