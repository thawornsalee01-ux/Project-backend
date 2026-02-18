import os
import json
import re
from collections import Counter
from typing import List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.diff.diff import Change
from src.AI.ai_comment import generate_ai_comment
from src.AI.ai_suggestion import generate_ai_suggestion

# ==================================================
# Load environment variables
# ==================================================
load_dotenv()

# ✅ ตั้งค่าโมเดล LangChain LLM
llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY"),
    model=os.getenv("LOCALMODEL_MODEL_SUM"),
    temperature=0.2,
)

# ==================================================
# Utility: Safe JSON Parser
# ==================================================
def _safe_parse_json(raw: str) -> dict:
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
# 🔥 SAFE FLOAT (กันกรณี LLM ส่ง "moderate", "mode erate", "4 (สูง)" ฯลฯ)
# ==================================================
def _safe_float(val) -> float:
    if val is None:
        return 0.0

    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, str):
        # ลองดึงเฉพาะตัวเลขจากสตริง
        nums = re.findall(r"-?\d+\.?\d*", val)
        if nums:
            try:
                return float(nums[0])
            except:
                return 0.0

    return 0.0

# ==================================================
def _add_ai_comments(changes: List[Change]) -> None:
    for idx, c in enumerate(changes, 1):
        print(f"\n🚀 [DEBUG] กำลังประมวลผล Paragraph {idx}/{len(changes)} ({c.change_type})")

        related_comments = [
            other.ai_comment for other in changes
            if other != c and getattr(other, "ai_comment", None)
        ]

        if not getattr(c, "ai_comment", None):
            generate_ai_comment(c, related_comments)

        if not getattr(c, "ai_suggestion", None):
            generate_ai_suggestion(c)

        print(f"✅ [DEBUG] เสร็จสิ้น Paragraph {idx}")

# ==================================================
def build_summary_text(changes: List[Change]) -> dict:
    if not changes:
        return {
            "summary_text": "ไม่มีการเปลี่ยนแปลงเนื้อหาสำคัญระหว่างสองเวอร์ชัน",
            "impact_scores": {
                "scope_impact_score": 0,
                "timeline_impact_score": 0,
                "cost_impact_score": 0,
                "resource_impact_score": 0,
                "risk_impact_score": 0,
                "contract_impact_score": 0,
                "stakeholder_impact_score": 0,
                "architecture_impact_score": 0,
            },
            "risk_comment": "ไม่มีความเสี่ยงเนื่องจากไม่มีการเปลี่ยนแปลง",
            "overall_risk_level": "LOW",   # 🔥 รับประกันว่ามี
        }

    # ===============================
    # STEP 1 — สรุปข้อความรวม
    # ===============================
    _add_ai_comments(changes)

    type_counter = Counter(c.change_type for c in changes)
    total = len(changes)

    base_summary = (
        f"โดยรวมมีการเปลี่ยนแปลงจำนวน {total} รายการ "
        f"(เพิ่ม {type_counter.get('ADDED', 0)} รายการ, "
        f"ลบ {type_counter.get('REMOVED', 0)} รายการ, "
        f"แก้ไข {type_counter.get('MODIFIED', 0)} รายการ)"
    )

    all_ai_comments = "\n".join(
        f"- Page {c.section_label}: {getattr(c, 'ai_comment', 'ไม่มี AI Comment')}"
        for c in changes
    )

    all_ai_suggestions = "\n".join(
        f"- Page {c.section_label}: {getattr(c, 'ai_suggestion', 'ไม่มี AI Suggestion')}"
        for c in changes
    )

    summary_prompt = ChatPromptTemplate.from_template("""
ข้อมูลสรุปเชิงปริมาณ:
{base_summary}

ความเห็น AI แยกตามหน้าเอกสาร:
{all_ai_comments}

ข้อเสนอแนะ AI แยกตามหน้าเอกสาร:
{all_ai_suggestions}

กรุณาจัดทำสรุปภาพรวม 2 ส่วน ดังนี้:

ส่วนที่ 1: สรุปภาพรวมการเปลี่ยนแปลง (3–5 บรรทัด)
- อธิบายเป็นข้อ ๆ
- ใช้ภาษาทางการ กระชับ
- ไม่เปลี่ยนข้อเท็จจริง
- อ้างอิงประเด็นสำคัญตาม "หน้า (Page)" ที่ระบุไว้

ส่วนที่ 2: สรุปข้อเสนอแนะภาพรวม (2 มุมมอง)
ให้สรุปเป็นข้อ ๆ แยกชัดเจน:

1) มุมมองผู้ได้รับบริการ (ลูกค้า)
2) มุมมองผู้ให้บริการ (ผู้ขาย)
                                                      
ส่วนที่ 3: วิเคราะห์ผู้ที่มีส่วนได้ส่วนเสียจากการเปลี่ยนแปลงนี้ระหว่างลูกค้าและผู้ขาย
- ระบุเป็นข้อ ๆ ว่าใครจะได้รับผลกระทบอย่างไรบ้าง

ตอบด้วยภาษาทางการ กระชับ เป็นข้อ ๆ
ไม่ต้องใส่markdown 
หากเป็นเพียงการแก้ไขเล็กน้อย ให้สรุปเพียง 1 บรรทัด
""")

    summary_chain = summary_prompt | llm | StrOutputParser()

    try:
        raw_summary = summary_chain.invoke({
            "base_summary": base_summary,
            "all_ai_comments": all_ai_comments,
            "all_ai_suggestions": all_ai_suggestions
        }).strip()

        full_summary_text = f"{base_summary}\n\n{raw_summary}" if raw_summary else base_summary

    except Exception as e:
        print(f"⚠️ [DEBUG] summary ล่ม: {e}")
        full_summary_text = base_summary

    # ===============================
    # STEP 2 — Impact Scoring
    # ===============================

    impact_prompt = ChatPromptTemplate.from_template("""
ข้อความสรุปทั้งเอกสาร:
{summary_text}

ให้คุณประเมินผลกระทบเป็นคะแนน 0-100 (ตัวเลข) และให้ "overall_risk_level" เป็น LOW | MEDIUM | HIGH

ตอบเป็น JSON เท่านั้น ตามโครงสร้างนี้:

{{
  "impact_scores": {{
    "scope_impact_score": 0,
    "timeline_impact_score": 0,
    "cost_impact_score": 0,
    "resource_impact_score": 0,
    "risk_impact_score": 0,
    "contract_impact_score": 0,
    "stakeholder_impact_score": 0,
    "architecture_impact_score": 0
  }},
  "risk_comment": "คำอธิบายความเสี่ยงเชิงข้อความ (ห้ามใส่ตัวเลข)",
  "overall_risk_level": "LOW | MEDIUM | HIGH"
}}
""")

    impact_chain = impact_prompt | llm | StrOutputParser()

    try:
        raw_risk = impact_chain.invoke({
            "summary_text": full_summary_text
        }).strip()

        data = _safe_parse_json(raw_risk)

        return {
            "summary_text": full_summary_text,
            "impact_scores": {
                "scope_impact_score": _safe_float(data.get("impact_scores", {}).get("scope_impact_score", 0)),
                "timeline_impact_score": _safe_float(data.get("impact_scores", {}).get("timeline_impact_score", 0)),
                "cost_impact_score": _safe_float(data.get("impact_scores", {}).get("cost_impact_score", 0)),
                "resource_impact_score": _safe_float(data.get("impact_scores", {}).get("resource_impact_score", 0)),
                "risk_impact_score": _safe_float(data.get("impact_scores", {}).get("risk_impact_score", 0)),
                "contract_impact_score": _safe_float(data.get("impact_scores", {}).get("contract_impact_score", 0)),
                "stakeholder_impact_score": _safe_float(data.get("impact_scores", {}).get("stakeholder_impact_score", 0)),
                "architecture_impact_score": _safe_float(data.get("impact_scores", {}).get("architecture_impact_score", 0)),
            },
            "risk_comment": data.get(
                "risk_comment",
                "ไม่พบความเสี่ยงที่มีนัยสำคัญจากภาพรวมการเปลี่ยนแปลง"
            ),
            "overall_risk_level": str(data.get("overall_risk_level", "LOW")).upper(),
        }

    except Exception as e:
        print(f"⚠️ [DEBUG] impact scoring ล่ม: {e}")

        return {
            "summary_text": full_summary_text,
            "impact_scores": {
                "scope_impact_score": 0,
                "timeline_impact_score": 0,
                "cost_impact_score": 0,
                "resource_impact_score": 0,
                "risk_impact_score": 0,
                "contract_impact_score": 0,
                "stakeholder_impact_score": 0,
                "architecture_impact_score": 0,
            },
            "risk_comment": "ระบบไม่สามารถประเมินผลกระทบได้",
            "overall_risk_level": "LOW",   
        }
