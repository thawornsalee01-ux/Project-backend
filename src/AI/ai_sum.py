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
# 🔥 SAFE FLOAT
# ==================================================
def _safe_float(val) -> float:
    if val is None:
        return 0.0

    if isinstance(val, (int, float)):
        return float(val)

    if isinstance(val, str):
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
            "overall_risk_level": "LOW",
        }

    # ===============================
    # STEP 1 — Summary (เหมือนเดิม)
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

ส่วนที่ 2: สรุปข้อเสนอแนะภาพรวม (2 มุมมอง)

ส่วนที่ 3: วิเคราะห์ผู้ที่มีส่วนได้ส่วนเสีย

ตอบด้วยภาษาทางการ กระชับ เป็นข้อ ๆ
ห้าม markdown
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
    # STEP 2 — Impact Scoring (⭐ แก้ใหม่)
    # ===============================

    structured_analysis = "\n".join(
        f"""
Paragraph: {c.section_label}
Topic: {getattr(c, "paragraph_topic", "")}
Change Category: {getattr(c, "change_category", "")}
AI Analysis: {getattr(c, "ai_comment", "")}
Change Details: {getattr(c, "change_details", [])}
"""
        for c in changes
    )

    impact_prompt = ChatPromptTemplate.from_template("""
คุณคือผู้เชี่ยวชาญด้านการประเมินความเสี่ยงโครงการระดับองค์กร (Enterprise Project Risk Assessor)

IMPORTANT RULES:
- ห้ามวิเคราะห์จาก summary
- ห้ามตีความข้อความเอกสารต้นฉบับ
- ห้ามใช้ความรู้ภายนอก
- ให้ใช้เฉพาะ structured analysis ด้านล่างเท่านั้น
- ทุกคะแนนต้องมีเหตุผลจาก structured analysis 
- ถ้าไม่มีหลักฐาน → ให้คะแนนต่ำ
- ห้ามตีความและให้คะแนนจาก ai_comment 
- ให้ใช้ ai_comment เป็นข้อมูลประกอบเสริมสำหรับ risk_comment เท่านั้น


========================================
STRUCTURED CHANGE ANALYSIS
========================================
{structured_analysis}
========================================


====================================================
RISK DIMENSIONS (ต้องประเมินทุกมิติ)
====================================================

1) scope
2) timeline
3) cost
4) resource
5) risk
6) contract
7) stakeholder
8) architecture


====================================================
SCORING SCALE (0–100)
====================================================

0–5      = negligible impact
6–15     = very low impact
16–30    = low impact
31–50    = moderate impact
51–70    = high impact
71–85    = very high impact
86–100   = critical impact


====================================================
DETAILED SCORING RUBRIC
====================================================


-----------------------------
1) SCOPE (ขอบเขตงาน)
-----------------------------

typo / wording only → 0–5  
quantity change <5% → 6–15  
quantity change 5–10% → 16–30  
minor deliverable change → 31–50  
major deliverable change → 51–70  
project objective change → 71–85  
redefine project scope → 86–100  

ตัวอย่าง:
42 เครื่อง → 40 เครื่อง = ลด <5% → very low impact


-----------------------------
2) TIMELINE (ระยะเวลา)
-----------------------------

no schedule impact → 0–5  
minor adjustment → 6–15  
delay <10% → 16–30  
milestone change → 31–50  
critical phase delay → 51–70  
timeline restructuring → 71–85  
full rebaseline → 86–100  


-----------------------------
3) COST (งบประมาณ)
-----------------------------

<1% change → 0–5  
1–3% → 6–15  
3–10% → 16–30  
10–20% → 31–50  
20–40% → 51–70  
>40% → 71–85  
funding structure change → 86–100  


-----------------------------
4) RESOURCE (ทรัพยากร)
-----------------------------

minor allocation → 0–5  
workload adjustment → 6–15  
role adjustment → 16–30  
team restructure → 31–50  
new capability required → 51–70  
organizational change → 71–85  
core team replacement → 86–100  


-----------------------------
5) RISK EXPOSURE (ความเสี่ยงรวม)
-----------------------------

no new risk → 0–5  
small uncertainty → 6–15  
moderate uncertainty → 16–30  
new risk domain → 31–50  
high probability risk → 51–70  
cascading risks → 71–85  
existential threat → 86–100  


-----------------------------
6) CONTRACT (สัญญา)
-----------------------------

wording only → 0–5  
clarification → 6–15  
minor obligation change → 16–30  
SLA / penalty change → 31–50  
liability change → 51–70  
legal structure change → 71–85  
contract renegotiation → 86–100  


-----------------------------
7) STAKEHOLDER
-----------------------------

no impact → 0–5  
communication change → 6–15  
expectation change → 16–30  
role change → 31–50  
power shift → 51–70  
new stakeholder group → 71–85  
governance change → 86–100  


-----------------------------
8) ARCHITECTURE
-----------------------------

configuration tweak → 0–5  
component adjustment → 6–15  
interface change → 16–30  
module redesign → 31–50  
integration change → 51–70  
platform migration → 71–85  
architecture paradigm change → 86–100  


====================================================
OVERALL RISK LEVEL (MANDATORY RULE)
====================================================

ใช้ค่าคะแนนที่ "สูงที่สุด" ของทุกมิติ (MAX SCORE)

0–25  = LOW
26–60 = MEDIUM
61–100 = HIGH


====================================================
OUTPUT FORMAT (JSON ONLY)
====================================================

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

  "risk_comment": "อธิบายเหตุผลเชิงวิเคราะห์โดยอ้างอิงหลาย paragraph จาก structured analysis และระบุว่ามิติใดมีผลสูงสุด ใส่newlineให้อ่านง่าย",

  "overall_risk_level": "LOW | MEDIUM | HIGH"
}}


STRICT OUTPUT RULES:
- JSON เท่านั้น
- ห้าม markdown
- ห้ามข้อความอื่น
- ห้ามอธิบายนอก JSON
""")

    impact_chain = impact_prompt | llm | StrOutputParser()

    try:
        raw_risk = impact_chain.invoke({
            "structured_analysis": structured_analysis
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