import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


# =====================================================
# REVIEW MODEL (temp ต่ำ = strict reasoning)
# =====================================================
review_llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY"),
    model=os.getenv("LOCALMODEL_MODEL_SUM", "openai/gpt-oss-120b"),
    temperature=0.0,
)


# =====================================================
# SELF REFLECTION FUNCTION
# =====================================================
def refine_answer(question: str, answer: str) -> str:
    """
    ตรวจคำตอบ AI ว่าตรงคำถามไหม
    ถ้าไม่ตรง → rewrite
    """

    prompt = f"""
คุณคือ AI ตรวจคำตอบ

หน้าที่:
ตรวจสอบว่าคำตอบตรงคำถามหรือไม่

คำถาม:
{question}

คำตอบเดิม:
{answer}

กฎ:
- ตอบเฉพาะสิ่งที่ถูกถาม
- ห้ามข้อมูลส่วนเกิน
- ห้ามเดา
- ถ้าไม่ตรง ให้เขียนใหม่ให้ตรง 100%
- ถ้าตรงแล้วให้ส่งคำตอบเดิม

ส่งเฉพาะคำตอบสุดท้าย
"""

    result = review_llm.invoke([HumanMessage(content=prompt)])
    return result.content.strip()


# =====================================================
# OPTIONAL — MULTI PASS REFINEMENT
# =====================================================
def refine_answer_loop(question: str, answer: str, max_pass=2) -> str:

    for _ in range(max_pass):
        improved = refine_answer(question, answer)

        if improved == answer:
            break

        answer = improved

    return answer
