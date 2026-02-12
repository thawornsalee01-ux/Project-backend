import os
from dotenv import load_dotenv

from src.AI.Tools.tavily_search import get_web_tools
from langchain_openai import ChatOpenAI
from langchain.messages import SystemMessage, HumanMessage

# =========================
# Load environment variables
# =========================
load_dotenv()

# =========================
# 1) Search ด้วย Tavily
# =========================
tools = get_web_tools()
search_tool = tools[0]

query = "ผู้ชนะแชมป์โลก F1 ปี2025ล่าสุดคือใคร"
results = search_tool.run(query)

print("🔍 Search Results:")
for r in results:
    print("-", r)

# =========================
# 2) รวมผลลัพธ์เป็น context
# =========================
def build_context(results, max_chars=4000):
    chunks = []
    total = 0

    for r in results:
        text = f"{r.get('title', '')} - {r.get('content', '')}"
        total += len(text)

        if total > max_chars:
            break

        chunks.append(text)

    return "\n".join(chunks)

context = build_context(results)

# =========================
# 3) ใช้ Local LLM (OpenAI-compatible)
# =========================
llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY") or "local-key",  # สำคัญ: ต้องมีค่า
    model=os.getenv("LOCALMODEL_MODEL_COMMENT"),
    temperature=0.2,
)

messages = [
    SystemMessage(
        content=(
            "คุณคือผู้ช่วย AI สำหรับสรุปข้อมูลจากเว็บ "
            "ตอบเป็นภาษาไทย ชัดเจน กระชับ "
            "ห้ามเดาข้อมูล ถ้าไม่มั่นใจให้บอกว่าไม่พบข้อมูล"
        )
    ),
    HumanMessage(
        content=f"""
คำถาม:
{query}

ข้อมูลจากเว็บ:
{context}

กรุณาตอบให้ตรงคำถาม พร้อมชื่อบุคคล และปีล่าสุด
"""
    ),
]

# =========================
# 4) เรียก LLM (LangChain ใหม่ใช้ invoke)
# =========================
response = llm.invoke(messages)

print("\n🤖 AI Answer:")
print(response.content)
