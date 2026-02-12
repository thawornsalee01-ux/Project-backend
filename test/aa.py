import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

print("BASE_URL =", os.getenv("LOCALMODEL_BASE_URL"))
print("API_KEY  =", os.getenv("LOCALMODEL_API_KEY"))
print("MODEL    =", os.getenv("LOCALMODEL_MODEL_SUM"))

llm = ChatOpenAI(
    base_url=os.getenv("LOCALMODEL_BASE_URL"),
    api_key=os.getenv("LOCALMODEL_API_KEY"),
    model=os.getenv("LOCALMODEL_MODEL_SUM"),
    temperature=0.2,
)

try:
    res = llm.invoke([
        HumanMessage(content="สวัสดี")
    ])
    print("✅ AI ตอบกลับ:")
    print(res.content)

except Exception as e:
    print("❌ ERROR:", e)
