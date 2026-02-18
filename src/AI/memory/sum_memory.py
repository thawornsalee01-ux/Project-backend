from langchain_core.messages import HumanMessage

def summarize_conversation(llm, text):

    prompt = f"""
สรุปบทสนทนานี้ให้สั้น กระชับ และเก็บเฉพาะประเด็นสำคัญที่ต้องจำระยะยาว

{text}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content