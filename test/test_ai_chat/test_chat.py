import sys
import os
from pathlib import Path

# ให้ import project ได้
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.db.session import SessionLocal
from src.AI.ai_chat.ai_chat_pipeline import run_ai_chat  # เปลี่ยน path ให้ตรงไฟล์คุณ

# ================================================
# 🔹 ตั้งค่า test
# ================================================
TEST_CHANGE_ID = 94   # 🔥 เปลี่ยนเป็น change_id ที่มีจริงใน DB
TEST_QUESTION = "การแก้ไขนี้เรื่องอะไร"

# ================================================
# 🔹 Run Test
# ================================================
def main():
    db = SessionLocal()

    try:
        print("🚀 เริ่มทดสอบ AI Chat\n")

        result = run_ai_chat(
            db=db,
            change_id=TEST_CHANGE_ID,
            user_message=TEST_QUESTION
        )

        print("\n==============================")
        print("📌 คำตอบจาก AI")
        print("==============================\n")
        print(result)

    except Exception as e:
        print(f"❌ ERROR: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    main()
