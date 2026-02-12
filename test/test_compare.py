# test_compare.py
import logging
from pathlib import Path
from src.service.compare import run_compare

logging.basicConfig(level=logging.INFO)

def main():
    # ตัวอย่างไฟล์ PDF ที่ใช้ทดสอบ
    v1_path = Path("data/samples/b1.pdf")
    v2_path = Path("data/samples/b2.pdf")

    if not v1_path.exists() or not v2_path.exists():
        print("❌ ต้องมีไฟล์ PDF test v1.pdf และ v2.pdf ใน folder data/test")
        return

    with open(v1_path, "rb") as f:
        v1_bytes = f.read()
    with open(v2_path, "rb") as f:
        v2_bytes = f.read()

    try:
        result = run_compare(
            doc_name="Test Document",
            v1_file_bytes=v1_bytes,
            v2_file_bytes=v2_bytes,
            v1_label="v1-test",
            v2_label="v2-test",
        )
        print("✅ run_compare result:")
        for k, v in result.items():
            print(f"{k}: {v}")
    except Exception as e:
        print("❌ run_compare failed:", e)

if __name__ == "__main__":
    main()
