from src.ingestion.pdf_load_ocr import TyphoonOCRClient

if __name__ == "__main__":
    client = TyphoonOCRClient()  
    file_path = "data/samples/TOr-1.pdf"      # PDF หรือรูปภาพ
    pages = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]               # เลือกหน้า ถ้า None = ทุกหน้า

    try:
        text = client.extract_text_from_file(file_path, pages=pages)
        print("===== OCR Result =====")
        print(text)
    except Exception as e:
        print("OCR failed:", e)
