from src.ingestion.pdf_load import PDFLoader
from src.ingestion.paragraph import ParagraphSplitter
from src.embedding.embed import EmbeddingService


def test_pdf_pipeline(pdf_path: str):
    loader = PDFLoader()
    splitter = ParagraphSplitter()
    service = EmbeddingService()

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = loader.load_from_bytes(pdf_bytes)
    paragraphs = splitter.split(pages)

    print(f"โหลด PDF สำเร็จ: {len(pages)} หน้า")
    print(f"แยกเป็น paragraph: {len(paragraphs)} paragraphs\n")

    for i, p in enumerate(paragraphs):
        print(f"=== Paragraph {i} (หน้า {p.page_number}) ===")

        token_chunks = service._chunk_tokens(p.text)
        print(f"จำนวน chunk: {len(token_chunks)}")

        for j, tc in enumerate(token_chunks):
            decoded = service.tokenizer.decode(tc)
            print(f"\n[Chunk {j}] token={len(tc)}")
            print(decoded[:300])

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    test_pdf_pipeline("data/samples/l4.pdf")
