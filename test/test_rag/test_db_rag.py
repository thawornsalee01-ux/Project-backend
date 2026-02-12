from src.RAG.pdf_load_rag import PDFLoaderHybrid
from src.RAG.embed_rag import PDFChunkEmbedder
from src.RAG.chroma_store import ChromaVectorStore
import os


if __name__ == "__main__":
    # =============================
    # 1️⃣ โหลด PDF เป็น bytes
    # =============================
    pdf_path = "data/samples/o3.pdf"

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    loader = PDFLoaderHybrid()
    full_text = loader.load_from_bytes(pdf_bytes)

    print("✅ PDF loaded")
    print("Text length:", len(full_text))

    # =============================
    # 2️⃣ chunk + embed
    # =============================
    embedder = PDFChunkEmbedder(
        chunk_size=200,
        overlap=60,
    )

    records = embedder.process_text(
        text=full_text,
        document_id="doc_003",
    )

    print("✅ Chunk + embed finished")
    print("Total chunks:", len(records))
    print("Vector dim:", len(records[0]["embedding"]))

    print("\n--- SAMPLE CHUNK ---")
    print(records[0]["text"][:500])
    print("Metadata:", records[0]["metadata"])

    # =============================
    # 3️⃣ บันทึกลง ChromaDB
    # =============================
    pdf_name = os.path.basename(pdf_path)  # ✅ อัตโนมัติ

    store = ChromaVectorStore(
        persist_dir="data/chroma",
        collection_name="rag_documents",
    )

    store.add_records(
        records=records,
        pdf_name=pdf_name,
    )

    print("✅ Saved to ChromaDB")
    print("Total vectors in DB:", store.count())
