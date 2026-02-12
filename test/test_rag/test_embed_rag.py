from src.RAG.embed_rag import PDFChunkEmbedder

if __name__ == "__main__":
    embedder = PDFChunkEmbedder()

    with open("data/samples/p1.pdf", "rb") as f:
        pdf_bytes = f.read()

    records = embedder.process_pdf(
        pdf_bytes=pdf_bytes,
        document_id="doc_001",
    )

    print("Total chunks:", len(records))
    print("Vector dim:", len(records[0]["embedding"]))
    print("Sample metadata:", records[0]["metadata"])

    print("\nFirst 10 vector values:")
    print(records[0]["embedding"][:10])

    print("\nSample text:\n", records[0]["text"][:1000])
