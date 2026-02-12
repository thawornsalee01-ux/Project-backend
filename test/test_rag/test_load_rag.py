from src.RAG.pdf_load_rag import PDFLoaderHybrid

loader = PDFLoaderHybrid(
    min_text_length=50,  # ต่อหน้า
)

text = loader.load_from_file("data/samples/13.pdf")

print("Text length:", len(text))
print(text)
