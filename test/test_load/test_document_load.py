from src.ingestion.document_load import DocumentLoader

with open("data/samples/basic Scope of Work_for_VST (1).pdf", "rb") as f:
    file_bytes = f.read()

loader = DocumentLoader()
pages = loader.load_from_bytes(file_bytes)

for page in pages:
    print(f"--- Page {page.page_number} ---")
    print(page.text)
