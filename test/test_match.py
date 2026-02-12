# main_test.py

from src.ingestion.pdf_load import PDFLoader
from src.ingestion.paragraph import ParagraphSplitter
from src.embedding.embed import EmbeddingService
from src.match.paragraph_match import ParagraphMatcher
from src.match.match_resolver import MatchResolver


def pretty_print_results(results, old_paragraphs, new_paragraphs, max_items: int = 50):
    for i, r in enumerate(results[:max_items]):
        print("=" * 80)
        print(f"[{i}] Change Type : {r.change_type}")
        print(f"    Old Index  : {r.old_paragraph_index}")
        print(f"    New Index  : {r.new_paragraph_index}")

        if r.similarity is not None:
            print(f"    Similarity : {r.similarity:.6f}")

        # ===============================
        # Chunk-level metrics
        # ===============================
        if r.chunk_coverage is not None:
            print(f"    Coverage   : {r.chunk_coverage:.4f}")

        if r.mean_chunk_similarity is not None:
            print(f"    Mean Chunk : {r.mean_chunk_similarity:.6f}")

        min_chunk = None
        if r.chunk_similarities:
            print("    Chunk similarities:")
            for idx, s in enumerate(r.chunk_similarities):
                flag = "✓" if s >= 0.85 else "!"
                print(f"      [{idx:02d}] {s:.4f} {flag}")

            min_chunk = min(r.chunk_similarities)
            print(f"    Min chunk similarity : {min_chunk:.4f}")

        # ===============================
        # Length + chunk drop heuristic
        # ===============================
        if r.old_paragraph_index is not None and r.new_paragraph_index is not None:
            old_p = old_paragraphs[r.old_paragraph_index]
            new_p = new_paragraphs[r.new_paragraph_index]

            old_len = len(old_p.text) if old_p.text else 0
            new_len = len(new_p.text) if new_p.text else 0

            if old_len > 0:
                length_ratio = new_len / old_len
                print(f"    Length ratio (new/old) : {length_ratio:.2f}")

            old_chunks = len(old_p.chunk_embeddings or [])
            new_chunks = len(new_p.chunk_embeddings or [])
            if old_chunks > 0:
                chunk_drop_ratio = new_chunks / old_chunks
                print(f"    Chunk drop ratio      : {chunk_drop_ratio:.2f}")

        # ===============================
        # Final verdict (3 levels)
        # ===============================
        if hasattr(r, "edit_severity"):
            emoji = {
                "HEAVY": "🔥",
                "MEDIUM": "⚠️",
                "LIGHT": "✅",
            }.get(r.edit_severity, "")
            print(f"    Edit Severity         : {r.edit_severity} {emoji}")

        # ===============================
        # Text preview
        # ===============================
        if r.old_text:
            snippet = r.old_text.replace("\n", " ")
            print(f"    Old Text : {snippet[:200]}{'...' if len(snippet) > 200 else ''}")

        if r.new_text:
            snippet = r.new_text.replace("\n", " ")
            print(f"    New Text : {snippet[:200]}{'...' if len(snippet) > 200 else ''}")

    print("=" * 80)
    print(f"Total results: {len(results)}")


def main():
    print("📄 Loading PDFs...")
    loader = PDFLoader()

    with open("data/samples/l2.pdf", "rb") as f:
        pdf_bytes_old = f.read()
    with open("data/samples/l3.pdf", "rb") as f:
        pdf_bytes_new = f.read()

    pages_old = loader.load_from_bytes(pdf_bytes_old)
    pages_new = loader.load_from_bytes(pdf_bytes_new)

    print(f"  V1 pages: {len(pages_old)}")
    print(f"  V2 pages: {len(pages_new)}")

    print("✂ Splitting paragraphs...")
    splitter = ParagraphSplitter()
    old_paragraphs = splitter.split(pages_old)
    new_paragraphs = splitter.split(pages_new)

    print(f"  V1 paragraphs: {len(old_paragraphs)}")
    print(f"  V2 paragraphs: {len(new_paragraphs)}")

    print("🔗 Embedding paragraphs...")
    embedder = EmbeddingService()
    embedder.embed_paragraphs(old_paragraphs)
    embedder.embed_paragraphs(new_paragraphs)

    print("🔍 Matching paragraphs (Stage 1)...")
    matcher = ParagraphMatcher(
        threshold=0.75,
        embed_weight=0.7,
        char_weight=0.3,
    )
    stage1_results = matcher.match(old_paragraphs, new_paragraphs)

    print("🧠 Resolving semantic changes (Stage 2)...")
    resolver = MatchResolver(chunk_threshold=0.85)
    stage2_results = resolver.resolve(
        stage1_results,
        old_paragraphs,
        new_paragraphs,
    )

    changed_only = [
        r for r in stage2_results
        if r.change_type in ("MODIFIED", "ADDED", "REMOVED")
    ]

    print("\n✅ Stage 2 results (only changes)")
    pretty_print_results(
        changed_only,
        old_paragraphs,
        new_paragraphs,
        max_items=20,
    )


# ⭐⭐ ตรงนี้ต้องอยู่นอก main ⭐⭐
if __name__ == "__main__":
    main()