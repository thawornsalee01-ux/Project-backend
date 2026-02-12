# test_full_pipeline.py

from src.ingestion.pdf_load import PDFLoader
from src.ingestion.paragraph import ParagraphSplitter
from src.embedding.embed import EmbeddingService

from src.match.paragraph_match import ParagraphMatcher
from src.match.match_resolver import MatchResolver
from src.diff.diff import DiffEngine


def print_changes(changes):
    for i, c in enumerate(changes):
        print("=" * 80)
        print(f"[{i}] {c.change_type.upper()} | {c.section_label}")

        # ---------- verdict ----------
        if c.risk_level:
            emoji = "🔥" if c.risk_level == "HIGH" else "🟡" if c.risk_level == "MEDIUM" else "🟢"
            print(f"EDIT SEVERITY : {c.risk_level} {emoji}")

        # ---------- metrics ----------
        if c.similarity is not None:
            print(f"paragraph_similarity  : {c.similarity:.4f}")

        if c.mean_similarity is not None:
            print(f"chunk_mean_similarity : {c.mean_similarity:.4f}")

        if c.coverage is not None:
            print(f"coverage               : {c.coverage:.2f}")

        # ---------- quick human hint ----------
        if c.change_type == "MODIFIED":
            if c.risk_level == "HIGH":
                print("→ Reason: เนื้อหาเดิมหาย / ถูกตัด / แก้สาระสำคัญ")
            elif c.risk_level == "MEDIUM":
                print("→ Reason: มีการเพิ่ม/ปรับ แต่โครงยังเดิม")
            else:
                print("→ Reason: แก้เล็กน้อย (ถ้อยคำ / เรียบเรียง)")

        # ---------- text ----------
        if c.old_text:
            print("OLD:", c.old_text[:600].replace("\n", " "), "...")

        if c.new_text:
            print("NEW:", c.new_text[:600].replace("\n", " "), "...")

    print("=" * 80)
    print(f"TOTAL CHANGES: {len(changes)}")


def main():
    print("📄 Load PDFs")
    loader = PDFLoader()

    with open("data/samples/l2.pdf", "rb") as f:
        old_pdf = f.read()

    with open("data/samples/l3.pdf", "rb") as f:
        new_pdf = f.read()

    pages_old = loader.load_from_bytes(old_pdf)
    pages_new = loader.load_from_bytes(new_pdf)

    print(f"Pages old: {len(pages_old)}, new: {len(pages_new)}")

    print("✂ Split paragraphs")
    splitter = ParagraphSplitter()
    old_paragraphs = splitter.split(pages_old)
    new_paragraphs = splitter.split(pages_new)

    print(f"Paragraphs old: {len(old_paragraphs)}, new: {len(new_paragraphs)}")

    print("🔗 Embedding")
    embedder = EmbeddingService()
    embedder.embed_paragraphs(old_paragraphs)
    embedder.embed_paragraphs(new_paragraphs)

    print("🔍 Matching paragraphs (Stage 1)")
    matcher = ParagraphMatcher(threshold=0.75)
    stage1_matches = matcher.match(old_paragraphs, new_paragraphs)

    print(f"MatchResult count (stage1): {len(stage1_matches)}")

    print("🧠 Resolve semantic changes (Stage 2)")
    resolver = MatchResolver(chunk_threshold=0.85)
    resolved_matches = resolver.resolve(
        stage1_matches,
        old_paragraphs,
        new_paragraphs,
    )

    print("📝 Diff")
    diff_engine = DiffEngine()
    changes = diff_engine.build_changes(resolved_matches)

    print_changes(changes)

    print("\n📊 DOCUMENT EDIT INTENSITY")
    print("=>", diff_engine.compute_edit_intensity(changes))


if __name__ == "__main__":
    main()
