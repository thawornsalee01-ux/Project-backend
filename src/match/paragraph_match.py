import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from src.ingestion.paragraph import Paragraph
from src.match.chunk_match import ChunkMatcher  # ✅ เพิ่ม
import Levenshtein


@dataclass
class MatchResult:
    old_paragraph_index: Optional[int]
    new_paragraph_index: Optional[int]
    similarity: float
    change_type: str
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    old_page: Optional[int] = None
    new_page: Optional[int] = None

    # --- optional (stage 2 เติมทีหลังได้ ไม่กระทบ API) ---
    chunk_coverage: Optional[float] = None
    mean_chunk_similarity: Optional[float] = None
    chunk_similarities: Optional[List[float]] = None
    heavy: Optional[bool] = None


class ParagraphMatcher:
    """
    Stage 1: Paragraph-level matching (coarse + chunk-aware)
    """

    def __init__(
        self,
        threshold: float = 0.75,
        embed_weight: float = 0.6,
        char_weight: float = 0.2,
        chunk_weight: float = 0.2,       # ✅ เพิ่มการถ่วงน้ำหนัก chunk
        chunk_threshold: float = 0.85,   # ✅ ใช้กำหนดความคล้ายระดับ chunk
    ):
        self.threshold = threshold
        self.embed_weight = embed_weight
        self.char_weight = char_weight
        self.chunk_weight = chunk_weight

        self.chunk_matcher = ChunkMatcher()
        self.chunk_threshold = chunk_threshold

    # ------------------------------------------------------
    # Helper Functions
    # ------------------------------------------------------
    def _cosine(self, a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _char_similarity(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return Levenshtein.ratio(a, b)

    # ------------------------------------------------------
    # Main Matching Function
    # ------------------------------------------------------
    def match(
        self,
        old_paragraphs: List[Paragraph],
        new_paragraphs: List[Paragraph],
    ) -> List[MatchResult]:

        results: List[MatchResult] = []
        used_new = set()

        # ---------- Stage 1: old → new ----------
        for old_idx, old_p in enumerate(old_paragraphs):
            best_score = 0.0
            best_new_idx = None

            for new_idx, new_p in enumerate(new_paragraphs):
                if new_idx in used_new:
                    continue
                if old_p.embedding is None or new_p.embedding is None:
                    continue

                # ----------------------------------------------
                # 1️⃣ คำนวณ similarity แบบ embedding + text
                # ----------------------------------------------
                embed_sim = self._cosine(old_p.embedding, new_p.embedding)
                char_sim = self._char_similarity(old_p.text, new_p.text)

                # ----------------------------------------------
                # 2️⃣ คำนวณ similarity แบบ chunk-level
                # ----------------------------------------------
                if old_p.chunk_embeddings and new_p.chunk_embeddings:
                    metrics = self.chunk_matcher.compare(
                        old_chunks=old_p.chunk_embeddings,
                        new_chunks=new_p.chunk_embeddings,
                        threshold=self.chunk_threshold,
                    )
                    chunk_sim = metrics.get("mean_similarity", 0.0)
                else:
                    chunk_sim = embed_sim  # fallback

                # ----------------------------------------------
                # 3️⃣ รวมคะแนนทั้ง 3 แบบ (hybrid)
                # ----------------------------------------------
                hybrid_sim = (
                    self.embed_weight * embed_sim
                    + self.char_weight * char_sim
                    + self.chunk_weight * chunk_sim
                )

                if hybrid_sim > best_score:
                    best_score = hybrid_sim
                    best_new_idx = new_idx

            # ----------------------------------------------
            # 4️⃣ บันทึกผลการจับคู่
            # ----------------------------------------------
            if best_new_idx is not None and best_score >= self.threshold:
                used_new.add(best_new_idx)

                if abs(best_score - 1.0) < 1e-6:
                    change_type = "UNCHANGED"
                else:
                    change_type = "CANDIDATE"

                results.append(
                    MatchResult(
                        old_paragraph_index=old_idx,
                        new_paragraph_index=best_new_idx,
                        similarity=best_score,
                        change_type=change_type,
                        old_text=old_p.text,
                        new_text=new_paragraphs[best_new_idx].text,
                        old_page=old_p.page_number,
                        new_page=new_paragraphs[best_new_idx].page_number,
                    )
                )
            else:
                # ไม่มีคู่ match → ถือว่าถูกลบ
                results.append(
                    MatchResult(
                        old_paragraph_index=old_idx,
                        new_paragraph_index=None,
                        similarity=0.0,
                        change_type="REMOVED",
                        old_text=old_p.text,
                        old_page=old_p.page_number,
                    )
                )

        # ---------- Stage 2: new not used → ADDED ----------
        for new_idx, new_p in enumerate(new_paragraphs):
            if new_idx not in used_new:
                results.append(
                    MatchResult(
                        old_paragraph_index=None,
                        new_paragraph_index=new_idx,
                        similarity=0.0,
                        change_type="ADDED",
                        new_text=new_p.text,
                        new_page=new_p.page_number,
                    )
                )

        return results
