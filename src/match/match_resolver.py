from typing import List
from src.match.chunk_match import ChunkMatcher
from src.match.paragraph_match import MatchResult
from src.ingestion.paragraph import Paragraph


class MatchResolver:
    """
    Stage 2:
    - CANDIDATE = MODIFIED เสมอ
    - วิเคราะห์ 3 ระดับ: LIGHT / MEDIUM / HEAVY
    """

    def __init__(self, chunk_threshold: float = 0.85):
        self.chunk_matcher = ChunkMatcher()
        self.chunk_threshold = chunk_threshold

    def resolve(
        self,
        matches: List[MatchResult],
        old_paragraphs: List[Paragraph],
        new_paragraphs: List[Paragraph],
    ) -> List[MatchResult]:

        resolved: List[MatchResult] = []

        for m in matches:
            # ----------------------------------------------
            # ADDED / REMOVED / UNCHANGED
            # ----------------------------------------------
            if m.change_type != "CANDIDATE":
                resolved.append(m)
                continue

            # ----------------------------------------------
            # CANDIDATE → MODIFIED
            # ----------------------------------------------
            m.change_type = "MODIFIED"

            if m.old_paragraph_index is None or m.new_paragraph_index is None:
                m.edit_severity = "HEAVY"
                m.heavy = True
                resolved.append(m)
                continue

            old_p = old_paragraphs[m.old_paragraph_index]
            new_p = new_paragraphs[m.new_paragraph_index]

            if not old_p.chunk_embeddings or not new_p.chunk_embeddings:
                m.edit_severity = "HEAVY"
                m.heavy = True
                resolved.append(m)
                continue

            # ----------------------------------------------
            # Chunk comparison
            # ----------------------------------------------
            metrics = self.chunk_matcher.compare(
                old_chunks=old_p.chunk_embeddings,
                new_chunks=new_p.chunk_embeddings,
                threshold=self.chunk_threshold,
            )

            coverage = metrics.get("coverage", 0.0)
            mean_sim = metrics.get("mean_similarity", 0.0)
            chunk_sims = metrics.get("chunk_similarities", [])

            min_chunk = min(chunk_sims) if chunk_sims else 0.0

            # ----------------------------------------------
            # Length + chunk drop
            # ----------------------------------------------
            old_len = len(old_p.text) if old_p.text else 0
            new_len = len(new_p.text) if new_p.text else 0
            length_ratio = (new_len / old_len) if old_len > 0 else 0.0

            old_chunk_count = len(old_p.chunk_embeddings)
            new_chunk_count = len(new_p.chunk_embeddings)
            chunk_drop_ratio = (
                new_chunk_count / old_chunk_count
                if old_chunk_count > 0 else 0.0
            )

            # ----------------------------------------------
            # ⭐⭐ FINAL SEVERITY RULE ⭐⭐
            # ----------------------------------------------
            if (
                coverage < 0.80
                or min_chunk < 0.70
                or length_ratio < 0.75
                or chunk_drop_ratio < 0.80   # ⭐ สำคัญ
            ):
                severity = "HEAVY"

            elif (
                coverage < 0.95
                or min_chunk < 0.85
                or length_ratio < 0.90
                or chunk_drop_ratio < 0.96
            ):
                severity = "MEDIUM"

            else:
                severity = "LIGHT"

            # ----------------------------------------------
            # Assign
            # ----------------------------------------------
            m.chunk_coverage = coverage
            m.mean_chunk_similarity = mean_sim
            m.chunk_similarities = chunk_sims

            m.edit_severity = severity
            m.heavy = (severity == "HEAVY")

            resolved.append(m)

        return resolved
