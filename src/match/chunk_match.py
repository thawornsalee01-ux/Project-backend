# src/match/chunk_matcher.py

import numpy as np
from typing import List, Dict


class ChunkMatcher:
    """
    Stage 2: Chunk-level semantic comparison
    """

    def _cosine(self, a: List[float], b: List[float]) -> float:
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def compare(
        self,
        old_chunks: List[List[float]],
        new_chunks: List[List[float]],
        threshold: float = 0.85,
    ) -> Dict[str, object]:

        if not old_chunks or not new_chunks:
            return {
                "coverage": 0.0,
                "mean_similarity": 0.0,
                "chunk_similarities": [],
            }

        chunk_sims = []

        for oc in old_chunks:
            sims = [self._cosine(oc, nc) for nc in new_chunks]
            best_sim = max(sims)
            chunk_sims.append(best_sim)

        matched = [s for s in chunk_sims if s >= threshold]

        coverage = len(matched) / len(chunk_sims)
        mean_sim = float(np.mean(chunk_sims)) if chunk_sims else 0.0

        return {
            "coverage": coverage,
            "mean_similarity": mean_sim,
            "chunk_similarities": chunk_sims,  # ⭐ เพิ่ม
        }
