import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List
from src.ingestion.paragraph import Paragraph


class EmbeddingService:
    """
    สร้าง embedding vector สำหรับ paragraph
    ใช้ model: intfloat/multilingual-e5-base
    รองรับ token-level chunking (ปลอดภัย 100%)
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        device: str | None = None,
        max_length: int = 1024,
        chunk_size: int = 50,
        chunk_overlap: int = 20,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length

        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id

    # --------------------------------------------------
    # 1) tokenize + chunk (NO special token)
    # --------------------------------------------------
    def _chunk_tokens(self, text: str) -> List[List[int]]:
        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
        )["input_ids"]

        if not tokens:
            return []

        chunks: List[List[int]] = []
        start = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunks.append(tokens[start:end])
            start += self.chunk_size - self.chunk_overlap

        return chunks

    # --------------------------------------------------
    # 2) embed token chunks → Tensor [n_chunks, dim]
    # --------------------------------------------------
    def _embed_token_chunks(self, token_chunks: List[List[int]]) -> torch.Tensor:
        if not token_chunks:
            return torch.empty((0, self.model.config.hidden_size))

        input_ids = []
        attention_masks = []

        for chunk in token_chunks:
            ids = [self.cls_id] + chunk + [self.sep_id]

            if len(ids) > self.max_length:
                ids = ids[: self.max_length - 1] + [self.sep_id]

            mask = [1] * len(ids)

            input_ids.append(torch.tensor(ids, dtype=torch.long))
            attention_masks.append(torch.tensor(mask, dtype=torch.long))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_id,
        ).to(self.device)

        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_masks,
            batch_first=True,
            padding_value=0,
        ).to(self.device)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        embeddings = output.pooler_output
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    # --------------------------------------------------
    # 3) embed paragraphs (API หลัก)
    # --------------------------------------------------
    def embed_paragraphs(self, paragraphs: List[Paragraph]) -> None:
        """
        - p.chunk_embeddings : List[List[float]]
        - p.embedding        : mean(chunk_embeddings)
        """

        for p in paragraphs:
            token_chunks = self._chunk_tokens(p.text)
            chunk_tensor = self._embed_token_chunks(token_chunks)

            if chunk_tensor.numel() == 0:
                p.chunk_embeddings = []
                p.embedding = None
                continue

            chunk_embeddings = chunk_tensor.cpu().numpy().tolist()

            # ✅ เก็บ chunk embedding (สำหรับ diff)
            p.chunk_embeddings = chunk_embeddings

            # ✅ paragraph embedding (สำหรับ matcher)
            p.embedding = np.mean(chunk_tensor.cpu().numpy(), axis=0).tolist()
