from typing import List

from .base import BaseSplitter
from .token_splitter import TokenSplitter


class RecursiveSplitter(BaseSplitter):
    def __init__(
            self,
            chunk_size: int = 300,
            chunk_overlap: int = 50,
            model_name: str = "gpt-4o-mini"
    ):
        self.token_splitter = TokenSplitter(
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []

        for para in paragraphs:
            token_count = len(self.token_splitter.encoder.encode(para))
            if token_count <= self.token_splitter.chunk_size:
                chunks.append(para)
            else:
                chunks.extend(self.token_splitter.split(para))

        return chunks
