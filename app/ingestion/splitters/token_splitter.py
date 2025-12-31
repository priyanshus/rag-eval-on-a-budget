from typing import List

import tiktoken

from .base import BaseSplitter


class TokenSplitter(BaseSplitter):
    def __init__(
            self,
            model_name: str = "gpt-4o-mini",
            chunk_size: int = 300,
            chunk_overlap: int = 50,
    ):
        assert chunk_overlap < chunk_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoder = tiktoken.encoding_for_model(model_name)

    def split(self, text: str) -> List[str]:
        tokens = self.encoder.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoder.decode(chunk_tokens)

            chunks.append(chunk_text)

            start += self.chunk_size - self.chunk_overlap

        return chunks
