
from typing import Dict, List
from .base import BaseSplitter


class CharSplitter(BaseSplitter):
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        assert chunk_overlap < chunk_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str, metadata: Dict) -> List[Dict]:
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append({
                "text": chunk,
                "metadata": metadata.copy()
            })
            start += self.chunk_size - self.chunk_overlap

        return chunks
