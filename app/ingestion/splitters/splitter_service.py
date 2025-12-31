import hashlib
import re
from typing import List

from app.ingestion.models import RawDocumentModel, ChunkedDocumentModel, ChunkModel
from app.ingestion.splitters.base import BaseSplitter


def _calculate_hash_for_chunk(chunk: str) -> str:
    text = chunk.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SplitterService:
    def __init__(self, splitter: BaseSplitter):
        self._splitter_strategy = splitter

    def _chunk(self, raw_document: RawDocumentModel) -> ChunkedDocumentModel:
        chunks = self._splitter_strategy.split(raw_document.article)
        chunk_models = []

        for chunk in chunks:
            chunk_hash = _calculate_hash_for_chunk(chunk)
            chunk_models.append(ChunkModel(hash=chunk_hash, chunked_text=chunk))

        return ChunkedDocumentModel(metadata=raw_document, chunks=chunk_models)

    def chunk(self, docs: List[RawDocumentModel]) -> List[ChunkedDocumentModel]:
        return [self._chunk(doc) for doc in docs]
