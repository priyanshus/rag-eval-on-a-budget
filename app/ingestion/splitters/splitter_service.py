import hashlib
import re
from typing import List

from app.ingestion.models import RawDocument, ChunkedDocumentModel, ChunkedDocumentsOutput
from app.ingestion.splitters.base import BaseSplitter


def _calculate_hash_for_chunk(chunk: str) -> str:
    """
    Normalize chunk text and return a SHA256 hash
    """
    text = chunk.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SplitterService:
    def __init__(self, splitter: BaseSplitter):
        self._splitter_strategy = splitter

    def _chunk(self, raw_document: RawDocument) -> ChunkedDocumentsOutput:
        """
        Splits a RawDocument into chunks and returns ChunkedDocumentsOutput
        """
        chunks_text = self._splitter_strategy.split(raw_document.text)
        chunk_models: List[ChunkedDocumentModel] = []

        for i, chunk in enumerate(chunks_text):
            chunk_hash = _calculate_hash_for_chunk(chunk)
            chunk_id = f"{raw_document.row_id}_chunk_{i}_{chunk_hash[:8]}"  # unique per chunk

            chunk_models.append(
                ChunkedDocumentModel(
                    chunk_text=chunk,
                    source_row_id=raw_document.row_id,
                    chunk_id=chunk_id,
                    metadata={
                        "author": raw_document.author,
                        "source": raw_document.source,
                        "title": raw_document.title
                    },
                    dense_vector=None,
                    sparse_vector=None
                )
            )

        return ChunkedDocumentsOutput(
            source_row_id=raw_document.row_id,
            chunks=chunk_models
        )

    def chunk(self, docs: List[RawDocument]) -> List[ChunkedDocumentsOutput]:
        """
        Chunk multiple RawDocuments
        """
        return [self._chunk(doc) for doc in docs]
