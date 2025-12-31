from dataclasses import dataclass
from typing import List


@dataclass
class RawDocumentModel:
    title: str
    author: str
    link: str
    article: str
    hash: str


@dataclass
class ChunkModel:
    chunked_text: str
    hash: str


@dataclass
class ChunkedDocumentModel:
    metadata: RawDocumentModel
    chunks: List[ChunkModel]


@dataclass
class IngestionDocumentModel:
    metadata: RawDocumentModel
    dense_vectors: List[float]
    bm25_vectors: {}
