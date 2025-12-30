
from dataclasses import dataclass


@dataclass
class RawDocumentModel:
    title: str
    author: str
    link: str
    article: str
    hash: str

@dataclass
class HybridIngestionDocumentModel:
    metadata: RawDocumentModel
    dense_vectors: list[float]
    bm25_vectors: list[float]

@dataclass
class DenseIngestionDocumentModel:
    metadata: RawDocumentModel
    dense_vectors: list[float]
