from typing import List, Dict, Optional

from pydantic import BaseModel


class RawDocument(BaseModel):
    """
    Represents a raw document read from CSV or other sources.
    """
    row_id: int  # unique CSV row identifier
    title: Optional[str] = None
    author: Optional[str] = None
    source: Optional[str] = None  # e.g., URL or filename
    text: str  # raw document content
    hash: str


class ChunkedDocumentModel(BaseModel):
    """
    Represents a single chunk generated from a RawDocument.
    Contains chunk text, metadata, and embeddings.
    """
    chunk_text: str  # text content for embedding
    source_row_id: int  # reference back to RawDocument.row_id
    chunk_id: str  # unique identifier for this chunk
    metadata: Optional[Dict] = None  # optional per-chunk metadata
    dense_vector: Optional[List[float]] = None  # dense embedding vector
    sparse_vector: Optional[Dict[str, List[float]]] = None


class ChunkedDocumentsOutput(BaseModel):
    """
    Represents the output of chunking for a single raw document.
    """
    source_row_id: int  # reference to RawDocument.row_id
    chunks: List[ChunkedDocumentModel]  # list of ingestion-ready chunks
