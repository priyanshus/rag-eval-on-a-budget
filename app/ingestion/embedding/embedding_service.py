from typing import List
from sentence_transformers import SentenceTransformer

from fastembed import SparseTextEmbedding
from app.ingestion.models import RawDocumentModel, HybridIngestionDocumentModel, DenseIngestionDocumentModel
from app.ingestion.splitters.base import BaseSplitter


class EmbeddingService:
    def __init__(
        self,
        dense_model: str,
        splitter: BaseSplitter,
        sparse_model: str = "Qdrant/bm25",
    ):
        self.splitter_service = splitter

        # Dense embedding model (MiniLM etc.)
        self.dense_model = SentenceTransformer(dense_model)

        # Sparse BM25 embedding model
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model)

    def embed_chunks_dense(self, docs: List[str]) -> [List[float]]:
        """
        Embed multiple text chunks into dense vectors.
        """
        return self.dense_model.encode(docs, show_progress_bar=True)

    def embed_chunk_dense(self, doc: str) -> List[float]:
        """
        Embed a single text chunk into a dense vector.
        """
        return self.dense_model.encode(doc)

    def embed_chunks_sparse(self, docs: List[str]) -> List[dict]:
        """
        Embed multiple text chunks into sparse BM25 vectors.
        Returns a list of dicts with 'indices' and 'values'.
        """
        return list(self.sparse_model.embed(docs))

    def embed_documents_for_hybrid_ingestion(
        self, raw_documents: List[RawDocumentModel]
    ) -> List[HybridIngestionDocumentModel]:
        """
        Split and embed all raw documents into chunks.
        Returns HybridIngestionDocumentModel with dense + sparse vectors.
        """
        embedded_docs: List[HybridIngestionDocumentModel] = []

        for doc in raw_documents:
            # Split into chunks
            splitted_docs = self.splitter_service.split(
                doc.article,
                {
                    "author": doc.author,
                    "link": doc.link,
                    "hash": doc.hash,
                    "text": doc.article,
                    "title": doc.title,
                },
            )

            chunks = [c["text"] if isinstance(c, dict) else c for c in splitted_docs]

            # Dense embeddings
            dense_embeddings = self.embed_chunks_dense(chunks)
            # Sparse embeddings
            sparse_embeddings = self.embed_chunks_sparse(chunks)

            for dense_vec, sparse_vec in zip(dense_embeddings, sparse_embeddings):
                hybrid_doc_model = HybridIngestionDocumentModel(
                    metadata=doc,
                    dense_vectors=dense_vec,
                    bm25_vectors={
                        "indices": sparse_vec.indices,
                        "values": sparse_vec.values,
                    },
                )
                embedded_docs.append(hybrid_doc_model)

        return embedded_docs

    def embed_documents_for_dense_ingestion(
        self, raw_documents: List[RawDocumentModel]
    ) -> List[DenseIngestionDocumentModel]:
        embedded_docs: List[DenseIngestionDocumentModel] = []

        for doc in raw_documents:
            # Split into chunks
            splitted_docs = self.splitter_service.split(
                doc.article,
                {
                    "author": doc.author,
                    "link": doc.link,
                    "hash": doc.hash,
                    "text": doc.article,
                    "title": doc.title,
                },
            )

            chunks = [c["text"] if isinstance(c, dict) else c for c in splitted_docs]

            for chunk in chunks:
                dense_embeddings = self.embed_chunk_dense(chunk)
                dense_doc_model = DenseIngestionDocumentModel(
                    metadata=doc,
                    dense_vectors=dense_embeddings
                )
                embedded_docs.append(dense_doc_model)

        return embedded_docs
