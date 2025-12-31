from typing import List

from fastembed import SparseTextEmbedding, SparseEmbedding
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from app.ingestion.models import ChunkedDocumentsOutput


class EmbeddingService:
    def __init__(
            self,
            dense_model: str,
            sparse_model: str = "Qdrant/bm25",
    ):
        self.dense_model = SentenceTransformer(dense_model)
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model)

    def embed_chunks_dense(self, docs: List[str]) -> ndarray:
        return self.dense_model.encode(docs, show_progress_bar=True)

    def embed_chunks_sparse(self, docs: List[str]) -> List[SparseEmbedding]:
        return list(self.sparse_model.embed(docs))

    def embed_documents_for_hybrid_ingestion(
            self, chunked_docs: List[ChunkedDocumentsOutput]
    ) -> List[ChunkedDocumentsOutput]:
        """
        For each ChunkedDocumentsOutput, compute dense and sparse embeddings
        and store them inside each ChunkedDocumentModel.
        """
        embedded_documents: List[ChunkedDocumentsOutput] = []

        for doc_output in chunked_docs:
            chunk_texts = [chunk.chunk_text for chunk in doc_output.chunks]

            # Compute embeddings
            dense_embeddings = self.embed_chunks_dense(chunk_texts)
            sparse_embeddings = self.embed_chunks_sparse(chunk_texts)

            # Update each chunk with embeddings
            for chunk, dense_vec, sparse_vec in zip(
                    doc_output.chunks, dense_embeddings, sparse_embeddings
            ):
                chunk.dense_vector = dense_vec.tolist()
                chunk.sparse_vector = {
                    "indices": sparse_vec.indices.tolist() if hasattr(sparse_vec.indices, "tolist") else list(
                        sparse_vec.indices),
                    "values": sparse_vec.values.tolist() if hasattr(sparse_vec.values, "tolist") else list(
                        sparse_vec.values)
                }
            # or store indices+values if needed

            embedded_documents.append(doc_output)

        return embedded_documents

    def embed_documents_for_dense_ingestion(
            self, chunked_docs: List[ChunkedDocumentsOutput]
    ) -> List[ChunkedDocumentsOutput]:
        """
        Compute only dense embeddings for ingestion.
        """
        embedded_documents: List[ChunkedDocumentsOutput] = []

        for doc_output in chunked_docs:
            chunk_texts = [chunk.chunk_text for chunk in doc_output.chunks]

            dense_embeddings = self.embed_chunks_dense(chunk_texts)

            # Update chunks with dense vectors
            for chunk, dense_vec in zip(doc_output.chunks, dense_embeddings):
                chunk.dense_vector = dense_vec.tolist()
                chunk.sparse_vector = None

            embedded_documents.append(doc_output)

        return embedded_documents
