from typing import List

from fastembed import SparseTextEmbedding, SparseEmbedding
from numpy import ndarray
from sentence_transformers import SentenceTransformer

from app.ingestion.models import IngestionDocumentModel, ChunkedDocumentModel


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

    def embed_chunks_sparse(self, docs: List[str]) -> list[SparseEmbedding]:
        return list(self.sparse_model.embed(docs))

    def embed_documents_for_hybrid_ingestion(
            self, chunked_docs: List[ChunkedDocumentModel]
    ) -> List[IngestionDocumentModel]:
        embedded_docs: List[IngestionDocumentModel] = []

        for doc in chunked_docs:
            chunked_text = [chunk.chunked_text for chunk in doc.chunks]

            # Dense embeddings
            dense_embeddings = self.embed_chunks_dense(chunked_text)

            # Sparse embeddings
            sparse_embeddings = self.embed_chunks_sparse(chunked_text)

            for dense_vec, sparse_vec in zip(dense_embeddings, sparse_embeddings):
                hybrid_doc_model = IngestionDocumentModel(
                    metadata=doc.metadata,
                    dense_vectors=dense_vec,
                    bm25_vectors={
                        "indices": sparse_vec.indices,
                        "values": sparse_vec.values,
                    },
                )
                embedded_docs.append(hybrid_doc_model)

        return embedded_docs

    def embed_documents_for_dense_ingestion(
            self, raw_documents: List[ChunkedDocumentModel]
    ) -> List[IngestionDocumentModel]:
        embedded_docs: List[IngestionDocumentModel] = []

        for doc in raw_documents:
            chunked_text = [chunk.chunked_text for chunk in doc.chunks]

            dense_embeddings = self.embed_chunks_dense(chunked_text)

            for dense_vec in dense_embeddings:
                ingestion_doc_model = IngestionDocumentModel(
                    metadata=doc.metadata,
                    dense_vectors=dense_vec,
                    bm25_vectors={}
                )

                embedded_docs.append(ingestion_doc_model)

        return embedded_docs
