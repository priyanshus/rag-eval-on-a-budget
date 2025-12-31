from typing import Dict, List

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.retrieval.dense_vector_retrieval_service import DenseVectorRetrievalService
from app.retrieval.hybrid_vector_retrieval_service import HybridQueryService


class RetrievalRunner:
    def __init__(self, client: QdrantClient, query: str, k=3):
        self._client = client
        self._query = query
        self._k = k
        self._dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def _fetch_similarity_result_using_dense_vectors_only(self) -> List[Dict]:
        service = DenseVectorRetrievalService(self._client)
        query_dense = self._dense_model.encode(self._query).tolist()
        matches = service.similarity_search(query_dense)
        # for match in matches:
        #     print(f"Vector Store: Dense")
        #     print(f"Score: {match['score']}")
        #     print(f"Text: {match['metadata']['chunk_text']}")
        #     print(f"Metadata: {match['metadata']['source']}")
        #     print("-" * 40)

        return matches

    def _fetch_similarity_result_using_hybrid_vectors(self) -> List[Dict]:
        query_dense = self._dense_model.encode(self._query).tolist()
        sparse_model = SparseTextEmbedding("Qdrant/bm25")

        # Dense vector
        query_dense = self._dense_model.encode(self._query).tolist()

        # Sparse vector
        sparse_vec = next(sparse_model.embed(self._query))
        query_sparse = {"indices": sparse_vec.indices, "values": sparse_vec.values}

        service = HybridQueryService(self._client)

        results = service.similarity_search(query_dense=query_dense, query_sparse=query_sparse, k=5)

        # for match in results:
        #     print(f"Vector Store: Hybrid")
        #     print(f"Score: {match['score']}")
        #     print(f"Text: {match['metadata']['chunk_text']}")
        #     print(f"Metadata: {match['metadata']['source']}")
        #     print("----------------------------")

        return results

    def fetch_similarity_result(self) -> List[Dict]:
        dense_results = self._fetch_similarity_result_using_dense_vectors_only()
        hybrid_results = self._fetch_similarity_result_using_hybrid_vectors()

        all_results = dense_results + hybrid_results

        # Sort by score descending
        top_results = sorted(
            all_results,
            key=lambda match: match.get("score", 0),
            reverse=True
        )

        # Return top 3
        return top_results[:3]


if __name__ == '__main__':
    query = "How does the conceptual metaphor “ARGUMENT IS WAR” shape the way people think about arguments?"
    client = QdrantClient(url="http://localhost:6333")

    retrieval_runner = RetrievalRunner(query=query, client=client)
    results = retrieval_runner.fetch_similarity_result()
    for match in results:
        print(f"Vector Store: Hybrid")
        print(f"Score: {match['score']}")
        print(f"Text: {match['metadata']['chunk_text']}")
        print(f"Metadata: {match['metadata']['source']}")
        print("----------------------------")
