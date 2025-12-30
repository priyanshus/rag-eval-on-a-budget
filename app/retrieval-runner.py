from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.retrieval.dense_vector_retrieval_service import DenseVectorRetrievalService
from app.retrieval.hybrid_vector_retrieval_service import HybridQueryService

if __name__ == '__main__':
    query = "What probabilistic process allows scammers to produce a group of users?"
    client = QdrantClient(url="http://localhost:6333")

    # 1. Retrieve only from dense vector collection
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Ensure flat vector
    query_dense = dense_model.encode(query).tolist()

    service = DenseVectorRetrievalService(client)
    matches = service.similarity_search(query_dense)

    for match in matches:
        print(f"Score: {match['score']}")
        print(f"Text: {match['text']}")
        print(f"Metadata: {match['metadata']}")
        print("-" * 40)

    #2. Retrieve from hybrid collection
    print("-" * 10 + "Hybrid Retrieval" + "-" * 10 )
    sparse_model = SparseTextEmbedding("Qdrant/bm25")


    # Dense vector
    query_dense = dense_model.encode(query)

    # Sparse vector
    sparse_vec = next(sparse_model.embed(query))
    query_sparse = {"indices": sparse_vec.indices, "values": sparse_vec.values}

    service = HybridQueryService(client, "articles_hybrid_collection")

    results = service.similarity_search(query_dense, query_sparse, k=5)

    for match in results:
        print(f"Score: {match['score']}")
        print(f"Text: {match['text']}")
        print(f"Metadata: {match['metadata']}")
        print("----------------------------")