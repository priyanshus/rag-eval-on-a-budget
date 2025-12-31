from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.generation.llm_client import LLMClient
from app.retrieval.hybrid_vector_retrieval_service import HybridQueryService

if __name__ == '__main__':
    load_dotenv()
    llm = LLMClient(
        model="openai/gpt-4o-mini"
    )

    question = "What are the reasons for failure of Rag evaluation?"
    client = QdrantClient(url="http://localhost:6333")

    # 1. Retrieve only from dense vector collection
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sparse_model = SparseTextEmbedding("Qdrant/bm25")

    # Dense vector
    query_dense = dense_model.encode(question)

    # Sparse vector
    sparse_vec = next(sparse_model.embed(question))
    query_sparse = {"indices": sparse_vec.indices, "values": sparse_vec.values}

    service = HybridQueryService(client, "articles_hybrid_collection")

    results = service.similarity_search(query_dense, query_sparse, k=5)

    # for match in results:
    #     print(f"Score: {match['score']}")
    #     print(f"Text: {match['text']}")
    #     print(f"Metadata: {match['metadata']}")

    reply = llm.generate(question=question, context_chunks=results)

    print("-" * 80)
    print(f"Question: {question}")
    print("-" * 80)
    print(f"System Reply: {reply}")
    print("-" * 80)
