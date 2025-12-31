from dotenv import load_dotenv
from qdrant_client import QdrantClient

from app.generation.llm_client import LLMClient
from app.retrieval_runner import RetrievalRunner

if __name__ == '__main__':
    load_dotenv()
    llm = LLMClient(
        model="openai/gpt-4o-mini",
        max_tokens=10000
    )

    question = "How does the conceptual metaphor “ARGUMENT IS WAR” shape the way people think about arguments? How do they correlated it with ARGUMENT IS A DANCE?"
    client = QdrantClient(url="http://localhost:6333")
    runner = RetrievalRunner(client=client, query=question)

    results = runner.fetch_similarity_result()
    reply = llm.generate(question=question, context_chunks=results)

    print("-" * 80)
    print(f"Question: {question}")
    print("-" * 80)
    print(f"System Reply: {reply}")
    print("-" * 80)
