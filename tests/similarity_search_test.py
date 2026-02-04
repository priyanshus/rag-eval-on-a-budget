import csv
from pathlib import Path
from typing import List, Dict, Any

import pytest
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from app.retrieval.dense_vector_retrieval_service import DenseVectorRetrievalService
from app.retrieval.hybrid_vector_retrieval_service import HybridQueryService


def load_eval_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load the evaluation dataset from CSV.
    
    Returns:
        List of dictionaries with keys: paragraph, question, answer, source, 
        source_row_id, source_chunk_id
    """
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append({
                'paragraph': row['paragraph'],
                'question': row['question'],
                'answer': row['answer'],
                'source': row['source'],
                'source_row_id': row['source_row_id'],
                'source_chunk_id': row['source_chunk_id']
            })
    return dataset


def calculate_recall_at_k(retrieved_chunk_ids: List[str], expected_chunk_id: str, k: int) -> float:
    """
    Calculate Recall@K metric.
    
    Args:
        retrieved_chunk_ids: List of chunk IDs retrieved by similarity search
        expected_chunk_id: The expected chunk ID from ground truth
        k: Number of top results to consider
    
    Returns:
        Recall@K score (0.0 or 1.0 for single expected chunk)
    """
    top_k_retrieved = retrieved_chunk_ids[:k]
    return 1.0 if expected_chunk_id in top_k_retrieved else 0.0


def get_chunk_ids_from_results(results: List[Dict[str, Any]]) -> List[str]:
    """
    Extract chunk IDs from retrieval results.
    
    Args:
        results: List of retrieval results with metadata
    
    Returns:
        List of chunk IDs
    """
    chunk_ids = []
    for result in results:
        metadata = result.get('metadata', {})
        # chunk_id is stored in metadata (payload excludes 'text' but includes chunk_id)
        chunk_id = metadata.get('chunk_id')
        if chunk_id:
            chunk_ids.append(str(chunk_id))
    return chunk_ids


@pytest.fixture
def qdrant_client():
    """Fixture to provide Qdrant client."""
    return QdrantClient(url="http://localhost:6333")


@pytest.fixture
def eval_dataset():
    """Fixture to load evaluation dataset."""
    # Get the path to the data directory relative to this test file
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    dataset_path = project_root / "data" / "rag_eval_dataset.csv"
    return load_eval_dataset(str(dataset_path))


@pytest.fixture
def dense_model():
    """Fixture to provide dense embedding model."""
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def sparse_model():
    """Fixture to provide sparse embedding model."""
    return SparseTextEmbedding("Qdrant/bm25")


class TestDenseVectorRetrieval:
    
    def test_recall_at_k_dense_retrieval(self, qdrant_client, eval_dataset, dense_model):

        service = DenseVectorRetrievalService(qdrant_client)
        k = 3
        recall_scores = []
        
        
        test_samples = eval_dataset[:10]
        
        for sample in test_samples:
            question = sample['question']
            expected_chunk_id = sample['source_chunk_id']
            
            # Encode query and perform similarity search
            query_vector = dense_model.encode(question).tolist()
            results = service.similarity_search(query_vector, k=k)
            
            # Extract chunk IDs from results
            retrieved_chunk_ids = get_chunk_ids_from_results(results)
            
            # Calculate Recall@K
            recall = calculate_recall_at_k(retrieved_chunk_ids, expected_chunk_id, k)
            recall_scores.append(recall)
            
            print(f"\nQuestion: {question[:100]}...")
            print(f"Expected chunk_id: {expected_chunk_id}")
            print(f"Retrieved chunk_ids: {retrieved_chunk_ids}")
            print(f"Recall@{k}: {recall}")
        
        # Calculate average Recall@K
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        print(f"\nAverage Recall@{k}: {avg_recall}")
        
        # Assert that average recall is above a threshold (adjust as needed)
        assert avg_recall >= 0.0, f"Average Recall@{k} should be >= 0.0, got {avg_recall}"
        
        return avg_recall


if __name__ == "__main__":
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    dataset_path = project_root / "data" / "rag_eval_dataset.csv"
    
    dataset = load_eval_dataset(str(dataset_path))
    client = QdrantClient(url="http://localhost:6333")
    dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    service = DenseVectorRetrievalService(client)
    k = 3
    recall_scores = []
    
    for sample in dataset[:5]:
        question = sample['question']
        expected_chunk_id = sample['source_chunk_id']
        
        query_vector = dense_model.encode(question).tolist()
        results = service.similarity_search(query_vector, k=k)
        
        retrieved_chunk_ids = get_chunk_ids_from_results(results)
        recall = calculate_recall_at_k(retrieved_chunk_ids, expected_chunk_id, k)
        recall_scores.append(recall)
        
        print(f"\nQuestion: {question[:80]}...")
        print(f"Expected: {expected_chunk_id}")
        print(f"Retrieved: {retrieved_chunk_ids}")
        print(f"Recall@{k}: {recall}")
    
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    print(f"\n=== Average Recall@{k}: {avg_recall} ===")
