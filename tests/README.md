# RAG Evaluation Tests

This directory contains test suites for evaluating the RAG (Retrieval-Augmented Generation) system's similarity search and retrieval performance using non-LLM metrics.

## Overview

The tests evaluate retrieval quality by comparing search results against ground truth data from `rag_eval_dataset.csv`. All metrics used are **non-LLM based**, making them fast, cost-effective, and deterministic.

## Test Files

### `similarity_search_test.py`

Tests for similarity search evaluation using retrieval metrics.

**Metrics Implemented:**
- **Recall@K**: Measures whether the expected chunk appears in the top K retrieved results
  - Returns 1.0 if the expected chunk is found in top K results, 0.0 otherwise
  - Calculates average Recall@K across multiple test samples

**Test Cases:**
- `test_recall_at_k_dense_retrieval`: Evaluates dense vector retrieval service using Recall@K metric
  - Tests on first 10 samples from evaluation dataset
  - Uses `DenseVectorRetrievalService` for similarity search
  - Reports per-sample and average Recall@K scores

**Usage:**
```bash
# Run with pytest
pytest tests/similarity_search_test.py -v

# Run directly (standalone script)
python tests/similarity_search_test.py
```

### `context_precision_test.py`

Tests for context precision evaluation using RAGAS metrics.

**Metrics:**
- `NonLLMContextPrecisionWithReference`: Non-LLM based context precision metric from RAGAS
- `ContextPrecision`: LLM-based context precision (for reference)

**Note:** This file contains example code for using RAGAS metrics and may need to be completed with proper test cases.

## Ground Truth Data

All tests use `data/rag_eval_dataset.csv` as the ground truth dataset. The CSV contains:

- `paragraph`: Source paragraph text
- `question`: Question generated from the paragraph
- `answer`: Expected answer
- `source`: Source URL or identifier
- `source_row_id`: Reference to source document row
- `source_chunk_id`: Expected chunk ID that should be retrieved for the question

## Prerequisites

### Services Required

1. **Qdrant Vector Database**: Must be running on `http://localhost:6333`
   ```bash
   # Using docker-compose (if available)
   docker-compose up -d
   ```

2. **Vector Collections**: The following collections must exist and be populated:
   - `articles_dense_collection`: For dense vector retrieval
   - `articles_hybrid_collection`: For hybrid retrieval (dense + sparse)

### Python Dependencies

All dependencies should be installed from `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies for tests:
- `pytest`: Test framework
- `qdrant_client`: Qdrant client library
- `sentence-transformers`: For dense embeddings
- `fastembed`: For sparse embeddings
- `ragas`: For evaluation metrics

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test File
```bash
pytest tests/similarity_search_test.py -v
```

### Run Specific Test Class
```bash
pytest tests/similarity_search_test.py::TestDenseVectorRetrieval -v
```

### Run with Output
```bash
pytest tests/similarity_search_test.py -v -s
```

The `-s` flag shows print statements for detailed output.

## Test Structure

### Helper Functions

**`load_eval_dataset(file_path)`**
- Loads evaluation dataset from CSV file
- Returns list of dictionaries with question-answer pairs and metadata

**`calculate_recall_at_k(retrieved_chunk_ids, expected_chunk_id, k)`**
- Calculates Recall@K metric
- Returns 1.0 if expected chunk is in top K, else 0.0

**`get_chunk_ids_from_results(results)`**
- Extracts chunk IDs from retrieval service results
- Handles metadata extraction from Qdrant payload

### Fixtures

- `qdrant_client`: Provides Qdrant client instance
- `eval_dataset`: Loads evaluation dataset
- `dense_model`: Provides sentence transformer model for dense embeddings
- `sparse_model`: Provides sparse embedding model

## Metrics Explained

### Recall@K

**Definition**: The fraction of relevant items that are successfully retrieved in the top K results.

**Formula**: 
```
Recall@K = 1 if expected_chunk_id in top_K_results else 0
```

**Interpretation**:
- **1.0**: Expected chunk found in top K results (perfect recall)
- **0.0**: Expected chunk not found in top K results (miss)

**Use Case**: Measures retrieval completeness - whether the system can find the relevant document chunk.

## Adding New Metrics

To add new evaluation metrics:

1. Create a calculation function similar to `calculate_recall_at_k()`
2. Add a new test method in the test class
3. Update this README with the new metric description

**Example metrics to add:**
- **Precision@K**: Fraction of retrieved items that are relevant
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant result
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Hit Rate**: Fraction of queries with at least one relevant result in top K

## Troubleshooting

### Qdrant Connection Error
```
Error: Could not connect to Qdrant at http://localhost:6333
```
**Solution**: Ensure Qdrant is running and accessible. Check with:
```bash
curl http://localhost:6333/collections
```

### Empty Retrieval Results
If tests return empty results, ensure:
1. Vector collections are created and populated
2. Collection names match those in the test code
3. Chunk IDs in the dataset match those in the vector store

### Missing Chunk IDs
If chunk IDs are not found in results:
- Verify the metadata structure in Qdrant matches expected format
- Check that `chunk_id` is stored in the payload metadata
- Ensure `get_chunk_ids_from_results()` correctly extracts IDs

## Future Enhancements

- [ ] Add Precision@K metric
- [ ] Add MRR (Mean Reciprocal Rank) metric
- [ ] Add NDCG@K metric
- [ ] Add tests for hybrid vector retrieval service
- [ ] Add batch evaluation across full dataset
- [ ] Add performance benchmarking
- [ ] Add visualization of metric trends
