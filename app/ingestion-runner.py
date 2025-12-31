from pathlib import Path

from qdrant_client import QdrantClient

from app.ingestion.dataloader.csv_loader_service import CsvLoaderService
from app.ingestion.embedding.embedding_service import EmbeddingService
from app.ingestion.splitters.recursive_splitter import RecursiveSplitter
from app.ingestion.splitters.splitter_service import SplitterService
from app.ingestion.vectorstore.dense_vector_store import DenseBatchIngestor
from app.ingestion.vectorstore.hybrid_vector_store import HybridBatchIngestor

if __name__ == '__main__':
    data_folder = Path(__file__).parent / "../data"
    input_doc_path = data_folder / "articles.csv"
    csv_loader_service = CsvLoaderService(input_doc_path)
    raw_csv_docs = csv_loader_service.csv_reader()
    qdrant_client = QdrantClient(url="http://localhost:6333")
    token_splitter = RecursiveSplitter()

    splitter_service = SplitterService(token_splitter)
    chunked_docs = splitter_service.chunk(raw_csv_docs)

    embedding_service = EmbeddingService(
        dense_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 1. Create hybrid collection
    print("Ingesting raw documents to hybrid vector store")
    hybrid_collection_name = "articles_hybrid_collection"
    hybrid_batch_ingestor = HybridBatchIngestor(hybrid_collection_name, client=qdrant_client)
    ingestion_ready_docs = embedding_service.embed_documents_for_hybrid_ingestion(chunked_docs)

    hybrid_batch_ingestor.create_collection()
    hybrid_batch_ingestor.batch_upsert(ingestion_ready_docs)

    # 2. Create dense collection
    print("Ingesting raw documents to dense only vector store")
    dense_collection_name = "articles_dense_collection"
    dense_batch_ingestor = DenseBatchIngestor(dense_collection_name, client=qdrant_client)
    ingestion_ready_docs = embedding_service.embed_documents_for_dense_ingestion(chunked_docs)

    dense_batch_ingestor.create_collection()
    dense_batch_ingestor.batch_upsert(ingestion_ready_docs)
