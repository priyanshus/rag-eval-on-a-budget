import uuid
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    PointStruct,
    SparseVector,
)

from app.ingestion.models import ChunkedDocumentsOutput


class HybridBatchIngestor:
    def __init__(
            self,
            collection_name: str,
            client: QdrantClient,
            dense_vector_size: int = 384,
    ):
        self.client = client
        self.collection_name = collection_name
        self.dense_vector_size = dense_vector_size

    def create_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_vector_size,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams()
                },
            )
            print(f"Hybrid collection created: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def batch_upsert(self, docs: List[ChunkedDocumentsOutput]):
        """
        Upload all chunks from multiple ChunkedDocumentsOutput to a hybrid Qdrant collection
        """
        points: List[PointStruct] = []

        for doc_output in docs:
            for chunk in doc_output.chunks:
                # Prepare payload
                payload: Dict = chunk.metadata.copy() if chunk.metadata else {}
                payload.update({
                    "source_row_id": chunk.source_row_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.chunk_text
                })

                bm25_vector: SparseVector = None
                if chunk.sparse_vector:
                    if isinstance(chunk.sparse_vector, dict):
                        bm25_vector = SparseVector(
                            indices=chunk.sparse_vector.get("indices", []),
                            values=chunk.sparse_vector.get("values", [])
                        )
                    else:
                        raise ValueError(
                            f"Chunk {chunk.chunk_id} has invalid sparse_vector format."
                        )

                # Validate dense vector
                if not chunk.dense_vector:
                    raise ValueError(f"Chunk {chunk.chunk_id} has no dense_vector.")

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    payload=payload,
                    vector={
                        "dense": chunk.dense_vector,
                        "bm25": bm25_vector
                    }
                )
                points.append(point)

        if points:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points,
                parallel=3,
                wait=True
            )
            print(f"Uploaded {len(points)} chunks to Qdrant collection '{self.collection_name}'")
        else:
            print("No points to upload.")
