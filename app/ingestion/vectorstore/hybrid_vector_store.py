import uuid
from typing import List
from dataclasses import asdict

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, PointStruct, SparseVector

from app.ingestion.models import HybridIngestionDocumentModel


class HybridBatchIngestor:
    def __init__(
        self,
        collection_name: str,
        client: QdrantClient,
        dense_vector_size: int = 384
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

    def batch_upsert(self, docs: List[HybridIngestionDocumentModel]):
        points: List[PointStruct] = []

        for doc in docs:
            payload = asdict(doc.metadata)

            if isinstance(doc.bm25_vectors, dict) and "indices" in doc.bm25_vectors and "values" in doc.bm25_vectors:
                bm25_vector = SparseVector(
                    indices=doc.bm25_vectors["indices"],
                    values=doc.bm25_vectors["values"]
                )
            else:
                raise ValueError("bm25_vectors must be a dict with 'indices' and 'values'.")

            point = PointStruct(
                id=str(uuid.UUID(doc.metadata.hash[:32])),
                payload=payload,
                vector={
                    "dense": doc.dense_vectors,
                    "bm25": bm25_vector
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"Uploaded {len(points)} documents to Qdrant collection '{self.collection_name}'")
