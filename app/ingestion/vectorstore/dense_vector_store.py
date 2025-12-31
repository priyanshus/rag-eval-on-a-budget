import uuid
from dataclasses import asdict
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from app.ingestion.models import IngestionDocumentModel


class DenseBatchIngestor:
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
                }
            )
            print(f"Dense collection created: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def batch_upsert(self, docs: List[IngestionDocumentModel]):
        points: List[PointStruct] = []

        for doc in docs:
            payload = asdict(doc.metadata)

            point = PointStruct(
                id=str(uuid.uuid4()),
                payload=payload,
                vector={
                    "dense": doc.dense_vectors
                }
            )
            points.append(point)

        if points:
            self.client.upload_points(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            print(f"Uploaded {len(points)} documents to Qdrant collection '{self.collection_name}'")
        else:
            print("No points to upload.")
