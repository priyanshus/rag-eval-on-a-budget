import uuid
from typing import List, Dict

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

from app.ingestion.models import ChunkedDocumentsOutput


class DenseBatchIngestor:
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
            )
            print(f"Dense collection created: {self.collection_name}")
        else:
            print(f"Collection already exists: {self.collection_name}")

    def batch_upsert(self, docs: List[ChunkedDocumentsOutput]):
        """
        Upload all chunks from ChunkedDocumentsOutput to a dense-only Qdrant collection
        """
        points: List[PointStruct] = []

        for doc_output in docs:
            for chunk in doc_output.chunks:
                if chunk.dense_vector is None:
                    raise ValueError(f"Chunk {chunk.chunk_id} has no dense_vector.")

                # Prepare payload: minimal metadata + source info
                payload: Dict = chunk.metadata.copy() if chunk.metadata else {}
                payload.update({
                    "source_row_id": chunk.source_row_id,
                    "chunk_id": chunk.chunk_id,
                    "chunk_text": chunk.chunk_text  # optional: include for RAG retrieval
                })

                point = PointStruct(
                    id=str(uuid.uuid4()),
                    payload=payload,
                    vector={"dense": chunk.dense_vector}
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
