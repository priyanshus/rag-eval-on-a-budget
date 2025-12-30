from typing import List, Dict
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct


class QdrantIngestionService:

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "articles_collection",
    ):
        self.url = url
        self.collection_name = collection_name
        self._client = QdrantClient(url=url)
        self._create_collection()


    def _create_collection(self):
        if not self._client.collection_exists(self.collection_name):
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "size": 384,
                    "distance": "Cosine",
                },
            )

    def batch_upsert(self, embeddings: List[List[float]], metadata: Dict):
        """
        Insert multiple embeddings sharing the same metadata.

        embeddings: list of vectors
        metadata: dict to attach to all vectors
        """
        points = [
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=metadata
            )
            for vector in embeddings
        ]

        self._client.upsert(
            collection_name=self.collection_name,
            points=points
        )
