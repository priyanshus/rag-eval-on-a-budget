from typing import List, Dict, Any

from qdrant_client import QdrantClient


class DenseVectorRetrievalService:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "articles_dense_collection",
    ):
        self.client = client
        self.collection_name = collection_name

    def similarity_search(
        self,
        query_vector: List[float],
        k: int = 3,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,   # dense vector
            using="dense",
            limit=k,
            score_threshold=score_threshold,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = point.payload

            results.append({
                "id": point.id,
                "score": point.score,
                "text": payload.get("text"),
                "metadata": {
                    key: value
                    for key, value in payload.items()
                    if key != "text"
                },
            })

        return results


