from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import Prefetch, FusionQuery, Fusion
from qdrant_client.models import SparseVector


class HybridQueryService:
    def __init__(self, client: QdrantClient, collection_name="articles_hybrid_collection"):
        self.client = client
        self.collection_name = collection_name

    def similarity_search(
            self,
            query_dense: List[float],
            query_sparse: Dict[str, List[float]],
            k: int = 5,
            dense_limit: int = 20,
            sparse_limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid search via RRF fusion of dense + sparse prefetch queries.
        """
        # Build sparse vector
        sparse_vector = SparseVector(
            indices=query_sparse["indices"],
            values=query_sparse["values"]
        )

        # Prefetch for sparse BM25
        prefetch_sparse = Prefetch(
            query=sparse_vector,
            using="bm25",
            limit=sparse_limit
        )

        # Prefetch for dense semantic
        prefetch_dense = Prefetch(
            query=query_dense,
            using="dense",
            limit=dense_limit
        )

        # Execute the hybrid query
        results = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[prefetch_sparse, prefetch_dense],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=k,
            score_threshold=0.5,
            with_payload=True
        )

        return [
            {
                "id": point.id,
                "score": point.score,
                "text": point.payload.get("text"),
                "metadata": {
                    k: v
                    for k, v in point.payload.items()
                    if k != "text"
                },
            }
            for point in results.points
        ]
