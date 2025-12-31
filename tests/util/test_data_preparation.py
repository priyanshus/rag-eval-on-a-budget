from qdrant_client import QdrantClient


def generate_question_from_vector_store():
    qdrant_client = QdrantClient(url="http://localhost:6333")
    offset = None

    while True:
        points, offset = qdrant_client.scroll(
            collection_name="articles_hybrid_collection",
            limit=100,
            offset=offset,
            with_payload=True,
        )

        if not points:
            break

        for point in points:
            text = point.payload.get("chunk_text")

            if not text or len(text) < 200:
                continue  # skip very small chunks

        if offset is None:
            break


if __name__ == '__main__':
    generate_question_from_vector_store()
