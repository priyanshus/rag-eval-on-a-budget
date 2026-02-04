import csv
import json

from qdrant_client import QdrantClient

from app.generation.llm_client import LLMClient


def build_message(paragraph: str):
    messages = []

    messages.append(
        {
            "role": "system",
            "content": """You generate RAG evaluation data. Return ONLY valid JSON. No explanations or markdown. Make sure you follow these rule to generate question and answer pair.
                 1.Questions must allow multiple reasonable answers justified by the paragraph
                 2.Answers must cite paragraph content
                 3.Do not copy sentences verbatim
                 4.Output MUST be valid JSON
                 5.Json schema to be followed: {"qa_pairs": [{"question": "", "answer": ""}]}
            """
        }
    )

    messages.append(
        {
            "role": "system",
            "content": f"Pargraph : {paragraph}",
        }
    )

    return messages


def generate_question_from_vector_store(max_docs=25):
    qdrant_client = QdrantClient(url="http://localhost:6333")
    offset = None
    chunks = []

    while len(chunks) < max_docs:
        points, offset = qdrant_client.scroll(
            collection_name="articles_hybrid_collection",
            limit=min(100, max_docs - len(chunks)),
            offset=offset,
            with_payload=True,
        )

        if not points:
            break

        for point in points:
            text = point.payload.get("chunk_text")
            chunk_id = point.payload.get("chunk_id")
            source_row_id = point.payload.get("source_row_id")
            source = point.payload.get("source")

            if not text or len(text) < 200:
                continue

            chunks.append({
                "text": text,
                "chunk_id": chunk_id,
                "source_row_id": source_row_id,
                "source": source

            })

            if len(chunks) >= max_docs:
                break

        if offset is None:
            break

    return chunks


if __name__ == '__main__':

    paragraphs = generate_question_from_vector_store()
    llm = LLMClient(model="openai/gpt-4o-mini", max_tokens=500)

    # print(paragraphs.__len__())
    # print(rendered_prompt)
    with open("../../data/rag_eval_dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["paragraph", "question", "answer", "source", "source_row_id", "source_chunk_id"])

        for paragraph in paragraphs:

            messages = build_message(paragraph['text'])
            response = llm.generate_questions(messages)

            try:
                data = json.loads(response)
                qa_pairs = data.get("qa_pairs", [])
            except json.JSONDecodeError:
                continue  # skip bad generations (log in real systems)

            for qa in qa_pairs:
                writer.writerow([
                    paragraph['text'],
                    qa.get("question", ""),
                    qa.get("answer", ""),
                    paragraph['source'],
                    paragraph['source_row_id'],
                    paragraph['chunk_id']
                ])
