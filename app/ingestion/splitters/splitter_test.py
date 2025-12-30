from app.ingestion.splitters.char_splitter import CharSplitter
from app.ingestion.splitters.recursive_splitter import RecursiveSplitter
from app.ingestion.splitters.sentence_splitter import SentenceSplitter
from app.ingestion.splitters.token_splitter import TokenSplitter

if __name__ == '__main__':
    with open("data.txt", "r", encoding="utf-8") as f:
        text = f.read()

    metadata = {"source": "article_12"}

    splitters = {
        "token": TokenSplitter(),
        "recursive": RecursiveSplitter(),
        "sentence": SentenceSplitter(),
        "character": CharSplitter(),
    }

    for name, splitter in splitters.items():
        chunks = splitter.split(text, metadata)
        print(f"{name}: {len(chunks)} chunks ")
        print("*****")