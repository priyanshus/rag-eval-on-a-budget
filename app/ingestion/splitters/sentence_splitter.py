from typing import List

import spacy

from .base import BaseSplitter


class SentenceSplitter(BaseSplitter):
    def __init__(self, max_sentences: int = 5):
        self.max_sentences = max_sentences
        self.nlp = spacy.load("en_core_web_sm")

    def split(self, text: str) -> List[str]:
        doc = self.nlp(text)
        sentences = [s.text.strip() for s in doc.sents]

        chunks = []
        for i in range(0, len(sentences), self.max_sentences):
            chunk = " ".join(sentences[i:i + self.max_sentences])
            chunks.append(chunk)

        return chunks
