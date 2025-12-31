import hashlib
import json
from pathlib import Path
from typing import List

import pandas as pd

from app.ingestion.models import RawDocument


class CsvLoaderService:
    def __init__(self, file_path: str):
        self.csv_path = Path(file_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

    def _row_to_hash(self, row):
        normalized = json.dumps(
            row.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":")
        )
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def csv_reader(self) -> List[RawDocument]:
        df = pd.read_csv(self.csv_path)
        documents: List[RawDocument] = []

        for idx, row in df.iterrows():
            documents.append(
                RawDocument(
                    title=row.get("title") if pd.notna(row.get("title")) else None,
                    author=row.get("author") if pd.notna(row.get("author")) else None,
                    source=row.get("link") if pd.notna(row.get("link")) else None,
                    text=row.get("text"),
                    hash=self._row_to_hash(row),
                    row_id=int(idx)
                )
            )

        return documents
