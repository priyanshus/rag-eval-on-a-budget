# splitters/base.py
from typing import List, Dict


class BaseSplitter:
    def split(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Returns:
        [
          {
            "text": chunk_text,
            "metadata": {...}
          }
        ]
        """
        raise NotImplementedError
