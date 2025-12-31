# splitters/base.py
from typing import List


class BaseSplitter:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError
