import numpy as np
from typing import Any

class FTMock:
    dim: int = ...
    def embed(self, word: str) -> np.ndarray: ...
    def embed_words(self, words: str) -> np.ndarray: ...
    def __init__(self, dim: Any) -> None: ...