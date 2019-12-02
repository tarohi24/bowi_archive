from dataclasses import dataclass

import numpy as np


@dataclass
class FTMock:
    dim: int = 300

    def embed(self, word: str) -> np.ndarray:
        return np.random.rand(self.dim)

    def embed_words(self, words: str) -> np.ndarray:
        return np.array([self.embed(w) for w in words])
