from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

import fasttext
import numpy as np

from bowi.embedding.base import Model, mat_normalize
from bowi.settings import models_dir
                                  

@dataclass
class FastText(Model):
    model: fasttext.FastText._FastText = field(init=False)

    def __post_init__(self):
        self.model = fasttext.load_model(
            str(models_dir.joinpath('fasttext/cc.en.300.bin').resolve()))
        self.dim: int = 300

    def isin_vocab(self, word: str) -> bool:
        return len(self.model.get_subwords(word)[0]) > 0

    def filter_tokens(self, words: List[str]) -> List[str]:
        return [w for w in words if self.isin_vocab(w)]

    def embed(self,
              word: str) -> np.ndarray:
        """
        You should filter words not in the vocab before you use this.
        """
        def embed(word: str) -> np.ndarray:
            """
            Isolated from the main part because FastText object is not
            hashable, which means we cannot use lru_cache.
            """
            if self.isin_vocab(word):
                return self.model.get_word_vector(word)
            else:
                raise RuntimeError(f'{word} is not in the model vocab.')
        return embed(word)

    def embed_words(self,
                    words: List[str],
                    normalize: bool = True,
                    ignore_missing: bool = True) -> np.ndarray:
        embs: List[np.ndarray] = [self.embed(w) for w in words
                                  if (not ignore_missing) or self.isin_vocab(w)]
        if normalize:
            return mat_normalize(np.array(embs))
        else:
            return np.array(embs)
