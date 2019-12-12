from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Set

import fasttext
import numpy as np

from bowi.embedding.base import Model
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

    def embed(self, word: str) -> np.ndarray:
        """
        You should filter words not in the vocab before you use this.
        """
        if self.isin_vocab(word):
            return self.model.get_word_vector(word)
        else:
            raise RuntimeError(f'{word} is not in the model vocab.')

    def embed_words(self,
                    words: List[str]) -> np.ndarray:
        emb_dic: Dict[str, np.ndarray] = defaultdict(lambda w: self.embed(w))
        embs: List[np.ndarray] = [emb_dic[w] for w in words]
        return np.array(embs)
