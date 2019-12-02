from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional

import fasttext
import numpy as np

from bowi.embedding.base import Model, return_matrix, return_vector
from bowi.settings import models_dir


@dataclass
class FastText(Model):
    model: fasttext.FastText._FastText = field(init=False)

    def __post_init__(self):
        self.model = fasttext.load_model(
            str(models_dir.joinpath('fasttext/cc.en.300.bin').resolve()))
        self.dim: int = 300

    @return_vector
    def embed(self, word: str) -> np.ndarray:
        return self.model.get_word_vector(word)

    def embed_words(self,
                    words: List[str]) -> List[Optional[np.ndarray]]:
        emb_caches: Dict[str, np.ndarray] = dict()
        unknown_words: Set[str] = set()
        for w in words:
            if w in unknown_words:
                continue
            if w not in emb_caches:
                embedding: np.ndarray = self.embed(w)
                if np.linalg.norm(embedding) > 0:
                    emb_caches[w] = embedding
                else:
                    unknown_words.add(w)
        return [emb_caches[w] if w not in unknown_words else None for w in words]
