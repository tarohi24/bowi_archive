"""
Improved BM25
"""
from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, List, Pattern, Type  # type: ignore

from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param
from bowi.methods.common.cache import KNNCacher, DFCacher


@dataclass
class BM25IParam(Param):
    n_words: int
    threshold: float = 0.5  # note that threshold must be above 0.5
    k1: float = 1.2  # default in Elasticsearch
    b: float = 0.75  # default in Elasticsearch


@dataclass
class BM25I(Method[BM25IParam]):
    param_type: ClassVar[Type] = BM25IParam
    knn_cacher: KNNCacher = field(init=False)
    df_cacher: DFCacher = field(init=False)
    avgdl: float = field(init=False)

    def __post_init__(self):
        self.knn_cacher = KNNCacher(dataset=self.context.es_index)
        self.df_cacher = DFCacher(dataset=self.context.es_index)
        self.avgdl: float = {
            'clef': 3804.996
        }[self.context.es_index]

    def bm25i(self,
              query: Dict[str, int],
              collection: Dict[str, int]) -> float:
        """
        Parameters
        -----
        query
            {word: tf}
        """
        score: float = 0
        col_dl_factor: float = sum(collection.values()) / self.avgdl
        for q, tf in query.items():
            simwords: Dict[str, float] = self.knn_cacher(q, include_self=True)
            for word, dist in simwords.items():
                idf: float = self.df_cacher.get_idf(word)
                sim: float = 1 - dist
                numerator: float = sim * idf * tf * (self.k1 + 1)
                denominator: float = tf * self.k1 * (1 - self.b + self.b * col_dl_factor)
                score += (numerator / denominator)
        return score
