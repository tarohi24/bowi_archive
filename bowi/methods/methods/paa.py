"""
Principal Angle Analysis
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dataclasses_jsonschema import JsonSchemaMixin
import numpy as np
import scipy

from bowi.elas.search import EsResult
from bowi.embedding.base import return_matrix
from bowi.embedding.fasttext import FastText
from bowi.methods.base import Method, Param
from bowi.models import RankItem, QueryDocument
from bowi.text import Filter, TextProcessor


@dataclass
class PAAParam(Param, JsonSchemaMixin):
    n_words: int


@dataclass
class PAA(Method):
    param: PAAParam
    fasttext: FastText = field(default_factory=FastText.create)

    @return_matrix
    def embed_words(self,
                    words: List[str]) -> np.ndarray:
        return np.array([self.fasttext.embed(w) for w in words])

    def paa(self,
            A: np.ndarray,
            B: np.ndarray) -> float:
        """
        Projection matrix
        """
        angles: np.ndarray = np.radians(scipy.linalg.subspace_angles(A.T, B.T))
        return np.sqrt(300 - np.sum(np.cos(angles) ** 2))

    def apply(self,
              query_doc: QueryDocument,
              size: int = 100) -> RankItem:
        filters: List[Filter] = self.get_default_filtes(
            n_words=self.param.n_words)
        processor: TextProcessor = TextProcessor(filters=filters)
        cands: EsResult = self.filter_by_terms(
            query_doc=query_doc,
            n_words=self.param.n_words,
            size=size)

        tokens_dict: Dict[Tuple[str, str], List[str]] = {
            hit.get_id_and_tag(): processor.apply(hit.source['text'])
            for hit in cands.hits
        }

        q_matrix: np.ndarray = self.embed_words(processor.apply(query_doc.text))

        scores: Dict[Tuple[str, str], float] = dict()
        for key, words in tokens_dict.items():
            mat: np.ndarray = self.embed_words(words)
            score: float = self.paa(mat, q_matrix)
            scores[key] = score

        return RankItem(query_id=query_doc.docid, scores=scores)
