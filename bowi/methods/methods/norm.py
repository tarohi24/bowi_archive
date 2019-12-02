"""
Text Similarity Estimation Based on Word Embeddings and
MatrixNorms for Targeted Marketing. NAACL. 2019
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dataclasses_jsonschema import JsonSchemaMixin
import numpy as np

from bowi.elas.search import EsResult
from bowi.embedding.base import return_matrix
from bowi.embedding.fasttext import FastText
from bowi.methods.base import Method, Param
from bowi.models import RankItem, QueryDocument
from bowi.text import Filter, TextProcessor


@dataclass
class NormParam(Param, JsonSchemaMixin):
    n_words: int  # used for pre-filtering
    norm: str


@dataclass
class Norm(Method):
    param: NormParam
    fasttext: FastText = field(default_factory=FastText.create)

    @classmethod
    def method_name(cls) -> str:
        return 'norm'

    @return_matrix
    def embed_words(self,
                    words: List[str]) -> np.ndarray:
        return np.array([self.fasttext.embed(w) for w in words])

    def norm(self,
             A: np.ndarray,
             B: np.ndarray) -> float:
        assert A.shape[1] == B.shape[1]
        aa: float = np.linalg.norm(np.dot(A, A.T), ord=self.param.norm)
        bb: float = np.linalg.norm(np.dot(B, B.T), ord=self.param.norm)
        ab: float = np.linalg.norm(np.dot(A, B.T), ord=self.param.norm)
        return ab / np.sqrt(aa * bb)

    def apply(self,
              query_doc: QueryDocument,
              size: int = 100) -> RankItem:
        cands: EsResult = self.filter_by_terms(query_doc=query_doc,
                                               n_words=self.param.n_words,
                                               size=size)

        filters: List[Filter] = self.get_default_filtes(
            n_words=self.param.n_words)
        processor: TextProcessor = TextProcessor(filters=filters)
        tokens_dict: Dict[Tuple[str, str], List[str]] = {
            hit.get_id_and_tag(): processor.apply(hit.source['text'])
            for hit in cands.hits
        }
        q_matrix: np.ndarray = self.embed_words(processor.apply(query_doc.text))

        scores: Dict[Tuple[str, str], float] = dict()
        for key, words in tokens_dict.items():
            mat: np.ndarray = self.embed_words(words)
            scores[key] = -self.norm(mat, q_matrix)

        return RankItem(query_id=query_doc.docid, scores=scores)
