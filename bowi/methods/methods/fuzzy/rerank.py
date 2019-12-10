"""
Note: This module assserts keywords are generated
in advance. Run fuzzy.naive in the same params
before running this script
"""
from collections import Counter
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Type, Optional

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, LoaderNode

from bowi.embedding.base import mat_normalize
from bowi.embedding.fasttext import FastText
from bowi.methods.common.methods import Method
from bowi.methods.common.pre_filtering import load_cols
from bowi.methods.common.types import TRECResult
from bowi.elas.client import EsClient
from bowi import settings

from bowi.methods.methods.fuzzy.param import FuzzyParam


logger = logging.getLogger(__file__)


@dataclass
class QueryKeywords:
    docid: str
    keywords: List[str]


@dataclass
class FuzzyRerank(Method[FuzzyParam]):
    param_type: ClassVar[Type] = FuzzyParam
    fasttext: FastText = field(init=False)
    es_client: EsClient = field(init=False)

    def __post_init__(self):
        super(FuzzyRerank, self).__post_init__()
        self.fasttext: FastText = FastText()
        self.es_client: EsClient = EsClient(es_index=self.context.es_index)

    def load_keywords(self) -> List[QueryKeywords]:
        path: Path = settings.cache_dir.joinpath(
            f'{self.context.es_index}/keywords/fuzzy.naive/100.json')
        with open(path) as fin:
            data: Dict[str, List[str]] = json.load(fin)
        lst: List[QueryKeywords] = [QueryKeywords(docid=key, keywords=val)
                                    for key, val in data.items()]
        return lst

    def load_query_mat(self,
                       qk: QueryKeywords) -> np.ndarray:
        docid: str = qk.docid
        mat: np.ndarray = np.load(settings.cache_dir.joinpath(
            f'{self.context.es_index}/embeddings/{docid}/embeddings.npy'))
        return mat

    def get_col_embs(self,
                     qk: QueryKeywords) -> Dict[str, np.ndarray]:
        """
        Get documents cached by keywrod search
        """
        embs: Dict[str, np.ndarray] = {}
        for doc in load_cols(docid=qk.docid,
                             runname='100',
                             dataset=self.context.es_index):
            tokens: List[str] = self.es_client.get_tokens_from_doc(
                docid=doc.docid)
            vecs: List[Optional[np.ndarray]] = self.fasttext.embed_words(tokens)
            embs[doc.docid] = np.array(
                [vec for vec in vecs if vec is not None])
        return embs

    def get_kembs(self,
                  qk: QueryKeywords) -> np.ndarray:
        """
        Find pre-generated keywords and embed them.

        Parameter
        -----
        doc_num
            Document number (starts with 0)
        """
        emb_list: List[np.ndarray] = [
            vec for vec in self.fasttext.embed_words(qk.keywords)
            if vec is not None]
        embs: np.ndarray = np.array(emb_list)
        print(embs.shape)
        return mat_normalize(embs)

    def _get_nns(self,
                 mat: np.ndarray,
                 keyword_embs: np.ndarray) -> List[int]:
        """
        Given embedding matrix mat, Get nearest keywords in keyword_embs

        Return
        -----
        list (len = mat.shape[0]) of nearest embedding's indexes
        """
        nns: List[int] = np.argmax(
            np.dot(mat, keyword_embs.T), axis=1).tolist()
        return nns

    def to_fuzzy_bows(self,
                      mat: np.ndarray,
                      keyword_embs: np.ndarray) -> np.ndarray:
        """
        Generate a FuzzyBoW vector according to keyword_embs

        Return
        -----
        1D array whose item[i] is the normalized frequency of ith keyword
        """
        nns = self._get_nns(mat=mat, keyword_embs=keyword_embs)
        counter: Counter = Counter(nns)
        counts: List[int] = [counter[i] if i in counter else 0
                             for i in range(keyword_embs.shape[0])]
        return np.array(counts) / np.sum(counts)

    def get_collection_fuzzy_bows(self,
                                  col_embs: Dict[str, np.ndarray],
                                  keyword_embs: np.ndarray) -> Dict[str, np.ndarray]:
        bow_dict: Dict[str, np.ndarray] = {
            docid: self.to_fuzzy_bows(mat=emb, keyword_embs=keyword_embs)
            for docid, emb in col_embs.items()
        }
        return bow_dict

    def match(self,
              qk: QueryKeywords,
              query_bow: np.ndarray,
              col_bows: Dict[str, np.ndarray]) -> TRECResult:
        """
        Yet this only computes cosine similarity as the similarity.
        There's room for adding other ways.
        """
        # dot is inadequate
        scores: Dict[str, float] = {docid: np.dot(query_bow, bow)
                                    for docid, bow in col_bows.items()}
        return TRECResult(query_docid=qk.docid,
                          scores=scores)

    def create_flow(self):
        # loading query
        loader: LoaderNode = LoaderNode(func=self.load_keywords)
        n_get_col_embs = TaskNode(self.get_col_embs)
        (n_get_col_embs < loader)('qk')
        n_get_kembs = TaskNode(self.get_kembs)
        (n_get_kembs < loader)('qk')

        # generate fBoW of the query
        n_load_query_mat = TaskNode(self.load_query_mat)
        (n_load_query_mat < loader)('qk')
        n_fbow = TaskNode(self.to_fuzzy_bows)
        (n_fbow < n_load_query_mat)('mat')
        (n_fbow < n_get_kembs)('keyword_embs')

        # generate fBoW of collection
        n_cfbow = TaskNode(self.get_collection_fuzzy_bows)
        (n_cfbow < n_get_col_embs)('col_embs')
        (n_cfbow < n_get_kembs)('keyword_embs')

        # integration
        n_match = TaskNode(self.match)
        (n_match < loader)('qk')
        (n_match < n_fbow)('query_bow')
        (n_match < n_cfbow)('col_bows')

        (self.dump_node < n_match)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, ])
        flow.typecheck()
        return flow
