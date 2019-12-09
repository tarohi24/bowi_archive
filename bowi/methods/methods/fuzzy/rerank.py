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
from typing import ClassVar, Dict, List, Type

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, LoaderNode
from tqdm import tqdm

from bowi.embedding.base import mat_normalize
from bowi.embedding.fasttext import FastText
from bowi.methods.common.methods import Method
from bowi.methods.common.pre_filtering import load_cols
from bowi.methods.common.types import TRECResult
from bowi.models import Document
from bowi.utils.text import get_all_tokens
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
    query_tokens: Dict[str, List[str]] = field(init=False)

    def __post_init__(self):
        super(FuzzyRerank, self).__post_init__()
        self.fasttext: FastText = FastText()
        # load query tokens
        dump_path: Path = settings.data_dir.joinpath(f'{self.context.es_index}/query/dump.bulk')
        with open(dump_path) as fin:
            lines: List[Dict] = [json.loads(line) for line in fin.read().splitlines()]
        self.query_tokens: Dict[str, List[str]] = {
            dic['docid']: get_all_tokens(dic['text'])
            for dic in lines
        }

    def load_keywords(self) -> List[QueryKeywords]:
        path: Path = settings.cache_dir.joinpath(
            f'{self.context.es_index}/keywords/fuzzy.naive/{self.context.runname}.json')
        with open(path) as fin:
            data: Dict[str, List[str]] = json.load(fin)
        return [QueryKeywords(docid=key, keywords=val)
                for key, val in data.items()]

    def load_query_mat(self,
                       qk: QueryKeywords) -> np.ndarray:
        docid: str = qk.docid
        return np.load(settings.cache_dir.joinpath(
            f'{self.context.es_index}/embeddings/{docid}.bulk.npy'))

    def get_cols(self,
                 qk: QueryKeywords) -> List[Document]:
        """
        Get documents cached by keywrod search
        """
        cols: List[Document] = load_cols(
            docid=qk.docid,
            runname=self.param.prefilter_name,
            dataset=self.context.es_index)
        return cols

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
                                  cols: List[Document],
                                  keyword_embs: np.ndarray) -> Dict[str, np.ndarray]:
        bow_dict: Dict[str, np.ndarray] = dict()
        for doc in tqdm(cols, desc='computing bow...', leave=True):
            tokens: List[str] = get_all_tokens(doc.text)
            embs: np.ndarray = np.array([
                vec for vec in self.fasttext.embed_words(tokens)
                if vec is not None])
            embs = mat_normalize(embs)
            bow: np.ndarray = self.to_fuzzy_bows(mat=embs,
                                                 keyword_embs=keyword_embs)
            bow_dict[doc.docid] = bow
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
        scores: Dict[str, float] = {docid: 1 / (np.linalg.norm(query_bow - bow) + 0.1)
                                    for docid, bow in col_bows.items()}
        return TRECResult(query_docid=qk.docid,
                          scores=scores)

    def create_flow(self):
        # loading query
        node_loader: LoaderNode = LoaderNode(func=self.load_keywords,
                                             batch_size=1)
        node_get_cols: TaskNode = TaskNode(func=self.get_cols)
        node_get_kembs: TaskNode = TaskNode(func=self.get_kembs)
        (node_get_cols < node_loader)('qk')
        (node_get_kembs < node_loader)('qk')

        # get query embeddin

        # generate fBoW of the query
        node_query_emb: TaskNode = TaskNode(func=self.load_query_mat)
        (node_query_emb < node_loader)('qk')
        node_query_bow: TaskNode = TaskNode(func=self.to_fuzzy_bows)
        (node_query_bow < node_query_emb)('mat')
        (node_query_bow < node_get_kembs)('keyword_embs')

        # generate fBoW of collection
        node_bow: TaskNode = TaskNode(func=self.get_collection_fuzzy_bows)
        (node_bow < node_get_cols)('cols')
        (node_bow < node_get_kembs)('keyword_embs')

        # integration
        node_match: TaskNode = TaskNode(func=self.match)
        (node_match < node_loader)('qk')
        (node_match < node_query_bow)('query_bow')
        (node_match < node_bow)('col_bows')

        (self.dump_node < node_match)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, ])
        flow.typecheck()
        return flow
