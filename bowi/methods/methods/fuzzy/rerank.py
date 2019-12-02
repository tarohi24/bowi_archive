from collections import Counter
from dataclasses import dataclass, field
import logging
from typing import ClassVar, Dict, List, Type


import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode
from tqdm import tqdm

from bowi.embedding.base import mat_normalize
from bowi.embedding.fasttext import FastText
from bowi.methods.common.methods import Method
from bowi.methods.common.pre_filtering import load_cols
from bowi.methods.common.types import TRECResult
from bowi.models import Document

from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.methods.methods.fuzzy.fuzzy import get_keyword_embs
from bowi.methods.methods.fuzzy.tokenize import get_all_tokens


logger = logging.getLogger(__file__)


@dataclass
class FuzzyRerank(Method[FuzzyParam]):
    param_type: ClassVar[Type] = FuzzyParam
    fasttext: FastText = field(init=False)

    def __post_init__(self):
        super(FuzzyRerank, self).__post_init__()
        self.fasttext: FastText = FastText()

    def get_cols(self,
                 query: Document) -> List[Document]:
        """
        Get documents cached by keywrod search
        """
        cols: List[Document] = load_cols(
            docid=query.docid,
            dataset=self.context.es_index)
        return cols

    def embed_words(self,
                    tokens: List[str]) -> np.ndarray:
        orig_emb: np.ndarray = self.fasttext.embed_words(tokens)
        orig_emb = orig_emb[[any(vec != 0) for vec in orig_emb], :]  # type: ignore
        matrix: np.ndarray = mat_normalize(orig_emb)  # (n_tokens, n_dim)
        return matrix

    def get_kembs(self,
                  mat: np.ndarray) -> np.ndarray:
        """
        Given embeddint matrix mat (n_tokens * n_dim) and tokens (list (n_tokens)),
        calculate keyword tokens
        """
        embs: np.ndarray = get_keyword_embs(embs=mat,
                                            keyword_embs=None,
                                            n_remains=self.param.n_words,
                                            coef=self.param.coef)
        return embs

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
            tokens: List[str] = get_all_tokens(doc)
            embs: np.ndarray = mat_normalize(self.embed_words(tokens))
            bow: np.ndarray = self.to_fuzzy_bows(mat=embs,
                                                 keyword_embs=keyword_embs)
            bow_dict[doc.docid] = bow
        return bow_dict

    def match(self,
              query_doc: Document,
              query_bow: np.ndarray,
              col_bows: Dict[str, np.ndarray]) -> TRECResult:
        """
        Yet this only computes cosine similarity as the similarity.
        There's room for adding other ways.
        """
        # dot is inadequate
        scores: Dict[str, float] = {docid: 1 / (np.linalg.norm(query_bow - bow) + 0.1)
                                    for docid, bow in col_bows.items()}
        return TRECResult(query_docid=query_doc.docid,
                          scores=scores)

    def create_flow(self):
        # query
        node_tokens: TaskNode[List[str]] = TaskNode(func=get_all_tokens)
        (node_tokens < self.load_node)('doc')

        node_emb: TaskNode[np.ndarray] = TaskNode(func=self.embed_words)
        (node_emb < node_tokens)('tokens')

        node_keyword_embs: TaskNode[np.ndarray] = TaskNode(
            func=self.get_kembs)
        (node_keyword_embs < node_emb)('mat')

        node_bow: TaskNode[np.ndarray] = TaskNode(
            func=self.to_fuzzy_bows)
        (node_bow < node_emb)('mat')
        (node_bow < node_keyword_embs)('keyword_embs')

        # col
        node_cols: TaskNode[List[Document]] = TaskNode(
            func=self.get_cols)
        (node_cols < self.load_node)('query')

        node_col_bows: TaskNode[Dict[str, np.ndarray]] = TaskNode(
            func=self.get_collection_fuzzy_bows)
        (node_col_bows < node_cols)('cols')
        (node_col_bows < node_keyword_embs)('keyword_embs')

        # integration
        node_match: TaskNode[TRECResult] = TaskNode(func=self.match)
        (node_match < self.load_node)('query_doc')
        (node_match < node_bow)('query_bow')
        (node_match < node_col_bows)('col_bows')

        (self.dump_node < node_match)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, ])
        flow.typecheck()
        return flow
