"""
Word Mover Distance
"""
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, ClassVar, Type

import datetime
import logging
import numpy as np
import pulp
from scipy.spatial.distance import euclidean
from typedflow.nodes import TaskNode
from typedflow.flow import Flow

from bowi.elas.search import EsClient, EsResult
from bowi.methods.methods.keywords import KeywordParam, KeywordBaseline
from bowi.methods.common.methods import Method
from bowi.methods.common.types import TRECResult, Param
from bowi.embedding.fasttext import FastText
from bowi.models import Document


logger = logging.getLogger(__file__)


@dataclass
class WMDParam(Param):
    n_words: int = 100

    def to_keyword_param(self) -> KeywordParam:
        return KeywordParam(n_words=self.n_words)


@dataclass
class WMD(Method[WMDParam]):
    param_type: ClassVar[Type] = WMDParam
    fasttext: FastText = field(init=False)
    kb: KeywordBaseline = field(init=False)
    escl: EsClient = field(init=False)

    def __post_init__(self):
        super(WMD, self).__post_init__()
        self.fasttext: FastText = FastText()
        self.kb = KeywordBaseline(context=self.context,
                                  param=self.param.to_keyword_param())
        self.escl = EsClient(self.context.es_index)

    def filter_in_advance(self,
                          doc: Document) -> EsResult:
        logger.warn(f'starting time: {datetime.datetime.now()}')
        keywords: List[str] = self.kb.extract_keywords(doc)
        res: EsResult = self.kb.search(doc, keywords)
        return res

    def count_prob(self,
                   tokens: List[str]) -> Dict[str, float]:
        """
        Compute emergence probabilities for each word
        """
        n_tokens: int = len(tokens)
        counter: Dict[str, int] = Counter(tokens)
        return {word: counter[word] / n_tokens for word in counter.keys()}

    def wmd(self,
            A_tokens: List[str],
            B_tokens: List[str]) -> float:
        """
        Return
        ---------------------
        Similarity
        """
        all_tokens: List[str] = list(set(A_tokens) | set(B_tokens))

        A_prob: Dict[str, float] = self.count_prob(A_tokens)
        B_prob: Dict[str, float] = self.count_prob(B_tokens)
        wv: Dict[str, np.ndarray] = {token: self.fasttext.embed(token) for token in all_tokens}
        var_dict = pulp.LpVariable.dicts(
            'T_matrix',
            list(product(all_tokens, all_tokens)),
            lowBound=0)
        prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
        prob += pulp.lpSum([var_dict[token1, token2] * euclidean(wv[token1], wv[token2])
                            for token1, token2 in product(all_tokens, all_tokens)])

        for token2 in B_prob.keys():
            prob += pulp.lpSum(
                [var_dict[token1, token2] for token1 in B_prob.keys()]
            ) == B_prob[token2]
        for token1 in A_prob.keys():
            prob += pulp.lpSum(
                [var_dict[token1, token2] for token2 in A_prob.keys()]
            ) == A_prob[token1]
        prob.solve()
        return -pulp.value(prob.objective)

    def retrieve(self,
                 doc: Document,
                 res: EsResult) -> TRECResult:
        scores: Dict[str, float] = dict()
        q_tokens: List[str] = self.kb.escl_query.get_tokens_from_doc(doc.docid)
        for docid in res.get_ids():
            tokens: List[str] = self.escl.get_tokens_from_doc(docid)
            scores[docid] = self.wmd(q_tokens, tokens)
        logger.warn(f'end time: {datetime.datetime.now()}')
        return TRECResult(doc.docid, scores)

    def create_flow(self,
                    debug: bool = False) -> Flow:
        node_filter = TaskNode(self.filter_in_advance)({  # type: ignore
            'doc': self.load_node,
        })
        node_retrieve = TaskNode(self.retrieve)({  # type: ignore
            'doc': self.load_node,
            'res': node_filter
        })
        (self.dump_node < node_retrieve)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, ],
                          debug=debug)
        return flow
