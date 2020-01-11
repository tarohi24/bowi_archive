"""
Topic ranking w/o topic modeling
"""
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import ClassVar, Dict, List, Tuple, Type  # type: ignore

import pandas as pd
import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, DumpNode

from bowi.models import Document
from bowi.elas.search import EsSearcher
from bowi.elas.client import EsClient
from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param, TRECResult
from bowi.methods.common.cache import DFCacher, KeywordCacher
from bowi.methods.methods.keywords import is_valid_word


logger = logging.getLogger(__file__)


@dataclass
class TopicParam(Param):
    n_words: int = 30  # words per cluster
    n_topics: int = 5


@dataclass
class TFIDF(Param):
    mat: np.ndarray
    wids: Dict[str, int]  # word: id
    id2word: Dict[int, str] = field(init=False)

    def __post_init__(self):
        self.id2word = {id_: word for word, id_ in self.wids.items()}


@dataclass
class Topicrank(Method[TopicParam]):
    param_type: ClassVar[Type] = TopicParam
    df_cacher: DFCacher = field(init=False)
    escl_query: EsClient = field(init=False)
    keyword_cacher: KeywordCacher = field(init=False)

    def __post_init__(self):
        super(Topicrank, self).__post_init__()
        self.df_cacher = DFCacher(dataset=self.context.es_index)
        self.escl_query = EsClient(f'{self.context.es_index}_query')
        self.keyword_cacher = KeywordCacher(context=self.context)

    def get_token_positions(self,
                            doc: Document) -> Dict[str, List[int]]:
        """
        {word: [pos1, pos2, ...]}
        """
        tokens: Dict[str, List[int]] = {
            word: [p['position'] for p in val['tokens']]
            for word, val
            in self.escl_query.es.termvectors(
                index='clef_query',
                id=doc.docid,
                fields=['text', ]
            )['term_vectors']['text']['terms'].items()
            if is_valid_word(word)
        }
        return tokens

    def compute_tfidf(self,
                      tokens: Dict[str, List[int]]) -> TFIDF:
        """
        Return
        -----
        (n_topics, n_words)
        """
        wids: Dict[str, int] = {word: i for i, word in enumerate(tokens.keys())}
        # Create a word list to fix word id
        words = [word for word, _ in sorted(wids.items(), key=lambda x: x[1])]
        # Expand {word: [p1, p2, ...]} -> [(word, p1), (word, p2), ...]
        positions: List[Tuple[int, int]] = sum([
            [(wids[word], pos) for pos in pos_list]
            for word, pos_list in tokens.items()
        ], [])
        ids, indices = list(zip(*positions))
        bins: List[int] = pd.qcut(indices,
                                  q=self.param.n_topics,
                                  labels=range(self.param.n_topics)).tolist()

        # Generate TFIDF matrix
        idfs: np.ndarray = np.array([self.df_cacher.get_idf(w) for w in words])
        tfidfs: np.ndarray = np.zeros((len(wids), self.param.n_topics))
        for wid, bin in zip(ids, bins):
            tfidfs[(wid, bin)] += idfs[wid]

        return TFIDF(mat=tfidfs.T, wids=wids)

    def get_keywords(self,
                     tfidf: TFIDF) -> List[List[str]]:
        mat: np.ndarray = tfidf.mat
        id2word: Dict[int, str] = tfidf.id2word
        assert mat.shape[0] == self.param.n_topics
        # top n_words
        keyword_ids: np.ndarray = np.argsort(mat, axis=1)[:, -1 * self.param.n_words:-1]
        keywords: List[List[str]] = [[id2word[i] for i in arr] for arr in keyword_ids]
        return keywords

    def search(self,
               doc: Document,
               keywords: List[List[str]]) -> TRECResult:
        searcher: EsSearcher = EsSearcher(es_index=self.context.es_index)
        scores: Dict[str, float] = defaultdict(float)
        for terms in keywords:
            sc: Dict[str, float] = searcher\
                .initialize_query()\
                .add_query(terms=terms, field='text')\
                .add_size(300)\
                .add_filter(terms=doc.tags, field='tags')\
                .add_source_fields(['docid', ])\
                .search()\
                .get_scores()
            for docid, point in sc.items():
                scores[docid] += point
        res: TRECResult = TRECResult(
            query_docid=doc.docid,
            scores=scores)
        return res

    def dump_keywords(self,
                      doc: Document,
                      keywords: List[List[str]]) -> None:
        for kwds in keywords:
            self.keyword_cacher.dump(docid=doc.docid, keywords=kwds)

    def create_flow(self, debug: bool = False) -> Flow:
        node_tokens = TaskNode(self.get_token_positions)({
            'doc': self.load_node,
        })
        node_tfidf = TaskNode(self.compute_tfidf)({
            'tokens': node_tokens,
        })
        node_keywords = TaskNode(self.get_keywords)({
            'tfidf': node_tfidf,
        })
        node_search = TaskNode(self.search)({
            'doc': self.load_node,
            'keywords': node_keywords
        })
        (self.dump_node < node_search)('res')
        node_dump_keywords = DumpNode(self.dump_keywords)({
            'doc': self.load_node,
            'keywords': node_keywords
        })
        flow: Flow = Flow(
            dump_nodes=[self.dump_node, node_dump_keywords],
            debug=debug)
        return flow
