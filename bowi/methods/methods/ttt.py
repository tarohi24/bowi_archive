"""
Topic ranking w/o topic modeling
"""
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import ClassVar, Dict, List, Type  # type: ignore

from nltk.tokenize import TextTilingTokenizer
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
class TTTParam(Param):
    w: int
    n_words: int = 30  # words per cluster


@dataclass
class TFIDF(Param):
    mat: np.ndarray
    wids: Dict[str, int]  # word: id
    id2word: Dict[int, str] = field(init=False)

    def __post_init__(self):
        self.id2word = {id_: word for word, id_ in self.wids.items()}


@dataclass
class TTT(Method[TTTParam]):
    param_type: ClassVar[Type] = TTTParam
    df_cacher: DFCacher = field(init=False)
    escl_query: EsClient = field(init=False)
    keyword_cacher: KeywordCacher = field(init=False)
    ttt: TextTilingTokenizer = field(init=False)

    def __post_init__(self):
        super(TTT, self).__post_init__()
        self.df_cacher = DFCacher(dataset=self.context.es_index)
        self.escl_query = EsClient(f'{self.context.es_index}_query')
        self.keyword_cacher = KeywordCacher(context=self.context)
        self.ttt = TextTilingTokenizer(w=self.param.w)

    def get_keywords(self,
                     doc: Document) -> List[List[str]]:
        text: str = doc.text
        # Add \n\n because paragraph boundaries cannot be obtained from dataset
        segments: List[str] = self.ttt.tokenize(text.replace('.', '.\n\n'))
        keywords: List[List[str]] = []
        for chunk in segments:
            tokens: List[str] = [tok for tok in self.escl_query.analyze_text(chunk.replace('\n\n', ' '))
                                 if is_valid_word(tok)]
            tfidf: pd.Series = pd.Series(self.df_cacher.to_tfidf(tokens))
            keywords.append(
                tfidf.sort_values(ascending=False)[:self.param.n_words].index.tolist())
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
        node_keywords = TaskNode(self.get_keywords)({
            'doc': self.load_node,
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
