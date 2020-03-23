"""
Topic ranking w/o topic modeling
"""
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from operator import itemgetter
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
class LinearParam(Param):
    window: int
    w: int
    sliding_step: int = 100
    n_words: int = 30  # words per cluster


@dataclass
class TFIDF(Param):
    mat: np.ndarray
    wids: Dict[str, int]  # word: id
    id2word: Dict[int, str] = field(init=False)

    def __post_init__(self):
        self.id2word = {id_: word for word, id_ in self.wids.items()}


@dataclass
class Linear(Method[LinearParam]):
    param_type: ClassVar[Type] = LinearParam
    df_cacher: DFCacher = field(init=False)
    escl_query: EsClient = field(init=False)
    keyword_cacher: KeywordCacher = field(init=False)
    ttt: TextTilingTokenizer = field(init=False)

    def __post_init__(self):
        super(Linear, self).__post_init__()
        self.df_cacher = DFCacher(dataset=self.context.es_index)
        self.escl_query = EsClient(f'{self.context.es_index}_query')
        self.keyword_cacher = KeywordCacher(context=self.context)
        self.ttt = TextTilingTokenizer(w=self.param.w)

    def chunk(self,
              doc: Document) -> List[List[str]]:
        text: str = doc.text
        # Add \n\n because paragraph boundaries cannot be obtained from dataset
        segments: List[str] = self.ttt.tokenize(text.replace('.', '.\n\n'))
        tokens: List[List[str]] = [
            [tok for tok in self.escl_query.analyze_text(chunk.replace('\n\n', ' '))
             if is_valid_word(tok)]
            for chunk in segments]
        return tokens

    def dist(self,
             p: float,
             count: int) -> float:
        assert count >= 0
        if count == 0:
            return 1 - p
        else:
            return p / (2 ** (count - 1))

    def _get_linear_scores(self,
                           word: str,
                           seq: List[str],
                           mult_idf: bool = True) -> np.ndarray:
        window: int = self.param.window
        isin_list: List[int] = [1 if w == word else 0 for w in seq]
        # counts: List[int] = [sum(isin_list[i:i + window]) for i in range(len(isin_list) - window)]
        start_inds: List[int] = list(range(0, len(isin_list), self.param.sliding_step))
        counts: List[int] = [sum(isin_list[start:srart + window]) for start in start_inds]
        try:
            df: int = self.df_cacher.df_dict[word]
        except KeyError:
            df: int = 1  # type: ignore
        p: float = df / self.df_cacher.total_docs
        return np.array([1 - self.dist(p, count) for count in counts])

    def get_keywords(self,
                     tokens: List[List[str]]) -> List[List[str]]:
        window: int = self.param.window
        keywords: List[str] = []
        word2id: Dict[str, int] = {word: i for i, word in enumerate(set(sum(tokens, [])))}
        words: List[str] = [w for w, _ in sorted(word2id.items(), key=itemgetter(1))]
        seq: List[str] = sum(tokens, [])
        rg: List[int] = list(range(0, len(seq), window))[:-1]
        # counter = Counter(words)

        scores = []
        for word in words:
            try:
                df: int = self.df_cacher.df_dict[word]
            except KeyError:
                df: int = 1  # type: ignore
            p: float = df / self.df_cacher.total_docs
            # tf: int = counter[word]
            sc: np.ndarray = np.array([
                (2 ** sum(w == word for w in seq[i:j]) - 1) / p
                for i, j in zip(rg, rg[1:])
            ])
            scores.append(sc)

        df: pd.DataFrame = pd.DataFrame(scores, index=words).T
        bins: List[int] = [sum([len(lst) for lst in tokens[:i]]) for i in range(len(tokens) + 1)]
        segments: List[int] = pd.cut([i * window for i in df.index],
                                     bins,
                                     labels=range(len(bins) - 1)).tolist()
        maxes: pd.DataFrame = df.groupby(segments).max()
        keywords: List[List[str]] = [
            maxes.iloc[i, :].sort_values(ascending=False)[:self.param.n_words].index.tolist()
            for i in range(len(maxes))]
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
        node_chunk = TaskNode(self.chunk)({
            'doc': self.load_node,
        })
        node_keywords = TaskNode(self.get_keywords)({
            'tokens': node_chunk,
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
        (self.dump_time_node < node_search)('res')
        flow: Flow = Flow(
            dump_nodes=[self.dump_node, node_dump_keywords, self.dump_time_node],
            debug=debug)
        return flow
