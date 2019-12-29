"""
Improved BM25
"""
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Type, Tuple  # type: ignore

from typedflow.flow import Flow
from typedflow.nodes import TaskNode

from bowi.models import Document
from bowi.elas.client import EsClient
from bowi.elas.search import EsSearcher
from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param, TRECResult
from bowi.methods.common.cache import KNNCacher, DFCacher


@dataclass
class BM25IParam(Param):
    n_words: int = 100
    threshold: float = 0.5  # note that threshold must be above 0.5
    k1: float = 1.2  # default in Elasticsearch
    b: float = 0.75  # default in Elasticsearch


@dataclass
class BM25I(Method[BM25IParam]):
    param_type: ClassVar[Type] = BM25IParam
    knn_cacher: KNNCacher = field(init=False)
    df_cacher: DFCacher = field(init=False)
    avgdl: float = field(init=False)
    es_client: EsClient = field(init=False)
    es_topic_client: EsClient = field(init=False)

    def __post_init__(self):
        super(BM25I, self).__post_init__()
        self.knn_cacher = KNNCacher(dataset=self.context.es_index)
        self.df_cacher = DFCacher(dataset=self.context.es_index)
        self.avgdl: float = {
            'clef': 3804.996
        }[self.context.es_index]
        self.es_client = EsClient(self.context.es_index)
        self.es_topic_client = EsClient(f'{self.context.es_index}_query')

    def get_query_keywords(self, doc: Document) -> List[str]:
        tfidfs: Dict[str, Tuple[int, float]] = self.es_topic_client.get_tfidfs(doc.docid)
        words: List[str] = [
            word for word, _
            in sorted(tfidfs.items(), key=lambda x: x[1][0], reverse=True)
        ][:self.param.n_words]
        return words

    def get_cols(self,
                 doc: Document,
                 keywords: List[str]) -> Dict[str, Dict[str, int]]:
        """
        Search by given keywords and extract collection docs
        with their tokens and their frequencies
        """
        searcher: EsSearcher = EsSearcher(es_index=self.context.es_index)
        colids: List[str] = searcher\
            .initialize_query()\
            .add_query(terms=keywords, field='text')\
            .add_size(300)\
            .add_filter(terms=doc.tags, field='tags')\
            .add_source_fields(['docid', ])\
            .search()\
            .get_ids()

        tfs: Dict[str, Dict[str, int]] = dict()
        for docid in colids:
            tfidfs: Dict[str, Tuple[int, float]] = self.es_client.get_tfidfs(
                docid=docid)
            tfs[docid] = {word: tf for word, (tf, _) in tfidfs.items()}
        return tfs

    def _get_bm25i(self,
                   query: List[str],
                   collection: Dict[str, int]) -> float:
        """
        Compute BM25i score

        Parameters
        -----
        query
            {word: tf}
        """
        score: float = 0
        col_dl_factor: float = sum(collection.values()) / self.avgdl
        for q in query:
            try:
                simwords: Dict[str, float] = self.knn_cacher.get_nn(
                    word=q,
                    threshold=self.param.threshold,
                    include_self=True)
            except KeyError:
                simwords: Dict[str, float] = {q: 0.0}  # type: ignore
            for word, dist in simwords.items():
                try:
                    tf: int = collection[word]
                except KeyError:
                    continue
                idf: float = self.df_cacher.get_idf(word)
                sim: float = 1 - dist
                numerator: float = sim * idf * tf * (self.param.k1 + 1)
                denominator: float = tf * self.param.k1 * (1 - self.param.b + self.param.b * col_dl_factor)
                score += (numerator / denominator)
        return score

    def get_bm25i(self,
                  doc: Document,
                  query: List[str],
                  col_tfs: Dict[str, Dict[str, int]]) -> TRECResult:
        """
        Compute BM25i for each document
        """
        scores: Dict[str, float] = {
            docid: self._get_bm25i(query=query, collection=col)
            for docid, col in col_tfs.items()
        }
        res: TRECResult = TRECResult(
            query_docid=doc.docid,
            scores=scores)
        return res

    def create_flow(self, debug: bool = False) -> Flow:
        node_keywords = TaskNode(self.get_query_keywords)({
            'doc': self.load_node,
        })
        node_get_cols = TaskNode(self.get_cols)({
            'doc': self.load_node,
            'keywords': node_keywords,
        })
        node_get_bm25 = TaskNode(self.get_bm25i)({
            'doc': self.load_node,
            'query': node_keywords,
            'col_tfs': node_get_cols
        })
        (self.dump_node < node_get_bm25)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, ],
                          debug=debug)
        return flow
