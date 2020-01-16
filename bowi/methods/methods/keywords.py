"""
extract keywords -> do search
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Pattern, Set, Type  # type: ignore

from typedflow.flow import Flow
from typedflow.nodes import TaskNode

from bowi.elas.client import EsClient
from bowi.elas.search import EsResult, EsSearcher
from bowi.models import Document
from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param, TRECResult
from bowi.methods.common.cache import DFCacher
from bowi.utils.text import is_valid_word


@dataclass
class KeywordParam(Param):
    n_words: int

    @classmethod
    def from_args(cls, args) -> KeywordParam:
        return KeywordParam(n_words=args.n_keywords)


def extract_keywords(tfs: Dict[str, int],
                     n_words: int,
                     df_cacher: DFCacher) -> List[str]:
    tfidf_dict: Dict[str, float] = {
        word: tf * df_cacher.get_idf(word)
        for word, tf in tfs.items()
        if is_valid_word(word)
    }
    keywords: List[str] = [
        word for word, _ in sorted(
            tfidf_dict.items(), reverse=True, key=lambda x: x[1])][:n_words]
    return keywords


@dataclass
class KeywordBaseline(Method[KeywordParam]):
    param_type: ClassVar[Type] = KeywordParam
    escl_query: EsClient = field(init=False)
    df_cacher: DFCacher = field(init=False)

    def __post_init__(self):
        super(KeywordBaseline, self).__post_init__()
        self.escl_query = EsClient(f'{self.context.es_index}_query')
        self.df_cacher = DFCacher(dataset=self.context.es_index)

    def extract_keywords(self, doc: Document) -> List[str]:
        tfs: Dict[str, int] = self.escl_query.get_tfs(doc.docid)
        return extract_keywords(tfs=tfs,
                                n_words=self.param.n_words,
                                df_cacher=self.df_cacher)

    def search(self,
               doc: Document,
               keywords: List[str]) -> EsResult:
        searcher: EsSearcher = EsSearcher(es_index=self.context.es_index)
        candidates: EsResult = searcher\
            .initialize_query()\
            .add_query(terms=keywords, field='text')\
            .add_size(self.context.n_docs)\
            .add_filter(terms=doc.tags, field='tags')\
            .add_source_fields(['text'])\
            .search()
        return candidates

    def to_trec_result(self,
                       doc: Document,
                       es_result: EsResult) -> TRECResult:
        res: TRECResult = TRECResult(
            query_docid=doc.docid,
            scores=es_result.get_scores()
        )
        return res

    def create_flow(self,
                    debug: bool = False) -> Flow:
        node_keywords = TaskNode(self.extract_keywords)({
            'doc': self.load_node})

        node_search = TaskNode(self.search)({
            'doc': self.load_node,
            'keywords': node_keywords
        })
        node_trec = TaskNode(self.to_trec_result)({
            'doc': self.load_node,
            'es_result': node_search
        })
        (self.dump_node < node_trec)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, ],
                          debug=debug)
        return flow
