"""
extract keywords -> do search
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Type  # type: ignore

from typedflow.flow import Flow
from typedflow.nodes import TaskNode, DumpNode

from bowi.elas.client import EsClient
from bowi.elas.search import EsResult, EsSearcher
from bowi.models import Document
from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param, TRECResult
from bowi.methods.common.cache import DFCacher, KeywordCacher
from bowi.utils.text import is_valid_word


@dataclass
class SotaParam(Param):
    n_words: int

    @classmethod
    def from_args(cls, args) -> SotaParam:
        return SotaParam(n_words=args.n_keywords)


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


def search(es_index: str,
           keywords: List[str],
           tags: List[str],
           size: int,
           source_fields: List[str] = ['text', ]) -> EsResult:
    searcher: EsSearcher = EsSearcher(es_index=es_index)
    candidates: EsResult = searcher\
        .initialize_query()\
        .add_query(terms=keywords, field='text')\
        .add_size(size)\
        .add_filter(terms=tags, field='tags')\
        .add_source_fields(source_fields)\
        .search()
    return candidates


@dataclass
class Sota(Method[SotaParam]):
    param_type: ClassVar[Type] = SotaParam
    escl_query: EsClient = field(init=False)
    df_cacher: DFCacher = field(init=False)
    keyword_cacher: KeywordCacher = field(init=False)

    def __post_init__(self):
        super(Sota, self).__post_init__()
        self.escl_query = EsClient(f'{self.context.es_index}_query')
        self.keyword_cacher = KeywordCacher(context=self.context)
        self.df_cacher = DFCacher(dataset=self.context.es_index)

    def extract_keywords(self, doc: Document) -> List[str]:
        """
        Top5s are relevant, otherwise irrelevant
        """
        tfs: Dict[str, int] = self.escl_query.get_tfs(doc.docid)
        keywords: List[str] = extract_keywords(tfs=tfs,
                                               n_words=self.param.n_words,
                                               df_cacher=self.df_cacher)
        init_res: EsResult = search(es_index=self.context.es_index,
                                    keywords=keywords,
                                    tags=doc.tags,
                                    size=100)
        tokens: List[List[str]] = [
            self.escl_query.analyze_text(item.source['text'])
            for item in init_res.hits]
        pos_counter: Counter = Counter(sum(tokens[:5], []))
        neg_counter: Counter = Counter(sum(tokens[5:], []))
        return [word for word, freq in pos_counter.items()
                if (word not in neg_counter) or (freq > neg_counter[word])]

    def search(self,
               doc: Document,
               keywords: List[str]) -> EsResult:
        return search(es_index=self.context.es_index,
                      keywords=keywords,
                      tags=doc.tags,
                      size=self.context.n_docs)

    def to_trec_result(self,
                       doc: Document,
                       es_result: EsResult) -> TRECResult:
        res: TRECResult = TRECResult(
            query_docid=doc.docid,
            scores=es_result.get_scores()
        )
        return res

    def dump_keywords(self,
                      doc: Document,
                      keywords: List[str]) -> None:
        self.keyword_cacher.dump(docid=doc.docid, keywords=keywords)

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
        node_dump_keywords = DumpNode(self.dump_keywords)({
            'doc': self.load_node,
            'keywords': node_keywords
        })
        (self.dump_node < node_trec)('res')
        (self.dump_time_node < node_trec)('res')
        flow: Flow = Flow(
            dump_nodes=[self.dump_node, node_dump_keywords, self.dump_time_node],
            debug=debug)
        return flow
