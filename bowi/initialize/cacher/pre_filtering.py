"""
Pre-filtering by keywords
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, List, Type, Counter

from typedflow.nodes import TaskNode, DumpNode
from typedflow.flow import Flow

from bowi.elas.search import EsResult, EsSearcher
from bowi.methods.common.methods import Method
from bowi.methods.methods.keywords import KeywordParam, KeywordBaseline
from bowi.models import Document
from bowi.utils.text import get_all_tokens
from bowi import settings


@dataclass
class PreSearcher(Method[KeywordParam]):
    param_type: ClassVar[Type] = KeywordParam

    def dump(self,
             query_doc: Document,
             res: EsResult) -> None:
        path: Path = settings.cache_dir.joinpath(
            f'{self.context.es_index}/text/{self.context.runname}/{query_doc.docid}.bulk')
        with open(path, 'w') as fout:
            for doc in res.to_docs():
                fout.write(doc.to_json())  # type: ignore
                fout.write('\n')

    def search(self,
               doc: Document) -> EsResult:
        tokens: List[str] = get_all_tokens(doc.text)
        counter: Counter[str] = Counter(tokens)
        keywords: List[str] = [word for word, _ in counter.most_common(self.param.n_words)]
        searcher: EsSearcher = EsSearcher(es_index=self.context.es_index)
        candidates: EsResult = searcher\
            .initialize_query()\
            .add_query(terms=keywords, field='text')\
            .add_size(self.context.n_docs)\
            .add_filter(terms=doc.tags, field='tags')\
            .add_source_fields(['text', 'tags', 'title'])\
            .search()
        return candidates

    def create_flow(self) -> Flow:
        node_search: TaskNode = TaskNode(self.search)
        (node_search < self.load_node)('doc')
        node_dump: DumpNode = DumpNode(func=self.dump)
        (node_dump < self.load_node)('query_doc')
        (node_dump < node_search)('res')

        flow: Flow = Flow(dump_nodes=[node_dump, ])
        return flow
