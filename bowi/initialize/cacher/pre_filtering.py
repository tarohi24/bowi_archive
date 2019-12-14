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

    def create_flow(self, debug: bool = False) -> Flow:
        node_search = TaskNode(self.search)({'doc': self.load_node})
        node_dump = DumpNode(func=self.dump)({
            'query_doc': self.load_node,
            'res': node_search
        })
        flow: Flow = Flow(dump_nodes=[node_dump, ], debug=debug)
        flow.typecheck()
        return flow
