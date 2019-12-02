"""
extract keywords -> do search
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import re
from typing import ClassVar, Generator, List, Pattern, Set, Type  # type: ignore

from nltk.corpus import stopwords as nltk_sw
from nltk.tokenize import RegexpTokenizer
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, DumpNode, LoaderNode

from bowi.elas.search import EsResult, EsSearcher
from bowi.models import Document
from bowi.methods.common.methods import Method
from bowi.methods.common.dumper import dump_keywords
from bowi.methods.common.types import Param, TRECResult, Context


stopwords: Set[str] = set(nltk_sw.words('english'))
tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
not_a_word_pat: Pattern = re.compile(r'^[^a-z0-9]*$')


@dataclass
class KeywordParam(Param):
    n_words: int

    @classmethod
    def from_args(cls, args) -> KeywordParam:
        return KeywordParam(n_words=args.n_keywords)


def extract_keywords_from_text(text: str,
                               n_words: int) -> List[str]:
    # lower and tokenize
    tokens: List[str] = tokenizer.tokenize(text.lower())
    # remove stopwords
    tokens: List[str] = [w for w in tokens if w not in stopwords]  # type: ignore
    tokens: List[str] = [w for w in tokens  # type: ignore
                         if not_a_word_pat.match(w) is None
                         and not w.isdigit()]
    counter: Counter = Counter(tokens)
    keywords: List[str] = [
        w for w, _ in counter.most_common(n_words)]
    return keywords


@dataclass
class KeywordBaseline(Method[KeywordParam]):
    param_type: ClassVar[Type] = KeywordParam

    def extract_keywords(self, doc: Document) -> List[str]:
        return extract_keywords_from_text(text=doc.text,
                                          n_words=self.param.n_words)

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

    def create_flow(self) -> Flow:

        def provide_context() -> Generator[Context, None, None]:
            while True:
                yield self.context

        node_keywords: TaskNode[List[str]] = TaskNode(self.extract_keywords)
        (node_keywords < self.load_node)('doc')
        node_search: TaskNode[EsResult] = TaskNode(self.search)
        (node_search < self.load_node)('doc')
        (node_search < node_keywords)('keywords')
        keyword_dumper: DumpNode = DumpNode(func=dump_keywords)
        (keyword_dumper < node_keywords)('keywords')
        (keyword_dumper < self.load_node)('doc')
        context_loader: LoaderNode[Context] = LoaderNode(provide_context,
                                                         batch_size=1)
        (keyword_dumper < context_loader)('context')

        node_trec: TaskNode[TRECResult] = TaskNode(self.to_trec_result)
        (node_trec < self.load_node)('doc')
        (node_trec < node_search)('es_result')

        (node_trec > self.dump_node)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, keyword_dumper])
        return flow
