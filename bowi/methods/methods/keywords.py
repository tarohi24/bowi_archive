"""
extract keywords -> do search
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import re
from pathlib import Path
import json
from typing import ClassVar, Dict, Generator, List, Pattern, Set, Type  # type: ignore

from nltk.corpus import stopwords as nltk_sw
from nltk.tokenize import RegexpTokenizer
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, DumpNode, LoaderNode

from bowi.elas.search import EsResult, EsSearcher
from bowi.models import Document
from bowi.methods.common.methods import Method
from bowi.methods.common.dumper import get_dump_dir
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
