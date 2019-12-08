"""
Available only for fasttext
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import ClassVar, List, Type, Generator, Optional

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, DumpNode, LoaderNode

from bowi.elas.search import EsResult, EsSearcher
from bowi.embedding.base import mat_normalize
from bowi.embedding.fasttext import FastText
from bowi.methods.common.methods import Method
from bowi.methods.common.types import TRECResult, Context
from bowi.models import Document
from bowi.methods.common.dumper import dump_keywords
from bowi.utils.text import get_all_tokens

from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.methods.methods.fuzzy.fuzzy import get_keyword_embs


logger = logging.getLogger(__file__)


@dataclass
class FuzzyNaive(Method[FuzzyParam]):
    param_type: ClassVar[Type] = FuzzyParam
    fasttext: FastText = field(init=False)

    def __post_init__(self):
        super(FuzzyNaive, self).__post_init__()
        self.fasttext: FastText = FastText()

    def get_tokens(self, doc: Document) -> List[str]:
        return get_all_tokens(doc.text)

    def extract_keywords(self,
                         tokens: List[str]) -> List[str]:
        optional_embs: List[Optional[np.ndarray]] = self.fasttext.embed_words(tokens)
        tokens: List[str] = [w for w, vec in zip(tokens, optional_embs)  # type: ignore
                             if vec is not None]
        matrix = mat_normalize(np.array([vec for vec in optional_embs if vec is not None]))
        assert len(tokens) == matrix.shape[0]

        k_embs: np.ndarray = get_keyword_embs(
            embs=matrix,
            keyword_embs=None,
            n_remains=self.param.n_words,
            coef=self.param.coef)
        logger.info(k_embs.sum(axis=1))
        indices: List[int] = [np.argmin(np.linalg.norm(matrix - vec, axis=1))
                              for vec in k_embs]
        logger.info(indices)
        keywords: List[str] = list(set([tokens[i] for i in indices]))
        logger.info(keywords)
        return keywords

    def to_trec_result(self,
                       doc: Document,
                       es_result: EsResult) -> TRECResult:
        res: TRECResult = TRECResult(
            query_docid=doc.docid,
            scores=es_result.get_scores()
        )
        return res

    def match(self,
              query_doc: Document,
              keywords: List[str]) -> TRECResult:
        searcher: EsSearcher = EsSearcher(es_index=self.context.es_index)
        candidates: EsResult = searcher\
            .initialize_query()\
            .add_query(terms=keywords, field='text')\
            .add_size(self.context.n_docs)\
            .add_filter(terms=query_doc.tags, field='tags')\
            .add_source_fields(['text'])\
            .search()
        trec_result: TRECResult = self.to_trec_result(doc=query_doc, es_result=candidates)
        return trec_result

    def create_flow(self):

        def provide_context() -> Generator[Context, None, None]:
            while True:
                yield self.context

        node_get_tokens: TaskNode[List[str]] = TaskNode(func=self.get_tokens)
        (node_get_tokens < self.load_node)('doc')

        node_get_keywords: TaskNode[List[str]] = TaskNode(func=self.extract_keywords)
        (node_get_tokens > node_get_keywords)('tokens')
        keywords_dumper: DumpNode = DumpNode(dump_keywords)
        (keywords_dumper < node_get_keywords)('keywords')
        (keywords_dumper < self.load_node)('doc')
        (keywords_dumper < LoaderNode(provide_context,
                                      batch_size=1))('context')
        node_match: TaskNode[TRECResult] = TaskNode(func=self.match)
        (node_match < self.load_node)('query_doc')
        (node_match < node_get_keywords)('keywords')

        (self.dump_node < node_match)('res')

        flow: Flow = Flow(dump_nodes=[self.dump_node, keywords_dumper])
        return flow
