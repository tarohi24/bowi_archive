"""
Available only for fasttext
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import ClassVar, List, Type, Optional

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, DumpNode

from bowi.elas.client import EsClient
from bowi.elas.search import EsResult, EsSearcher
from bowi.embedding.base import mat_normalize
from bowi.embedding.fasttext import FastText
from bowi.methods.common.methods import Method
from bowi.methods.common.types import TRECResult
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
    es_client: EsClient = field(init=False)

    def __post_init__(self):
        super(FuzzyNaive, self).__post_init__()
        self.fasttext: FastText = FastText()
        self.es_client: EsClient = EsClient(es_index=self.context.es_index)

    def get_tokens(self, doc: Document) -> List[str]:
        return get_all_tokens(doc.text)

    def extract_keywords(self,
                         doc: Document,
                         tokens: List[str]) -> List[str]:
        optional_embs: List[Optional[np.ndarray]] = self.fasttext.embed_words(tokens)
        idfs: np.ndarray = self.es_client.get_idfs(docid=doc.docid)
        assert len(optional_embs) == len(idfs)

        indices: List[bool] = [vec is not None for vec in optional_embs]
        idfs = idfs[indices]  # type: ignore
        tokens: List[str] = [w for w, is_valid in zip(tokens, indices) if is_valid]  # type: ignore
        matrix = mat_normalize(np.array([vec for vec in optional_embs if vec is not None]))
        assert len(tokens) == matrix.shape[0] == len(idfs)

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

        def _dump_kwards(keywords: List[str],
                         doc: Document) -> None:
            return dump_keywords(keywords=keywords, doc=doc, context=self.context)

        node_get_tokens = TaskNode(self.get_tokens)({'doc': self.load_node})
        node_get_keywords = TaskNode(self.extract_keywords)(
            {'tokens': node_get_tokens, 'doc': self.load_node})
        keywords_dumper = DumpNode(_dump_kwards)({
            'keywords': node_get_keywords,
            'doc': self.load_node
        })
        node_match = TaskNode(self.match)({
            'query_doc': self.load_node,
            'keywords': node_get_keywords
        })
        (self.dump_node < node_match)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, keywords_dumper])
        return flow
