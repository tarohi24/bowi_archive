"""
Available only for fasttext
"""
from __future__ import annotations
from dataclasses import dataclass, field
import logging
from typing import ClassVar, List, Type, Optional, Dict, Tuple

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

from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.methods.methods.fuzzy.fuzzy import get_keyword_inds


logger = logging.getLogger(__file__)


@dataclass
class FuzzyNaive(Method[FuzzyParam]):
    param_type: ClassVar[Type] = FuzzyParam
    fasttext: FastText = field(init=False)
    es_client: EsClient = field(init=False)

    def __post_init__(self):
        super(FuzzyNaive, self).__post_init__()
        self.fasttext: FastText = FastText()
        # note that this client is used for processing queries
        self.es_client: EsClient = EsClient(
            es_index=f'{self.context.es_index}_query')

    def extract_keywords(self,
                         doc: Document) -> List[str]:
        tfidf_dict: Dict[str, Tuple[int, float]] = {
            word: tfidf
            for word, tfidf in self.es_client.get_tfidfs(docid=doc.docid).items()
            if self.fasttext.isin_vocab(word) and tfidf[0] >= self.param.min_tf
        }
        words, tfidfs = list(zip(*tfidf_dict.items()))
        tfs, idfs = [np.array(lst) for lst in list(zip(*tfidfs))]
        embs: np.ndarray = mat_normalize(self.fasttext.embed_words(words))
        key_inds: List[int] = get_keyword_inds(embs=embs,
                                               keyword_embs=None,
                                               n_keywords=self.param.n_words,
                                               tfs=tfs,
                                               idfs=idfs)
        keywords: List[str] = [words[i] for i in key_inds]
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

    def create_flow(self, debug: bool = False):

        def _dump_kwards(keywords: List[str],
                         doc: Document) -> None:
            return dump_keywords(keywords=keywords, doc=doc, context=self.context)

        node_get_keywords = TaskNode(self.extract_keywords)(
            {'doc': self.load_node})
        keywords_dumper = DumpNode(_dump_kwards)({
            'keywords': node_get_keywords,
            'doc': self.load_node
        })
        node_match = TaskNode(self.match)({
            'query_doc': self.load_node,
            'keywords': node_get_keywords
        })
        (self.dump_node < node_match)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, keywords_dumper],
                          debug=debug)
        flow.typecheck()
        return flow
