"""
Word Mover Distance
"""
from dataclasses import dataclass, field
from typing import Dict, List, ClassVar, Type

import logging
import numpy as np
from typedflow.nodes import TaskNode
from typedflow.flow import Flow

from bowi.elas.search import EsClient, EsResult
from bowi.methods.methods.keywords import KeywordParam, KeywordBaseline
from bowi.methods.common.methods import Method
from bowi.methods.common.types import TRECResult, Param
from bowi.embedding.fasttext import FastText
from bowi.models import Document


logger = logging.getLogger(__file__)


@dataclass
class WMDParam(Param):
    n_words: int = 100

    def to_keyword_param(self) -> KeywordParam:
        return KeywordParam(n_words=self.n_words)


@dataclass
class WMD(Method[WMDParam]):
    param_type: ClassVar[Type] = WMDParam
    fasttext: FastText = field(init=False)
    kb: KeywordBaseline = field(init=False)
    escl: EsClient = field(init=False)

    def __post_init__(self):
        super(WMD, self).__post_init__()
        self.fasttext: FastText = FastText()
        self.kb = KeywordBaseline(context=self.context,
                                  param=self.param.to_keyword_param())
        self.escl = EsClient(self.context.es_index)

    def filter_in_advance(self,
                          doc: Document) -> Dict[str, List[str]]:
        keywords: List[str] = self.kb.extract_keywords(doc)
        res: EsResult = self.kb.search(doc, keywords)
        tokens_dict: Dict[str, List[str]] = dict()
        for docid in res.get_ids():
            try:
                tokens: List[str] = self.escl.get_tokens_from_doc(docid)
            except KeyError:
                pass
            tokens_dict[docid] = (tokens)
        return tokens_dict

    def embed_res(self,
                  tokens_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Return
        -----
        {docid: (n_words, dim)}
        """
        embs: np.ndarray = {
            docid: self.fasttext.embed_words(tokens)
            for docid, tokens in tokens_dict.items()
        }
        return embs

    def retrieve(self,
                 doc: Document,
                 embs: Dict[str, np.ndarray]) -> TRECResult:
        scores: Dict[str, float] = dict()
        q_tokens: List[str] = self.kb.escl_query.get_tokens_from_doc(doc.docid)
        # (n_words, dim)
        q_emb: np.ndarray = self.fasttext.embed_words(q_tokens)
        for docid, mat in embs.items():
            # (n_tok_q, n_tok_col)
            dists: np.ndarray = np.dot(q_emb, mat.T)
            # (n_tok_q, )
            costs: np.ndarray = np.amax(dists, axis=1)
            scores[docid] = costs.sum()
        return TRECResult(doc.docid, scores)

    def create_flow(self,
                    debug: bool = False) -> Flow:
        node_filter = TaskNode(self.filter_in_advance)({  # type: ignore
            'doc': self.load_node,
        })
        node_embed = TaskNode(self.embed_res)({
            'tokens_dict': node_filter,
        })
        node_retrieve = TaskNode(self.retrieve)({
            'doc': self.load_node,
            'embs': node_embed,
        })
        (self.dump_node < node_retrieve)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, self.dump_time_node],
                          debug=debug)
        return flow
