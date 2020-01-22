from dataclasses import dataclass, field
import datetime
import logging
from typing import ClassVar, Type, List, Set, Dict

import numpy as np
from typedflow.nodes import TaskNode
from typedflow.flow import Flow

from bowi.methods.common.methods import Method
from bowi.methods.methods.keywords import KeywordBaseline, KeywordParam
from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.embedding.fasttext import FastText
from bowi.models import Document
from bowi.elas.search import EsClient, EsResult
from bowi.methods.common.types import TRECResult


logger = logging.getLogger(__file__)


@dataclass
class FBoW(Method[FuzzyParam]):
    param_type: ClassVar[Type] = FuzzyParam
    kb: KeywordBaseline = field(init=False)
    ft: FastText = field(init=False)
    escl: EsClient = field(init=False)

    def __post_init__(self):
        super(FBoW, self).__post_init__()
        kb_param: KeywordParam = KeywordParam(n_words=100)
        self.kb = KeywordBaseline(param=kb_param,
                                  context=self.context)
        self.escl = EsClient(self.context.es_index)
        logger.info('loading fasttext...')
        self.ft = FastText()

    def filter_in_advance(self,
                          doc: Document) -> EsResult:
        logger.warn(f'starting time: {datetime.datetime.now()}')
        keywords: List[str] = self.kb.extract_keywords(doc)
        res: EsResult = self.kb.search(doc, keywords)
        return res

    def retrieve(self,
                 doc: Document,
                 res: EsResult) -> TRECResult:
        q_tts: Set[str] = set(self.kb.escl_query.get_tokens_from_doc(doc.docid))
        # (l_x, d)
        X: np.ndarray = self.ft.embed_words(list(q_tts))
        # (d, )
        x: np.ndarray = np.amax(X, axis=0)
        scores: Dict[str, float] = dict()
        for docid in res.get_ids():
            tokens: List[str] = self.escl.get_tokens_from_doc(docid)
            tts: Set[str] = set(tokens)
            # (l_y, d)
            Y: np.ndarray = self.ft.embed_words(list(tts))
            y: np.ndarray = np.amax(Y, axis=0)

            print(x.shape, y.shape)

            assert len(x) == len(y)

            zeros: np.ndarray = np.zeros(len(x))
            r: np.ndarray = np.amin(np.stack([x, y, zeros]), axis=0)
            q: np.ndarray = np.amax(np.stack([x, y, zeros]), axis=0)
            scores[docid] = r.sum() / q.sum()
        logger.warn(f'end time: {datetime.datetime.now()}')
        return TRECResult(doc.docid, scores)

    def create_flow(self,
                    debug: bool = False) -> Flow:
        node_filter = TaskNode(self.filter_in_advance)({  # type: ignore
            'doc': self.load_node,
        })
        node_retrieve = TaskNode(self.retrieve)({  # type: ignore
            'doc': self.load_node,
            'res': node_filter
        })
        (self.dump_node < node_retrieve)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, ],
                          debug=debug)
        return flow
