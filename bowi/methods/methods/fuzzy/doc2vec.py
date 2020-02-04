from dataclasses import dataclass, field
import datetime
import logging
from typing import ClassVar, Type, List, Dict

from gensim.models.doc2vec import Doc2Vec as GDoc2vec
import numpy as np
from typedflow.nodes import TaskNode
from typedflow.flow import Flow

from bowi.methods.common.methods import Method
from bowi.methods.methods.keywords import KeywordBaseline, KeywordParam
from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.models import Document
from bowi.elas.search import EsClient, EsResult
from bowi.methods.common.types import TRECResult
from bowi.methods.common.types import Param
from bowi.settings import project_root


logger = logging.getLogger(__file__)


@dataclass
class Doc2vecParam(Param):
    n_words: int


@dataclass
class Doc2vec(Method[FuzzyParam]):
    param_type: ClassVar[Type] = Doc2vecParam
    kb: KeywordBaseline = field(init=False)
    escl: EsClient = field(init=False)
    model: GDoc2vec = field(init=False)

    def __post_init__(self):
        super(Doc2vec, self).__post_init__()
        kb_param: KeywordParam = KeywordParam(n_words=100)
        self.kb = KeywordBaseline(param=kb_param,
                                  context=self.context)
        self.escl = EsClient(self.context.es_index)
        logger.info('loading fasttext...')
        self.model = GDoc2vec.load(
            str((project_root / f'doc2vec/clef.model').resolve()))

    def filter_in_advance(self,
                          doc: Document) -> EsResult:
        keywords: List[str] = self.kb.extract_keywords(doc)
        res: EsResult = self.kb.search(doc, keywords)
        return res

    def retrieve(self,
                 doc: Document,
                 res: EsResult) -> TRECResult:
        q_tokens: List[str] = self.kb.escl_query.get_tokens_from_doc(doc.docid)
        # (d, )
        x: np.ndarray = self.model.infer_vector(q_tokens)
        scores: Dict[str, float] = dict()
        for docid in res.get_ids():
            tokens: List[str] = self.escl.get_tokens_from_doc(docid)
            y: np.ndarray = self.model.infer_vector(tokens)
            y /= np.linalg.norm(y)
            assert len(x) == len(y)
            scores[docid] = np.dot(x, y)
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
        flow: Flow = Flow(dump_nodes=[self.dump_node, self.dump_time_node],
                          debug=debug)
        return flow
