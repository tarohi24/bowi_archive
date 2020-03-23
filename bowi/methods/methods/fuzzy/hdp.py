from dataclasses import dataclass, field
import logging
from typing import ClassVar, Type, List, Dict

import numpy as np
from typedflow.nodes import TaskNode
from typedflow.flow import Flow
from gensim.corpora.dictionary import Dictionary
from gensim.models import HdpModel

from bowi.methods.common.methods import Method
from bowi.methods.methods.keywords import KeywordBaseline, KeywordParam
from bowi.models import Document
from bowi.elas.search import EsClient, EsResult
from bowi.methods.common.types import TRECResult
from bowi.methods.common.types import Param


logger = logging.getLogger(__file__)


@dataclass
class HDPParam(Param):
    n_words: int


@dataclass
class HDP(Method[HDPParam]):
    param_type: ClassVar[Type] = HDPParam
    kb: KeywordBaseline = field(init=False)
    escl: EsClient = field(init=False)

    def __post_init__(self):
        super(HDP, self).__post_init__()
        kb_param: KeywordParam = KeywordParam(n_words=100)
        self.kb = KeywordBaseline(param=kb_param,
                                  context=self.context)
        self.escl = EsClient(self.context.es_index)
        logger.info('loading fasttext...')

    def filter_in_advance(self,
                          doc: Document) -> EsResult:
        keywords: List[str] = self.kb.extract_keywords(doc)
        res: EsResult = self.kb.search(doc, keywords)
        return res

    def retrieve(self,
                 doc: Document,
                 res: EsResult) -> TRECResult:
        q_tokens: List[str] = self.kb.escl_query.get_tokens_from_doc(doc.docid)
        tokens_list: List[List[str]] = [q_tokens, ]
        docids: List[str] = []
        for docid in res.get_ids():
            docids.append(docid)
            tokens: List[str] = self.escl.get_tokens_from_doc(docid)
            tokens_list.append(tokens)
        dictionary: Dictionary = Dictionary(tokens_list)
        corpus: List[List[int]] = [dictionary.doc2bow(toks) for toks in tokens_list]
        id2word: Dictionary = Dictionary.from_corpus(corpus)
        hdp_lda: HdpModel = HdpModel(corpus, id2word)
        mat: np.ndarray = hdp_lda.inference(corpus)
        print(mat.shape)
        assert mat.shape[0] == (len(docids) + 1)

        q_vec, x = mat[0], mat[1:]
        score_vec = np.dot(x, q_vec)
        scores: Dict[str, float] = {
            docid: score
            for docid, score in zip(docids, score_vec)
        }
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
        (self.dump_time_node < node_retrieve)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, self.dump_time_node],
                          debug=debug)
        return flow
