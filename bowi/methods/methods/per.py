from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable, ClassVar, Dict, List, Type, TypedDict, TypeVar)

from nltk.tokenize import sent_tokenize
import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode

from bowi.embedding.base import mat_normalize
from bowi.embedding.bert import Bert
from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param, TRECResult
from bowi.models import ColDocument
from bowi.methods.common.pre_filtering import load_emb


T = TypeVar('T')
K = TypeVar('K')


class Stragegy(Enum):
    PESSIMICTIC = 1
    OPTIMISTIC = 2


@dataclass
class PerParam(Param):
    n_words: int
    strategy: Stragegy

    @classmethod
    def from_args(cls, args) -> PerParam:
        return PerParam(n_words=args.n_words,
                        strategy=Stragegy[args.strategy])


@dataclass
class Per(Method[PerParam]):
    param_type: ClassVar[Type] = PerParam
    bert: Bert = field(default_factory=Bert)

    def get_bert_embs(self,
                      doc: ColDocument) -> Dict[str, np.ndarray]:
        docid: str = doc.docid
        embs: Dict[str, np.ndarray] = load_emb(
            docid=docid,
            dataset=self.mprop.context['es_index'],
            model='bert')
        return embs

    def embed_query(self,
                    doc: ColDocument) -> np.ndarray:
        sents: List[str] = sent_tokenize(doc.text)
        embeddings: np.ndarray = self.bert.embed_words(sents)
        return mat_normalize(embeddings)

    class QandC(TypedDict):
        query_doc: ColDocument
        query_emb: np.ndarray
        col_emb: Dict[str, np.ndarray]

    def _projection_norm(self,
                         vecs: np.ndarray,
                         bases: np.ndarray) -> float:
        """
        compute the norm of a vector which is a projection of vec
        into bases space
        Note: assert that each vectors are unit vectors (norm == 1.0)

        Parameters
        -----
        vec: 2D
        bases: 2D (n_base, dim)
        """
        comps: np.ndarray = np.dot(bases, vecs.T)
        assert comps.shape == (len(bases), len(vecs))
        norms: np.ndarray = np.linalg.norm(comps, axis=0).reshape(-1)
        assert norms.shape == (len(vecs), )
        return norms.sum()

    def score(self,
              qandc: QandC) -> TRECResult:
        norms: Dict[str, float] = {
            docid: self._projection_norm(qandc['query_emb'], mat)
            for docid, mat in qandc['col_emb'].items()
        }
        res: TRECResult = TRECResult(
            query_docid=qandc['query_doc'].docid,
            scores=norms)
        return res

    def create_flow(self):
        node_col_emb: TaskNode[ColDocument, Dict[str, np.ndarray]] = self.get_node(
            func=self.get_bert_embs,
            arg_type=ColDocument)
        node_query_emb: TaskNode[ColDocument, np.ndarray] = self.get_node(
            func=self.embed_query,
            arg_type=ColDocument)
        node_score: TaskNode[self.QandC, TRECResult] = self.get_node(
            func=self.score,
            arg_type=self.QandC)

        # define the topology
        node_col_emb.set_upstream_node('load', self.mprop.load_node)
        node_query_emb.set_upstream_node('load', self.mprop.load_node)

        node_score.set_upstream_node('col_emb', node_col_emb)
        node_score.set_upstream_node('query_doc', self.mprop.load_node)
        node_score.set_upstream_node('query_emb', node_query_emb)

        self.mprop.dump_node.set_upstream_node('score', node_score)

        flow: Flow = Flow(dump_nodes=[self.mprop.dump_node, ])
        return flow
