"""
Topic ranking w/o topic modeling
"""
from collections import defaultdict
from dataclasses import dataclass, field
import logging
from typing import ClassVar, Counter, Dict, List, Type, Tuple  # type: ignore

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode

from bowi.models import Document
from bowi.elas.client import EsClient
from bowi.elas.search import EsSearcher
from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param, TRECResult
from bowi.methods.common.cache import KNNCacher, DFCacher
from bowi import settings


logger = logging.getLogger(__file__)


@dataclass
class TopicParam(Param):
    n_words: int = 30  # words per cluster
    n_topics: int = 5


@dataclass
class Topicrank(Method[TopicParam]):
    param_type: ClassVar[Type] = TopicParam
    df_cacher: DFCacher = field(init=False)
    es_topic_client: EsClient = field(init=False)

    def __post_init__(self):
        self.df_cacher = DFCacher(dataset=self.context.es_index)
        self.es_topic_client = EsClient(f'{self.context.es_index}_query')

    def get_all_tokens(self,
                       doc: Document) -> List[str]:
        tokens: List[str] = self.es_topic_client.get_tokens_from_doc(doc.docid)
        return tokens

    def get_portion_of_each_word(self,
                                 tokens: List[str],
                                 splitted_doc: List[List[str]]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Return
        -----
        (word, topics), wid
        """
        doclen: int = len(tokens)
        chunk_size: int = doclen // self.param.n_topics
        splitted: List[List[str]] = [
            tokens[i * chunk_size:(i + 1) * chunk_size]
            for i in range(self.param.n_topics)]

        vocab: Dict[str, int] = {word: i for i, word in enumerate(set(tokens))}
        mat: np.ndarray = np.zeros((len(vocab), self.param.n_topics))
        for i, part in enumerate(splitted_doc):
            counter: Counter[str] = Counter(part)
            for word, freq in counter.items():
                wid: int = vocab[word]
                mat[(wid, i)] = freq

        # normalize
        mat = (mat.T / mat.sum(axis=1)).T
        return mat, vocab
