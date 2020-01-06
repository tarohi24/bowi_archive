"""
Topic ranking w/o topic modeling
"""
from dataclasses import dataclass, field
import logging
from typing import ClassVar, Counter, Dict, List, Type  # type: ignore

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode

from bowi.models import Document
from bowi.elas.client import EsClient
from bowi.methods.common.methods import Method
from bowi.methods.common.types import Param, TRECResult
from bowi.methods.common.cache import DFCacher


logger = logging.getLogger(__file__)


@dataclass
class TopicParam(Param):
    n_words: int = 30  # words per cluster
    n_topics: int = 5


@dataclass
class WordMatrix:
    mat: np.ndarray
    word_to_id: Dict[str, int]
    id_to_word: Dict[int, str] = field(init=False)

    def __post_init__(self):
        self.id_to_word: Dict[float, str] = {i: word for word, i in self.word_to_id.items()}


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
                                 tokens: List[str]) -> WordMatrix:
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
        for i, part in enumerate(splitted):
            counter: Counter[str] = Counter(part)
            for word, freq in counter.items():
                wid: int = vocab[word]
                mat[(wid, i)] = freq

        # normalize
        mat = (mat.T / mat.sum(axis=1)).T
        return WordMatrix(mat=mat, word_to_id=vocab)

    def get_keywords(self,
                     word_mat: WordMatrix) -> List[List[str]]:
        # TODO: this is too naive
        # top n_words
        keywords: List[List[str]] = []
        for arr in word_mat.mat:
            ind: np.ndarray = arr.argsort()[-1 * self.param.n_words:][::-1]
            keywords.append([word_mat.id_to_word[i] for i in ind])
        return keywords
            
            
                     
