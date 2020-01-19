"""
Word Mover Distance
"""
from collections import Counter
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Tuple

from dataclasses_jsonschema import JsonSchemaMixin
import numpy as np
import pulp
from scipy.spatial.distance import euclidean

from docsim.elas.search import EsResult
from docsim.embedding.fasttext import FastText
from docsim.methods.base import Method, Param
from docsim.models import QueryDocument, RankItem
from docsim.text import Filter, TextProcessor


@dataclass
class WMDParam(Param, JsonSchemaMixin):
    n_words: int


@dataclass
class WMD(Method):
    param: WMDParam
    fasttext: FastText = field(default_factory=FastText.create)

    def count_prob(self,
                   tokens: List[str]) -> Dict[str, float]:
        """
        Compute emergence probabilities for each word
        """
        n_tokens: int = len(tokens)
        counter: Dict[str, int] = Counter(tokens)
        return {word: counter[word] / n_tokens for word in counter.keys()}

    def wmd(self,
            A_tokens: List[str],
            B_tokens: List[str]) -> float:
        """
        Return
        ---------------------
        Similarity
        """
        all_tokens: List[str] = list(set(A_tokens) | set(B_tokens))

        A_prob: Dict[str, float] = self.count_prob(A_tokens)
        B_prob: Dict[str, float] = self.count_prob(B_tokens)
        wv: Dict[str, np.ndarray] = {token: self.fasttext.embed(token) for token in all_tokens}
        var_dict = pulp.LpVariable.dicts(
            'T_matrix',
            list(product(all_tokens, all_tokens)),
            lowBound=0)
        prob = pulp.LpProblem('WMD', sense=pulp.LpMinimize)
        prob += pulp.lpSum([var_dict[token1, token2] * euclidean(wv[token1], wv[token2])
                            for token1, token2 in product(all_tokens, all_tokens)])

        for token2 in B_prob.keys():
            prob += pulp.lpSum(
                [var_dict[token1, token2] for token1 in B_prob.keys()]
            ) == B_prob[token2]
        for token1 in A_prob.keys():
            prob += pulp.lpSum(
                [var_dict[token1, token2] for token2 in A_prob.keys()]
            ) == A_prob[token1]
        prob.solve()
        return -pulp.value(prob.objective)

    def retrieve(self,
                 query_doc: QueryDocument,
                 size: int = 100) -> RankItem:
        filters: List[Filter] = self.get_default_filtes(
            n_words=self.param.n_words)
        processor: TextProcessor = TextProcessor(filters=filters)
        q_words: List[str] = processor.apply(query_doc.text)
        candidates: EsResult = self.filter_by_terms(
            query_doc=query_doc,
            n_words=self.param.n_words,
            size=size)

        tokens_dict: Dict[Tuple[str, str], List[str]] = {
            (hit.source['docid'], hit.source['tags'][0]): processor.apply(hit.source['text'])
            for hit in candidates.hits
        }

        scores: Dict[Tuple[str, str], float] = dict()
        for key, tokens in tokens_dict.items():
            score: float = self.wmd(q_words, tokens)
            scores[key] = score

        return RankItem(query_id=query_doc.docid, scores=scores)
