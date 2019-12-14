"""
Note: This module assserts keywords are generated
in advance. Run fuzzy.naive in the same params
before running this script
"""
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Type, Tuple

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, LoaderNode, DumpNode
from tqdm import tqdm

from bowi.embedding.base import mat_normalize
from bowi.embedding.fasttext import FastText
from bowi.methods.common.methods import Method
from bowi.methods.common.pre_filtering import load_cols
from bowi.methods.common.types import TRECResult
from bowi.methods.common.dumper import get_dump_dir
from bowi.elas.client import EsClient
from bowi import settings

from bowi.methods.methods.fuzzy.param import FuzzyParam


logger = logging.getLogger(__file__)


@dataclass
class QueryKeywords:
    docid: str
    keywords: List[str]


@dataclass
class TfidfEmb:
    words: List[str]
    tfs: np.ndarray  # int
    idfs: np.ndarray  # float
    embs: np.ndarray  # (n_words, n_dim)


@dataclass
class FuzzyRerank(Method[FuzzyParam]):
    param_type: ClassVar[Type] = FuzzyParam
    fasttext: FastText = field(init=False)

    def __post_init__(self):
        super(FuzzyRerank, self).__post_init__()
        self.fasttext: FastText = FastText()

    def load_keywords(self) -> List[QueryKeywords]:
        path: Path = settings.cache_dir.joinpath(
            f'{self.context.es_index}/keywords/fuzzy.naive/{str(self.param.n_words)}.json')
        with open(path) as fin:
            data: Dict[str, List[str]] = json.load(fin)
        lst: List[QueryKeywords] = [QueryKeywords(docid=key, keywords=val)
                                    for key, val in data.items()]
        return lst

    def _get_tfidf_emb(self,
                       docid: str,
                       es_index: str) -> TfidfEmb:
        previous_level = logging.root.manager.disable
        logging.disable(logging.WARN)  # supress logger of Elasticsearch
        es_client: EsClient = EsClient(es_index=es_index)
        tfidf_dict: Dict[str, Tuple[int, float]] = {
            word: tfidf
            for word, tfidf in es_client.get_tfidfs(docid=docid).items()
            if self.fasttext.isin_vocab(word) and tfidf[0] >= self.param.min_tf
        }
        words, tfidfs = list(zip(*tfidf_dict.items()))
        tfs, idfs = [np.array(lst) for lst in list(zip(*tfidfs))]
        embs: np.ndarray = self.fasttext.embed_words(words)
        logging.disable(previous_level)
        return TfidfEmb(words=words, tfs=tfs, idfs=idfs, embs=embs)

    def load_query(self,
                   qk: QueryKeywords) -> TfidfEmb:
        docid: str = qk.docid
        tfidf_emb: TfidfEmb = self._get_tfidf_emb(docid=docid,
                                                  es_index=f'{self.context.es_index}_query')
        return tfidf_emb

    def get_col_tfidfs(self,
                       qk: QueryKeywords) -> Dict[str, TfidfEmb]:
        """
        Get documents cached by keywrod search
        """
        tfidf_embs: Dict[str, TfidfEmb] = dict()
        for doc in tqdm(load_cols(docid=qk.docid,
                                  runname='100',
                                  dataset=self.context.es_index),
                        desc='Getting colembs...'):
            tfidf_embs[doc.docid] = self._get_tfidf_emb(docid=doc.docid,
                                                        es_index=self.context.es_index)
        return tfidf_embs

    def get_kembs(self,
                  qk: QueryKeywords) -> np.ndarray:
        """
        Find pre-generated keywords and embed them.
        """
        embs: np.ndarray = np.array(self.fasttext.embed_words(qk.keywords))
        return mat_normalize(embs)

    def _get_nns(self,
                 mat: np.ndarray,
                 keyword_embs: np.ndarray) -> List[int]:
        """
        Given embedding matrix mat, Get nearest keywords in keyword_embs

        Return
        -----
        list (len = mat.shape[0]) of nearest embedding's indexes
        """
        nns: List[int] = np.argmax(
            np.dot(mat, keyword_embs.T), axis=1).tolist()
        return nns

    def to_fuzzy_bows(self,
                      tfidf_emb: TfidfEmb,
                      keyword_embs: np.ndarray) -> np.ndarray:
        """
        Generate a FuzzyBoW vector according to keyword_embs

        Return
        -----
        1D array whose item[i] is the normalized frequency of ith keyword
        """
        mat: np.ndarray = tfidf_emb.embs
        # tfidfs: np.ndarray = tfidf_emb.tfs * tfidf_emb.idfs
        tfidfs: np.ndarray = tfidf_emb.tfs
        assert tfidfs.ndim == 1
        nns: List[int] = self._get_nns(mat=mat, keyword_embs=keyword_embs)
        scores: np.ndarray = np.array([0.0 for _ in range(len(keyword_embs))])
        for i in range(len(nns)):
            scores[nns[i]] += tfidfs[i]
        return scores / np.sum(scores)  # normalize

    def get_collection_fuzzy_bows(self,
                                  col_items: Dict[str, TfidfEmb],
                                  keyword_embs: np.ndarray) -> Dict[str, np.ndarray]:
        bow_dict: Dict[str, np.ndarray] = {
            docid: self.to_fuzzy_bows(tfidf_emb=item, keyword_embs=keyword_embs)
            for docid, item in col_items.items()
        }
        return bow_dict

    def match(self,
              qk: QueryKeywords,
              query_bow: np.ndarray,
              col_bows: Dict[str, np.ndarray]) -> TRECResult:
        """
        Yet this only computes cosine similarity as the similarity.
        There's room for adding other ways.
        """
        # dot is inadequate
        scores: Dict[str, float] = {docid: np.dot(query_bow, bow)
                                    for docid, bow in tqdm(col_bows.items(),
                                                           desc='Computing bow...')}
        return TRECResult(query_docid=qk.docid,
                          scores=scores)

    def dump_bow(self,
                 qk: QueryKeywords,
                 bow: np.ndarray) -> None:
        bow_str: str = ','.join([str(val) for val in bow])
        with open(get_dump_dir(context=self.context).joinpath('bow.json'), 'a') as fout:
            fout.write(bow_str + '\n')

    def create_flow(self, debug: bool = False):
        # loading query
        loader: LoaderNode = LoaderNode(func=self.load_keywords,
                                        batch_size=1)
        n_get_cols = TaskNode(self.get_col_tfidfs)({'qk': loader})
        n_get_kembs = TaskNode(self.get_kembs)({'qk': loader})

        # generate fBoW of the query
        n_load_query = TaskNode(self.load_query)({'qk': loader})
        n_fbow = TaskNode(self.to_fuzzy_bows)({
            'tfidf_emb': n_load_query,
            'keyword_embs': n_get_kembs
        })

        # generate fBoW of collection
        n_cfbow = TaskNode(self.get_collection_fuzzy_bows)({
            'col_items': n_get_cols,
            'keyword_embs': n_get_kembs
        })

        # integration
        n_match = TaskNode(self.match)({
            'qk': loader,
            'query_bow': n_fbow,
            'col_bows': n_cfbow
        })

        node_bow_dumper = DumpNode(self.dump_bow)({
            'qk': loader,
            'bow': n_fbow
        })
        (self.dump_node < n_match)('res')
        flow: Flow = Flow(dump_nodes=[self.dump_node, node_bow_dumper],
                          debug=debug)
        flow.typecheck()
        return flow
