from typing import Any, Dict, List, Set, Generator

import pytest
import numpy as np
from typedflow.batch import Batch
from typedflow.nodes import LoaderNode
from typedflow.exceptions import EndOfBatch

from bowi.embedding.base import mat_normalize
from bowi.models import Document
from bowi.embedding import fasttext
from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.methods.methods.fuzzy import rerank

from bowi.tests.embedding.fasttext import FTMock
from bowi.tests.methods.methods.base import *  # noqa


@pytest.fixture
def param() -> FuzzyParam:
    return FuzzyParam(
        n_words=3,
        model='fasttext',
        prefilter_name='100',
        min_tf=3
    )


@pytest.fixture
def keyword_embs() -> np.ndarray:
    return mat_normalize(np.random.rand(2, 300))


@pytest.fixture
def model(param, context, mocker, patch_cachedir) -> rerank.FuzzyRerank:
    mocker.patch.object(rerank, 'FastText', FTMock)
    model = rerank.FuzzyRerank(param=param, context=context)
    return model


@pytest.fixture
def tfidf_emb() -> rerank.TfidfEmb:
    words: List[str] = 'hello world everyone'.split()
    tfs: np.ndarray = np.arange(len(words)) + 1
    idfs: np.ndarray = np.array([1.1, 2.1, 3.2])
    embs: np.ndarray = mat_normalize(np.random.rand(len(words), 300))
    return rerank.TfidfEmb(words=words, tfs=tfs, idfs=idfs, embs=embs)


def get_tokens() -> List[str]:
    tokens: List[str] = 'hello world everyone'.split()
    return tokens


def test_fuzzy_bows_with_one_keyword(mocker, model, tfidf_emb, keyword_embs):
    # 1 keyword
    bow: np.ndarray = model.to_fuzzy_bows(
        tfidf_emb=tfidf_emb,
        keyword_embs=keyword_embs[:1])
    ones: np.ndarray = np.ones(300)
    assert bow.shape == (1, )
    np.testing.assert_almost_equal(bow[0], 1)


def test_match(mocker, model):
    col_bows: Dict[str, np.ndarray] = {
        'a': np.ones(3),
        'b': np.array([0.5, 0.3, 0.2]),
    }
    col_bows = {docid: vec / np.linalg.norm(vec)  # noqa
                for docid, vec in col_bows.items()}
    qbow = col_bows['a']

    qdoc = mocker.MagicMock()
    qdoc.docid = 'query'
    res = model.match(qk=rerank.QueryKeywords(docid='AAA', keywords='hey jude'.split()),
                      query_bow=qbow,
                      col_bows=col_bows)
    assert res.scores['a'] > res.scores['b']


def test_typecheck(model):
    flow = model.create_flow()
