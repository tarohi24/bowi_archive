from typing import Any, Dict, List, Set, Generator

import pytest
import numpy as np
from typedflow.batch import Batch
from typedflow.nodes import LoaderNode
from typedflow.exceptions import EndOfBatch

from bowi.models import Document
from bowi.embedding import fasttext
from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.methods.methods.fuzzy import rerank
from bowi.tests.methods.methods.base import context  # noqa

from bowi.tests.embedding.fasttext import FTMock


@pytest.fixture
def param() -> FuzzyParam:
    return FuzzyParam(
        n_words=3,
        model='fasttext',
        coef=1,
        prefilter_name='100'
    )


def load_keywords() -> List[rerank.QueryKeywords]:
    return [rerank.QueryKeywords(docid='AAA', keywords='hey jude'.split())]
            

@pytest.fixture
def model(param, context, mocker) -> rerank.FuzzyRerank:  # noqa
    mocker.patch.object(rerank, 'FastText', FTMock)
    model = rerank.FuzzyRerank(param=param, context=context)
    model.load_keywords = load_keywords
    return model


def get_tokens() -> List[str]:
    tokens: List[str] = 'hello world everyone'.split()
    return tokens


def test_fuzzy_bows(mocker, model):
    mat = model.fasttext.embed_words(get_tokens())
    embs = mat[:1]
    bow: np.ndarray = model.to_fuzzy_bows(mat, embs)
    ones: np.ndarray = np.ones(embs.shape[0])
    np.testing.assert_array_almost_equal(bow, ones / np.sum(ones))

    # 2 keywords
    mocker.patch.object(model.param, 'n_words', 2)
    embs = mat[:2]
    assert embs.shape[0] == 2
    sorted_sims: np.ndarray = np.sort(model.to_fuzzy_bows(mat, embs))
    desired = np.sort([2 / 3, 1 / 3])
    np.testing.assert_array_almost_equal(sorted_sims, desired)


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
    flow.typecheck()
