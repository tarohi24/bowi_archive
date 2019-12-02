from typing import List

import pytest

from bowi.elas.search import EsResult, EsResultItem
from bowi.methods.common.types import TRECResult
from bowi.methods.methods.keywords import KeywordParam, KeywordBaseline, extract_keywords_from_text

from bowi.tests.methods.methods.base import context, doc, text  # noqa
from bowi.testing.patches import patch_data_dir


@pytest.fixture
def param() -> KeywordParam:
    param: KeywordParam = KeywordParam(n_words=2)
    return param


@pytest.fixture
def method(param, context):
    method: KeywordBaseline = KeywordBaseline(context=context, param=param)
    return method


@pytest.fixture
def sample_hits():
    res = EsResult([
        EsResultItem.from_dict(
            {'_source': {'docid': 'EP200'}, '_score': 3.2}),
    ])
    return res


def test_extract_query_from_text(method, text):
    keywords: List[str] = extract_keywords_from_text(text=text,
                                                     n_words=2)
    assert keywords == ['test', 'danger', ]


def test_extract_keyword(method, doc):
    keywords: List[str] = method.extract_keywords(doc=doc)
    assert keywords == ['test', 'danger', ]


def test_integrated(mocker, method, doc, sample_hits):
    mocker.patch('bowi.settings.es', 'foo')
    mocker.patch('bowi.elas.search.EsSearcher.search',
                 return_value=sample_hits)
    search_res: EsResult = method.search(doc=doc, keywords=[])
    trec_res: TRECResult = method.to_trec_result(doc=doc, es_result=search_res)
    assert trec_res.query_docid == 'EP111'
    assert trec_res.scores == {'EP200': 3.2}


def test_flow_creation(mocker, method, sample_hits):
    mocker.patch('bowi.settings.es', 'foo')
    mocker.patch('bowi.elas.search.EsSearcher.search',
                 return_value=sample_hits)
    method.create_flow().run()
