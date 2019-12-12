import pytest

from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.methods.methods.fuzzy.naive import FuzzyNaive
from bowi.tests.methods.methods.base import context  # noqa

from bowi.tests.embedding.fasttext import FTMock


@pytest.fixture
def param() -> FuzzyParam:
    return FuzzyParam(
        n_words=2,
        model='fasttext',
        prefilter_name='100'
    )


@pytest.fixture
def fuzzy(mocker, param, context) -> FuzzyNaive:  # noqa
    return FuzzyNaive(param=param, context=context)


@pytest.fixture(autouse=True)
def mock_ft(mocker):
    mocker.patch('bowi.methods.methods.fuzzy.naive.FastText', new=FTMock)


def test_flow(fuzzy):
    fuzzy.create_flow().typecheck()
