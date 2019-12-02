import pytest

from bowi.methods.common.types import Context
from bowi.models import Document


__all__ = ('context', 'text', 'doc')


@pytest.fixture
def context() -> Context:
    return Context(
        es_index='dummy',
        method='keywords',
        runname='40',
        n_docs=100)


@pytest.fixture
def text() -> str:
    text: str = 'This is this IS a test. TEST. test; danger Danger da_ is.'
    return text


@pytest.fixture
def doc() -> Document:
    text: str = 'This is this IS a test. TEST. test; danger Danger da_ is.'
    doc: Document = Document(
        docid='EP111',
        title='sample',
        text=text,
        tags=['G10P'])
    return doc
