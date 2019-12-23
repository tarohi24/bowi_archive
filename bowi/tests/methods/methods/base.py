from pathlib import Path
import pytest

from bowi.methods.common.types import Context
from bowi.models import Document
from bowi.methods.common.cache import KeywordCacher
from bowi import settings


__all__ = ('context', 'text', 'doc', 'patch_cachedir')


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


@pytest.fixture
def patch_cachedir(monkeypatch) -> None:
    new_cache_dir: Path = settings.project_root / 'bowi/tests/cache'
    monkeypatch.setattr(settings,
                        'cache_dir',
                        new_cache_dir)
