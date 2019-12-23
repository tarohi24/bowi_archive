from pathlib import Path

import pytest

from bowi.methods.common.types import Context
from bowi.methods.common.cache import KeywordCacher
from bowi import settings


@pytest.fixture
def context() -> Context:
    return Context(
        es_index='dummy',
        method='naive',
        runname='30',
        n_docs=100)


@pytest.fixture(autouse=True)
def patch_cache_dir(monkeypatch, tmpdir):
    monkeypatch.setattr(settings, 'cache_dir', Path(tmpdir))


def test_keyword_cacher_path(context):
    cacher = KeywordCacher(context=context)
    assert cacher._get_dump_path() == settings.cache_dir\
        .joinpath('dummy/keywords/naive/30.keywords')
