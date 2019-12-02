from bowi.methods.methods.fuzzy.tokenize import get_all_tokens
from bowi.tests.methods.methods.base import doc  # noqa


def test_get_all_tokens(doc):  # noqa
    assert get_all_tokens(doc) == 'test test test danger danger da_'.split()
