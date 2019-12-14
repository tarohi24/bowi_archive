import pytest

from bowi.initialize.converters.cmu import ItemNotFound, CmuConveter


@pytest.fixture
def converter():
    return CmuConveter()


def test_get_itemize(converter):
    with pytest.raises(ItemNotFound):
        converter._get_itemize('')
    assert converter._get_itemize('key: value') == ('key', 'value')
