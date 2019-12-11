from pathlib import Path
from typing import List, TypeVar

import pytest

from bowi import settings
from bowi.methods.run import get_method, parse, create_parser
from bowi.methods.methods.keywords import KeywordBaseline


P = TypeVar('P')


@pytest.fixture
def sample_yaml() -> Path:
    path: Path = settings.project_root.joinpath('bowi/tests/params/sample.yaml')
    return path


def test_get_method():
    assert get_method('keywords') == KeywordBaseline
    with pytest.raises(KeyError):
        get_method('dummy')


def test_parse(sample_yaml):
    argstrs: List[str] = f'{str(sample_yaml.resolve())} --debug'.split()
    parser = create_parser()
    lst: List[P] = parse(parser.parse_args(argstrs))
    assert len(lst) == 1
    assert lst[0].context.n_docs == 100
    assert lst[0].context.es_index == 'clef'
    assert lst[0].context.method == 'keywords'
    assert lst[0].context.runname == '40'
    assert lst[0].param.n_words == 40
