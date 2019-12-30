import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from bowi.elas.client import EsClient
from bowi.clf.eval import Evaluator
from bowi import settings


@pytest.fixture
def resdir(tmpdir) -> Path:
    dump_dir: Path = Path(tmpdir) / 'dummy' / 'keywords'
    dump_dir.mkdir(parents=True)
    dummy_res: Dict[str, List[str]] = {
        'A': ['AAA', 'BBB'],
        'B': ['CCC', ],
    }
    with open(dump_dir / 'clf.json', 'w') as fout:
        json.dump(dummy_res, fout)
    return dump_dir


@pytest.fixture(autouse=True)
def client_patch(monkeypatch):

    def get_source_mock(self, docid: str) -> Dict[str, Any]:
        return {'tags': ['AAA', ]}

    monkeypatch.setattr(EsClient, 'get_source', get_source_mock)


@pytest.fixture
def evaluator(resdir) -> Evaluator:
    return Evaluator(res_dir=resdir)


def test_init(evaluator):
    assert evaluator.dataset == 'dummy'


def test_accuracy(evaluator):
    assert evaluator._get_accuracy(docid='A') == 1.0
    assert evaluator._get_accuracy(docid='B') == 0.0
