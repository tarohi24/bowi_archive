from pathlib import Path
from typing import Dict, List, Match, Optional

import pytest

from bowi.elas.client import EsClient

from bowi.clf.clf import KNNClassifier


@pytest.fixture
def clf() -> KNNClassifier:
    return KNNClassifier()


@pytest.fixture
def prel_path(tmpdir) -> Path:
    path: Path = Path(tmpdir) / 'a.prel'
    with open(path, 'w') as fout:
        s: str = '''
EP1780094A1 Q0 EP0351087A2 1 621.735800654052 STANDARD
EP1780094A1 Q0 EP0351087B1 2 584.3542763869701 STANDARD
'''
        fout.write(s)
    return path


def test_prel_regex(clf):
    s: str = 'EP1780094A1 Q0 EP0351087A2 1 621.735800654052 STANDARD'
    match: Optional[Match] = clf.prel_pat.match(s)
    assert match is not None
    assert match.group('qid') == 'EP1780094A1'
    assert match.group('relid') == 'EP0351087A2'
    assert match.group('rank') == '1'
    assert match.group('score') == '621.735800654052'


def test_load_prel(prel_path, clf):
    res: Dict[str, List[str]] = clf._load_prel_file(prel_path)
    assert list(res.keys()) == ['EP1780094A1']
    assert res == {
        'EP1780094A1': ['EP0351087A2', 'EP0351087B1']
    }


def test_clf(mocker, prel_path, clf):
    def get_docdic(docid: str) -> Dict:
        tags: List[str] = {
            'EP0351087A2': ['AAAA', 'BBBB'],
            'EP0351087B1': ['BBBB', ],
        }[docid]
        return {'tags': tags}

    client_mock = mocker.MagicMock(spec=EsClient)
    client_mock.get_source = get_docdic
    clf.clf(prel_file=prel_path, es_client=client_mock)
