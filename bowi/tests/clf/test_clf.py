from pathlib import Path
from typing import Dict, List, Match, Optional

import pytest

from bowi.clf.clf import KNNClassifier


@pytest.fixture
def clf() -> KNNClassifier:
    return KNNClassifier()


def test_prel_regex(clf):
    s: str = 'EP1780094A1 Q0 EP0351087A2 1 621.735800654052 STANDARD'
    match: Optional[Match] = clf.prel_pat.match(s)
    assert match is not None
    assert match.group('qid') == 'EP1780094A1'
    assert match.group('relid') == 'EP0351087A2'
    assert match.group('rank') == '1'
    assert match.group('score') == '621.735800654052'


def test_load_prel(tmpdir, clf):
    path: Path = Path(tmpdir) / 'a.prel'
    with open(path, 'w') as fout:
        s: str = '''
EP1780094A1 Q0 EP0351087A2 1 621.735800654052 STANDARD
EP1780094A1 Q0 EP0351087B1 2 584.3542763869701 STANDARD
'''
        fout.write(s)
    res: Dict[str, List[str]] = clf._load_prel_file(path)
    assert list(res.keys()) == ['EP1780094A1']
    assert res == {
        'EP1780094A1': ['EP0351087A2', 'EP0351087B1']
    }
