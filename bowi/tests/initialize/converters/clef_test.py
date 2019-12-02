from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET

import pytest

from bowi.initialize.converters.clef import CLEFConverter, NoneException
from bowi import settings


@pytest.fixture(autouse=True)
def patch_data_dir(monkeypatch):
    monkeypatch.setattr(settings, 'data_dir',
                        settings.project_root.joinpath('bowi/tests/data'))


@pytest.fixture
def converter() -> CLEFConverter:
    return CLEFConverter()


@pytest.fixture
def docids() -> List[str]:
    docids: List[str] = ['EP-0050001-A2', 'EP-1010180-B1']
    return docids


@pytest.fixture
def roots(docids) -> List[ET.Element]:

    def to_xml_root(docid: str) -> ET.Element:
        """
        Parameters
        ----------
        docid
            EP... *with* '-'
            e.g. EP-0050001-A2
        """
        first: str = docid[3]
        second: str = docid[4:6]
        third: str = docid[6:8]
        forth: str = docid[8:10]
        fpath: Path = settings.data_dir.joinpath(
            f'clef/orig/collection/00000{first}/{second}/{third}/{forth}/{docid}.xml')  # noqa
        root: ET.Element = ET.parse(str(fpath.resolve())).getroot()
        return root

    roots: List[ET.Element] = [to_xml_root(docid) for docid in docids]
    return roots


def test_get_title(converter, roots):
    titles: List[str] = [
        'A golf aid.',
        'A READ-ONLY MEMORY AND READ-ONLY MEMORY DEVICE',
    ]
    assert titles == [converter._get_title(root) for root in roots]

def test_get_docid(converter, roots, docids):
    assert [docid.replace('-', '') for docid in docids]\
        == [converter._get_docid(root) for root in roots]

def test_get_tags(converter, roots):
    assert set(converter._get_tags(roots[0])) == {'A63B'}
    assert set(converter._get_tags(roots[1])) == {'H01L', 'G11C'}


def test_get_text(converter, roots):
    with pytest.raises(NoneException):
        converter._get_text(roots[0])

    assert 'The present invention'.split()\
        == converter._get_text(roots[1]).split()[:3]
