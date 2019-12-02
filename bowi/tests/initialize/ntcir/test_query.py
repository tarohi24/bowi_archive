import json
from pathlib import Path
from typing import Dict, Set

import pytest
import xml.etree.ElementTree as ET

from bowi.initialize.converters.ntcir import NTCIRConverter
from bowi.initialize.ntcir.query import loading, replace_tab, get_document
from bowi.models import Document
from bowi import settings


@pytest.fixture(autouse=True)
def patch_data_dir(monkeypatch):
    monkeypatch.setattr(settings, 'data_dir',
                        settings.project_root.joinpath('bowi/tests/data'))


@pytest.fixture
def converter() -> NTCIRConverter:
    return NTCIRConverter()


def test_load_queries():
    lst: Set[Path] = set([path.name for path in loading()])
    assert lst == {'1001', '1002', '1003'}


@pytest.fixture
def root(patch_data_dir) -> ET.Element:
    path: Path = settings.data_dir.joinpath(f'ntcir/orig/query/1001')
    return replace_tab(path)


def test_attributes(root, converter):
    assert converter._get_docid(root) == '200106296192'
    assert converter._get_tags(root) == ['G06K', ]
    assert converter._get_title(root) == 'Machine-readable record with a two-dimensional lattice of synchronization code interleaved with data code'  # noqa


def test_get_document(root):
    doc: Document = get_document(root)
    assert doc.docid == '200106296192'


def test_to_json(root):
    doc: Document = get_document(root)
    dic: Dict = json.loads(doc.to_json())
    assert dic['docid'] == '200106296192'
