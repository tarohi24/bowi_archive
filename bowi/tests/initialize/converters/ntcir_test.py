from pathlib import Path
from typing import List

import pytest

from bowi.initialize.converters.ntcir import NTCIRConverter
from bowi.models import Document
from bowi import settings


@pytest.fixture(autouse=True)
def patch_data_dir(monkeypatch):
    monkeypatch.setattr(settings, 'data_dir',
                        settings.project_root.joinpath('bowi/tests/data'))


@pytest.fixture
def converter() -> NTCIRConverter:
    return NTCIRConverter()


@pytest.fixture
def test_file() -> Path:
    test_file: Path = settings.data_dir.joinpath(
        'ntcir/orig/collection/sample.txt')
    return test_file


@pytest.fixture
def docs(converter, test_file) -> List[Document]:
    docs: List[Document] = list(converter.to_document(test_file))
    return docs


def test_docs_properties(docs):
    assert docs[0].title\
        == 'Process for making improved corrosion preventive zinc cyanamide'
    assert docs[0].docid == '199305176894'
    assert docs[0].tags == ['C01C']
    assert docs[0].text.split()[:3] == 'The invention will'.split()
