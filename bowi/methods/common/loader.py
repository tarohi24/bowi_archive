"""
Query loader
"""
from pathlib import Path
from typing import Generator

from tqdm import tqdm
from typedflow.nodes import LoaderNode

from bowi import settings
from bowi.methods.common.types import Context
from bowi.models import Document


__all__ = ['load_query_files', ]


def load_query_files(dataset: str) -> Generator[Document, None, None]:
    qpath: Path = settings.data_dir.joinpath(f'{dataset}/query/dump.bulk')
    pbar = tqdm()
    with open(qpath) as fin:
        while (line := fin.readline()):
            doc: Document = Document.from_json(line)  # type: ignore
            yield doc
            pbar.update(1)
