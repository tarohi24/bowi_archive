"""
load -> create an instance -> insert into ES
"""
import logging
from pathlib import Path
from typing import Generator, List
import xml.etree.ElementTree as ET

from tqdm import tqdm
from typedflow.typedflow import Task, DataLoader, Dumper, Pipeline
from typedflow.utils import dump_to_each_file

from bowi.elas import models
from bowi.initialize.converters.clef import CLEFConverter
from bowi.models import Document
from bowi.settings import data_dir


converter: CLEFConverter = CLEFConverter()
logger = logging.getLogger(__file__)


def loading() -> Generator[Path, None, None]:
    directory: Path = data_dir.joinpath(f'clef/orig/collection/EP')
    for path in tqdm(directory.glob(f'**/*.xml')):
        yield path


def get_document(path: Path) -> Document:
    root: ET.Element = ET.parse(str(path.resolve())).getroot()
    docid: str = converter._get_docid(root)
    tags: List[str] = converter._get_tags(root)
    title: str = converter._get_title(root)
    text: str = converter._get_text(root)
    doc: Document = Document(docid=models.KeywordField(docid),
                                   title=models.TextField(title),
                                   text=models.TextField(text),
                                   tags=models.KeywordListField(tags))
    return doc


def get_dump_path(batch_id: int) -> Path:
    path: Path = data_dir.joinpath(f'clef/dump/{str(batch_id)}.bulk')
    return path


if __name__ == '__main__':
    gen: Generator[Path, None, None] = loading()
    loader: DataLoader = DataLoader[Path](gen=gen, batch_size=300)
    get_doc_task: Task = Task[Path, Document](func=get_document)

    dumper: Dumper = Dumper[Document](
        func=lambda batch: dump_to_each_file(batch, get_dump_path))
    pipeline: Pipeline = Pipeline(
        loader=loader,
        pipeline=[get_doc_task],
        dumper=dumper)
    pipeline.run()
