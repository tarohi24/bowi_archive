"""
load xml -> parse and extract -> dump into a file (query.bulk)
"""
from pathlib import Path
from typing import Generator, List
import xml.etree.ElementTree as ET

from tqdm import tqdm
from typedflow.typedflow import Task, DataLoader, Dumper, Pipeline
from typedflow.utils import dump_to_one_file

from bowi.elas import models
from bowi.intialize.converters.clef import CLEFConverter
from bowi.models import Document
from bowi.settings import data_dir


def loading() -> Generator[Path, None, None]:
    directory: Path = data_dir.joinpath(f'clef/orig/query')
    for path in tqdm(directory.glob(f'topics/*.xml')):
        yield path


def get_document(path: Path) -> Document:
    root: ET.Element = ET.parse(str(path.resolve())).getroot()
    docid: str = converter._get_docid(root)
    tags: List[str] = converter._get_tags(root)
    title: str = converter._get_title(root)
    text: str = converter._get_text(root)
    return Document(docid=models.KeywordField(docid),
                       title=models.TextField(title),
                       text=models.TextField(text),
                       tags=models.KeywordListField(tags))


if __name__ == '__main__':
    converter: CLEFConverter = CLEFConverter()
    gen: Generator[Path, None, None] = loading()
    loader: DataLoader = DataLoader[Path](gen=gen)
    task: Task = Task[Path, Document](func=get_document)
    dump_path: Path = data_dir.joinpath('clef/query/dump.bulk')
    dumper: Dumper = Dumper[Document](
        func=lambda batch: dump_to_one_file(batch, dump_path))
    pipeline: Pipeline = Pipeline(
        loader=loader,
        pipeline=[task],
        dumper=dumper)
    pipeline.run()
