"""
load -> create an instance -> insert into ES
"""
import logging
from lxml import etree
from pathlib import Path
from typing import Generator, List
import xml.etree.ElementTree as ET

from tqdm import tqdm
from typedflow.typedflow import Task, DataLoader, Dumper, Pipeline
from typedflow.utils import dump_to_each_file

from bowi.initialize.converters.ntcir import NTCIRConverter
from bowi.models import Document
from bowi.settings import data_dir


converter: NTCIRConverter = NTCIRConverter()
logger = logging.getLogger(__file__)
parser = etree.XMLParser(resolve_entities=False)


def loading() -> Generator[ET.Element, None, None]:

    def replace_tab(body: str) -> ET.Element:
        replaced: str = body.replace('<tab>', ' ').replace('&', 'and')
        elem: ET.Element = ET.fromstring(replaced, parser=parser)
        return elem

    directory: Path = data_dir.joinpath(f'ntcir/orig/collection/formatted')
    for path in tqdm(directory.glob(f'**/*')):
        if path.is_dir():
            continue
        with open(path) as fin:
            for line in fin.read().splitlines():
                try:
                    yield replace_tab(line)
                except Exception as e:
                    print(repr(e))


def get_document(root: ET.Element) -> Document:
    docid: str = converter._get_docid(root)
    tags: List[str] = converter._get_tags(root)
    title: str = converter._get_title(root)
    text: str = converter._get_text(root)
    doc: Document = Document(docid=docid,
                             title=title,
                             text=text,
                             tags=tags)
    return doc


def get_dump_path(batch_id: int) -> Path:
    path: Path = data_dir.joinpath(f'ntcir/dump/{str(batch_id)}.bulk')
    return path


if __name__ == '__main__':
    gen: Generator[ET.Element, None, None] = loading()
    loader: DataLoader = DataLoader[ET.Element](gen=gen, batch_size=300)
    task: Task = Task[ET.Element, Document](func=get_document)

    dumper: Dumper = Dumper[Document](
        func=lambda batch: dump_to_each_file(batch, get_dump_path))
    pipeline: Pipeline = Pipeline(
        loader=loader,
        pipeline=[task],
        dumper=dumper)
    pipeline.run()
