"""
load -> create an instance -> insert into ES
"""
import logging
from pathlib import Path
from typing import Generator, List
import xml.etree.ElementTree as ET

import numpy as np
from tqdm import tqdm
from typedflow.nodes import LoaderNode, TaskNode, DumpNode
from typedflow.flow import Flow

from bowi.initialize.converters.clef import CLEFConverter
from bowi.models import Document
from bowi import settings


converter: CLEFConverter = CLEFConverter()
logger = logging.getLogger(__file__)


def loading() -> Generator[Path, None, None]:
    directory: Path = settings.data_dir.joinpath(f'clef/orig/collection/EP')
    for path in tqdm(directory.glob(f'**/*.xml')):
        yield path


def get_document(path: Path) -> Document:
    root: ET.Element = ET.parse(str(path.resolve())).getroot()
    docid: str = converter._get_docid(root)
    tags: List[str] = converter._get_tags(root)
    title: str = converter._get_title(root)
    text: str = converter._get_text(root)
    doc: Document = Document(docid=docid, title=title, text=text, tags=tags)
    return doc


def dump(doc: Document) -> None:
    basedir: Path = settings.data_dir.joinpath(f'clef/dump')
    with open(basedir.joinpath(f'{str(np.random.randint(200))}'), 'a') as fout:
        fout.write(doc.to_json())  # type: ignore
        fout.write('\n')


if __name__ == '__main__':
    loader: LoaderNode = LoaderNode(func=loading, batch_size=300)
    task_get_doc: TaskNode = TaskNode(func=get_document)
    (task_get_doc < loader)('path')
    dumper: DumpNode = DumpNode(func=dump)
    (dumper < task_get_doc)('doc')
    flow: Flow = Flow(dump_nodes=[dumper, ])
    flow.typecheck()
    flow.run()
