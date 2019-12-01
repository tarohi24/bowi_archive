"""
Generate ground truth labels
load original qrel -> map a key to the new key -> dump new qrel
"""
from dataclasses import dataclass
import linecache
from pathlib import Path
from typing import Generator, List
import xml.etree.ElementTree as ET

from typedflow.typedflow import Task, DataLoader, Dumper, Pipeline
from typedflow.utils import dump_to_one_file
from tqdm import tqdm

from bowi.initialize.converters.ntcir import NTCIRConverter
from bowi.settings import data_dir


converter: NTCIRConverter = NTCIRConverter()


def loading() -> Generator[str, None, None]:
    """
    Return
    -----
    e.g. 1001    A   PATENT-US-GRT-1997-05611575 1
    """
    path: Path = data_dir.joinpath('ntcir/gt.qrel')
    with open(path) as fin:
        lines: List[str] = fin.read().splitlines()
    for line in tqdm(lines):
        yield line
    return


def get_docid(doc_num: str) -> str:
    """
    Parameters
    -----
    doc_nun
        4 digits number (*string*)

    Return
    -----
    docid (e.g. 199705611575)
    """
    path: Path = data_dir.joinpath(f'ntcir/orig/query/{doc_num}')
    docid_line: str = linecache.getline(str(path.resolve()), 6)
    replaced: str = f'<TOPIC>{docid_line}</TOPIC>'
    elem: ET.Element = ET.fromstring(replaced)
    return converter._get_docid(elem)


@dataclass
class QrelItem:
    query: str
    target: str
    relevance: int

    def to_json(self) -> str:
        s: str = f'{self.query} 0 {self.target} {str(self.relevance)}'
        return s


def parse(line: str) -> QrelItem:
    splitted: List[str] = line.split()
    docid: str = get_docid(splitted[0])
    target: str = ''.join(splitted[2].split('-')[-2:])
    qi: QrelItem = QrelItem(
        query=docid,
        target=target,
        relevance=int(splitted[3]))
    return qi


if __name__ == '__main__':
    gen: Generator[str, None, None] = loading()
    loader: DataLoader = DataLoader[Path](gen=gen)
    parse_task: Task = Task[str, QrelItem](
        func=lambda line: parse(line))
    dump_path: Path = data_dir.joinpath('ntcir/query/gt.qrel')
    dumper: Dumper = Dumper[QrelItem](
        func=lambda batch: dump_to_one_file(batch, dump_path))
    pipeline: Pipeline = Pipeline(
        loader=loader,
        pipeline=[parse_task],
        dumper=dumper)
    pipeline.run()
