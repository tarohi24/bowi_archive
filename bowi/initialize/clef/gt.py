"""
Generate ground truth labels
load original qrel -> map a key to the new key -> dump new qrel
"""
import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Generator, List, Optional, Match, Tuple

from typedflow.batch import Batch
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, LoaderNode, DumpNode
from typedflow.tasks import Task, DataLoader, Dumper

from tqdm import tqdm

from bowi.settings import data_dir


def loading() -> Generator[str, None, None]:
    """
    Return
    -----
    e.g. "PAC-1 0 EP-0971326-B1 1"
    """
    path: Path = data_dir.joinpath('clef/orig/gt.qrel')
    with open(path) as fin:
        lines: List[str] = fin.read().splitlines()
    for line in tqdm(lines):
        yield line
    return


def load_mapping() -> Dict[str, str]:

    def _parse(line: str) -> Tuple[str, str]:
        # accepts a line such as "PAC-119_EP-1258304-A1.xml"
        eps_op: Optional[Match[str]] = re.match(
            r'(PAC-\d+)_EP-(\d+)-([A-Z][0-9]).xml', line)
        assert eps_op is not None
        eps: Match[str] = eps_op
        return (eps.group(1), f'EP{eps.group(2)}{eps.group(3)}')

    pac_eps: List[str] = [
        p.name for p in data_dir.joinpath('clef/orig/query/topics').glob('PAC-*.xml')]
    parsed: List[Tuple[str, str]] = [_parse(name) for name in pac_eps]
    dic: Dict[str, str] = {k: v for k, v in parsed}
    return dic


@dataclass
class QrelItem:
    query: str
    target: str
    relevance: int

    def to_json(self) -> str:
        return f'{self.query} 0 {self.target} {str(self.relevance)}'


def parse(line: str,
          mapping: Dict[str, str]) -> QrelItem:
    splitted: List[str] = line.split()
    qi: QrelItem = QrelItem(
        query=mapping[splitted[0]],
        target=splitted[2].replace('-', ''),
        relevance=int(splitted[3]))
    return qi


def dump_to_one_file(batch: Batch[QrelItem],
                     path: Path) -> None:
    with open(path, 'a') as fout:
        for item in batch.data:
            fout.write(item.to_json() + '\n')


def main() -> int:
    mapping: Dict[str, str] = load_mapping()

    # loader
    gen: Generator[str, None, None] = loading()
    loader: DataLoader[Path] = DataLoader(gen=gen)
    loader_node: LoaderNode[Path] = LoaderNode(loader=loader)

    # task
    parse_task: Task = Task[str, QrelItem](
        func=lambda line: parse(line, mapping=mapping))
    task_node: TaskNode[str, QrelItem] = TaskNode(task=parse_task,
                                                  arg_type=str)
    task_node.set_upstream_node('loader', loader_node)

    # dump node
    dump_path: Path = data_dir.joinpath('clef/query/gt.qrel')
    dumper: Dumper = Dumper[QrelItem](
        func=lambda batch: dump_to_one_file(batch, dump_path))
    dump_node: DumpNode[QrelItem] = DumpNode(dumper=dumper,
                                             arg_type=QrelItem)
    dump_node.set_upstream_node('task', task_node)

    flow: Flow = Flow(dump_nodes=[dump_node, ])
    asyncio.run(flow.run())
    return 0


if __name__ == '__main__':
    exit(main())
