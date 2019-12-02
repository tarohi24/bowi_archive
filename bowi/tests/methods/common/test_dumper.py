from pathlib import Path
from typing import Dict, List, Union

import pytest
from typedflow.batch import Batch
from typedflow.exceptions import FaultItem
from typedflow.nodes import DumpNode

from bowi.methods.common.dumper import get_dump_dir, dump_prel
from bowi.methods.common.types import TRECResult
from bowi.settings import results_dir

from bowi.tests.methods.methods.base import context, doc, text  # noqa


@pytest.fixture
def node(context) -> DumpNode:
    def dump(res: TRECResult) -> None:
        dump_prel(res=res, context=context)
    node: DumpNode= DumpNode(func=dump)
    return node


def get_res() -> TRECResult:
    scores: Dict[str, float] = {
        f'EP10{i}': float(i)
        for i in range(3)
    }
    result: TRECResult = TRECResult(
        query_docid='EP111',
        scores=scores
    )
    return result


def test_path_func(context):
    assert get_dump_dir(context)\
        == results_dir.joinpath('dummy/keywords/40')


def test_to_prel():
    res = get_res()
    assert res.to_prel() == """EP111 Q0 EP102 1 2.0 STANDARD
EP111 Q0 EP101 2 1.0 STANDARD
EP111 Q0 EP100 3 0.0 STANDARD"""


def test_dump(context, node):
    res = get_res()
    data: List[TRECResult] = [{'res': res}, ]
    batch: Batch[TRECResult] = Batch(batch_id=0, data=data)
    path: Path = get_dump_dir(context).joinpath('pred.prel')

    try:
        path.unlink()
    except FileNotFoundError:
        pass
    node.dump(batch=batch)
    with open(path) as fin:
        out: List[str] = fin.read().splitlines()
    assert out == ['EP111 Q0 EP102 1 2.0 STANDARD',
                   'EP111 Q0 EP101 2 1.0 STANDARD',
                   'EP111 Q0 EP100 3 0.0 STANDARD']
    path.unlink()


def test_dump_with_fault(context, node):
    data = [FaultItem(), ]
    batch: Batch[TRECResult] = Batch(batch_id=0, data=data)
    path: Path = get_dump_dir(context).joinpath('pred.prel')

    try:
        path.unlink()
    except FileNotFoundError:
        pass
    node.dump(batch=batch)
    with pytest.raises(FileNotFoundError):
        with open(path) as fin:
            pass

    data: List[Union[TRECResult, FaultItem]] = [{'res': get_res()}, FaultItem(), ]
    batch: Batch[TRECResult] = Batch(batch_id=0, data=data)
    node.dump(batch=batch)
    with open(path) as fin:
        out: List[str] = fin.read().splitlines()
    assert out == ['EP111 Q0 EP102 1 2.0 STANDARD',
                   'EP111 Q0 EP101 2 1.0 STANDARD',
                   'EP111 Q0 EP100 3 0.0 STANDARD']
    path.unlink()
