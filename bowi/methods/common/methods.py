from __future__ import annotations
from dataclasses import dataclass, field
import datetime
from typing import ClassVar, Generic, Generator, Type, TypeVar  # type: ignore

from typedflow.flow import Flow
from typedflow.nodes import LoaderNode, DumpNode

from bowi.methods.common.loader import load_query_files
from bowi.methods.common.dumper import dump_prel, dump_time
from bowi.methods.common.types import Context, TRECResult
from bowi.models import Document


T = TypeVar('T')


@dataclass
class Method(Generic[T]):
    context: Context
    param: T
    param_type: ClassVar[Type[T]] = field(init=False)
    load_node: LoaderNode = field(init=False)
    dump_node: DumpNode = field(init=False)
    dump_time_node: DumpNode = field(init=False)

    def __post_init__(self):

        def get_queries() -> Generator[Document, None, None]:
            return load_query_files(dataset=self.context.es_index)

        def dump_result(res: TRECResult) -> None:
            dump_prel(res=res, context=self.context)

        def _dump_time(res: TRECResult) -> None:
            """
            Parameter
            -----
            doc
                A dummy parameter (prevent typedflow from detecting mismatch)
            """
            now: datetime.datetime = datetime.datetime.now()
            dump_time(docid=res.query_docid, start_time=now, context=self.context)

        self.load_node: LoaderNode = LoaderNode(
            func=get_queries,
            batch_size=1)
        self.dump_node: DumpNode = DumpNode(func=dump_result)
        self.dump_time_node: DumpNode = DumpNode(func=_dump_time)

    def create_flow(self,
                    debug: bool = False) -> Flow:
        ...
