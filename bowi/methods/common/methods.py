from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar, Generic, Generator, Type, TypeVar  # type: ignore

from typedflow.flow import Flow
from typedflow.nodes import LoaderNode, DumpNode

from bowi.methods.common.loader import load_query_files
from bowi.methods.common.dumper import dump_prel
from bowi.methods.common.types import Context, TRECResult
from bowi.models import Document


T = TypeVar('T')


@dataclass
class Method(Generic[T]):
    context: Context
    param: T
    param_type: ClassVar[Type[T]] = field(init=False)
    load_node: LoaderNode[Document] = field(init=False)
    dump_node: DumpNode[TRECResult] = field(init=False)

    def __post_init__(self):

        def get_queries() -> Generator[Document, None, None]:
            return load_query_files(dataset=self.context.es_index)

        def dump_result(res: TRECResult) -> None:
            dump_prel(res=res, context=self.context)

        self.load_node: LoaderNode[Document] = LoaderNode(
            func=get_queries,
            batch_size=1)
        self.dump_node: DumpNode[TRECResult] = DumpNode(func=dump_result)

    def create_flow(self,
                    debug: bool = False) -> Flow:
        ...
