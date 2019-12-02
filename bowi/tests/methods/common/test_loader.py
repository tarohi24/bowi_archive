from typing import Generator

from typedflow.nodes import LoaderNode

from bowi.models import Document
from bowi.methods.common.loader import load_query_files


def node():
    def get_queries() -> Generator[Document, None, None]:
        return load_query_files(dataset='clef')
    node: LoaderNode[Document] = LoaderNode(func=get_queries)
    return node


def test_node_creation():
    node()
