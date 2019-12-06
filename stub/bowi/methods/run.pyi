from bowi.initialize.cacher import pre_filtering as pre_filtering
from bowi.methods.common.dumper import get_dump_dir as get_dump_dir
from bowi.methods.common.methods import Method as Method
from bowi.methods.common.types import Context as Context, Param as Param
from bowi.methods.methods import keywords as keywords
from bowi.methods.methods.fuzzy import naive as naive, rerank as rerank
from pathlib import Path
from typedflow.flow import Flow as Flow
from typing import List, Type, TypeVar

M = TypeVar('M', bound=Method)

def get_method(method_name: str) -> Type[M]: ...
def parse(path: Path) -> List[M]: ...
def main() -> int: ...
