from bowi.initialize.converters.base import CannotSplitText as CannotSplitText, Converter as Converter, NoneException as NoneException, find_text_or_default as find_text_or_default, get_or_raise_exception as get_or_raise_exception
from bowi.models import Document as Document
from pathlib import Path
from typing import Any, List

logger: Any

class CLEFConverter(Converter):
    def to_document(self, fpath: Path) -> List[Document]: ...
