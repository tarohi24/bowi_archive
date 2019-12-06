from bowi.initialize.converters import base as base
from bowi.models import Document as Document
from pathlib import Path
from typing import Generator

class NTCIRConverter(base.Converter):
    def escape(self, orig: str) -> str: ...
    def to_document(self, fpath: Path) -> Generator[Document, None, None]: ...
