import logging
from pathlib import Path
from typing import Iterable, Optional, TypeVar
import xml.etree.ElementTree as ET

from bowi.models import ColDocument

logger = logging.getLogger(__file__)
T = TypeVar('T')


class NoneException(Exception):
    pass


class CannotSplitText(Exception):
    pass


def get_or_raise_exception(obj: Optional[T]) -> T:
    if obj is None:
        raise NoneException
    else:
        return obj


def find_text_or_default(root: ET.Element,
                         xpath: str,
                         default: str) -> str:
    try:
        content: ET.Element = get_or_raise_exception(root.find(xpath))
    except NoneException:
        return default
    try:
        text: str = get_or_raise_exception(content.text)
    except NoneException:
        return default
    return text


class Converter:
    """
    convert something into IRBase
    """
    def to_document(self,
                    fpath: Path) -> Iterable[ColDocument]:
        raise NotImplementedError('This is an abstract method.')


class DummyConverter(Converter):
    pass
