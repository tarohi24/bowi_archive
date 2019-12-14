from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
import sys
from typing import List, Optional, Pattern, Match, Tuple
import xml.etree.ElementTree as ET

from bowi.initialize.converters.base import (
    Converter,
    CannotSplitText,
    NoneException,
    get_or_raise_exception,
    find_text_or_default
)
from bowi.models import Document


logger = logging.getLogger(__file__)


class ItemNotFound(Exception):
    """
    An exception raised if a reading line is not
    in itemize format
    """
    pass


@dataclass
class CmuConveter(Converter):
    item_pat: Pattern = field(init=False)

    def __post_init__(self):
        self.item_pat = re.compile(r'^(?P<key>[a-zA-Z0-9]+): (?P<val>.+)$')

    def _get_itemize(self,
                     line: str) -> Tuple[str, str]:
        """
        Read line which consists of an item name its value in the
        following format.
        
        Item: value

        If line dosen't follow the format, this raises ItemNotFound
        """
        _match: Optional[Match] = self.item_pat.match(line)
        if _match is None:
            raise ItemNotFound()
        match: Match = _match
        return (match.group('key'), match.group('val'))
