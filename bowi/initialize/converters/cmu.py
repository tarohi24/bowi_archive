from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import List, Optional, Pattern, Match, Tuple

from bowi.initialize.converters.base import Converter
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
        self.item_pat = re.compile(r'^(?P<key>[\-a-zA-Z0-9]+): (?P<val>.+)$')

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

    def _get_docid(self,
                   fpath: Path) -> str:
        docid: str = fpath.stem
        return docid

    def _get_tags(self,
                  fpath: Path) -> List[str]:
        """
        Get categories (actually one category) from the path
        """
        cat: str = fpath.parent.name
        return [cat, ]

    def to_document(self,
                    fpath: Path) -> Document:
        docid: str = self._get_docid(fpath)
        tags: List[str] = self._get_tags(fpath)
        with open(fpath) as fin:
            text: List[str] = fin.read().splitlines()
        for i, line in enumerate(text):
            if line == '':
                continue
            try:
                key, val = self._get_itemize(line)
            except ItemNotFound:
                body: str = '\n'.join([s for s in text[i:] if s != ''])
                break
            if key == 'Subject':
                title: str = val

        try:
            title
        except NameError:
            raise RuntimeError(f'Title is not found in {docid}')
        try:
            body
        except NameError:
            raise RuntimeError(f'Body is not found in {docid}')

        return Document(docid=docid, tags=tags, text=body, title=title)
