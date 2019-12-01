from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List
import xml.etree.ElementTree as ET

from bowi.elas import models
from bowi.initialize.converters import base
from bowi.models import ColDocument


@dataclass
class NTCIRConverter(base.Converter):

    def _get_title(self,
                   root: ET.Element) -> str:
        title: str = base.find_text_or_default(root, 'TITLE', '')
        return title

    def _get_docid(self,
                   root: ET.Element) -> str:
        def convert_docid(orig: str) -> str:
            """
            >>> converters('PATENT-US-GRT-1993-05176894')
            '199305176894'
            """
            return ''.join(orig.split('-')[-2:])

        docid: str = base.find_text_or_default(root, 'DOCNO', '')
        return convert_docid(docid)

    def _get_tags(self,
                  root: ET.Element) -> List[str]:
        def convert_tags(orig: str) -> List[str]:
            """
            >>> convert_tags('C01C 03/16')
            'C01C'
            """
            return orig.split(' ')[:1]

        clfs: str = base.find_text_or_default(root, 'PRI-IPC', '')
        return convert_tags(clfs)

    def _get_text(self,
                  root: ET.Element) -> str:
        text: str = base.find_text_or_default(root, 'SPEC', '')
        return text

    def escape(self, orig: str) -> str:
        return orig\
            .replace('<tab>', '\t')\
            .replace('"', '&quot;')\
            .replace("&", "&amp;")\
            .replace("\"", "&quot;")

    def to_document(self,
                    fpath: Path) -> Generator[ColDocument, None, None]:
        with open(fpath, 'r') as fin:
            lines: List[str] = [self.escape(line)
                                for line in fin.read().splitlines()]

        for line in lines:
            root: ET.Element = ET.fromstring(line)
            docid: str = self._get_docid(root)
            tags: List[str] = self._get_tags(root)
            title: str = self._get_title(root)
            text: str = self._get_text(root)
            yield ColDocument(docid=models.KeywordField(docid),
                              title=models.TextField(title),
                              text=models.TextField(text),
                              tags=models.KeywordListField(tags))
