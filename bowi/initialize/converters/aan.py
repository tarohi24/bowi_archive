from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import xml.etree.ElementTree as ET

from bowi.initialize.converters.base import Converter
from bowi.models import Document
from bowi.settings import data_dir

logger = logging.getLogger(__file__)
with open(data_dir.joinpath('aan/orig/titles.txt'), 'r') as fin:
    title_dic: Dict[str, str] = {
        line[0]: line[1]
        for line
        in [l.split('\t') for l in fin.read().splitlines()]
    }


@dataclass
class AANConverter(Converter):

    def _get_paragraph_list(self,
                            root: ET.Element) -> List[str]:
        pass

    def _get_info(self,
                  fpath: Path) -> Tuple[str, List[str], str, str]:
        docid: str = fpath.stem
        tags: List[str] = [docid.split('-')[0], ]  # the first alphabet
        try:
            title: str = title_dic[docid]
        except KeyError:
            logger.warn(f'The title of {docid} is not found')
            title: str = ''  # type: ignore
        with open(fpath, 'r') as fin:
            text: str = fin.read()
        return docid, tags, title, text

    def to_document(self,
                    fpath: Path) -> List[Document]:
        docid, tags, title, text = self._get_info(fpath)
        return [Document(docid=docid, title=title, text=text, tags=tags)]


if __name__ == '__main__':
    converter = AANConverter()
    fpath: Path = Path(sys.argv[1])
    print(converter.to_document(fpath))
