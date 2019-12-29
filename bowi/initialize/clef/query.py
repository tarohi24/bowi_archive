"""
load xml -> parse and extract -> dump into a file (query.bulk)
"""
from pathlib import Path
from typing import Generator, List
import xml.etree.ElementTree as ET

from tqdm import tqdm

from bowi.intialize.converters.clef import CLEFConverter
from bowi.models import Document
from bowi.settings import data_dir


def get_path() -> Generator[Path, None, None]:
    directory: Path = data_dir.joinpath(f'clef/orig/query')
    for path in tqdm(directory.glob(f'topics/*.xml')):
        yield path


def get_document(converter: CLEFConverter,
                 path: Path) -> Document:
    root: ET.Element = ET.parse(str(path.resolve())).getroot()
    docid: str = converter._get_docid(root)
    tags: List[str] = converter._get_tags(root)
    title: str = converter._get_title(root)
    text: str = converter._get_text(root)
    return Document(docid=docid,
                    title=title,
                    text=text,
                    tags=tags)


def main() -> int:
    converter: CLEFConverter = CLEFConverter()
    dump_path: Path = data_dir.joinpath('clef/query/dump.bulk')
    for path in get_path():
        doc: Document = get_document(converter=converter,
                                     path=path)
        with open(dump_path, 'a') as fout:
            fout.write(doc.to_json() + '\n')  # type: ignore
    return 0


if __name__ == '__main__':
    exit(main())
