"""
load -> create bulk query -> insert to ES
"""
import logging
import json
from pathlib import Path
from typing import Dict, Generator

from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm

from bowi import settings
from bowi.models import Document
from bowi.initialize.converters.cmu import CmuConveter


logger = logging.getLogger(__file__)


def load_corpus() -> Generator[Dict, None, None]:
    """
    Assert that 20_newsgroups directory exists under the home dir
    """
    directory: Path = settings.data_dir.joinpath('cmu/orig/20_newsgroups')
    converter: CmuConveter = CmuConveter()
    for fpath in tqdm(list(directory.glob('**/*')),
                      desc='Processing corpus...'):
        print(fpath)
        try:
            int(fpath.name)
        except ValueError:
            continue
        try:
            doc: Document = converter.to_document(fpath)
        except RuntimeError as e:
            logger.warn(str(e))
            continue
        yield doc.to_json()


def insert_doc(doc: Document) -> None:
    pass


def main() -> int:
    index: str = 'cmu'

    es = settings.es
    # delete the old index
    try:
        es.indices.delete(index=index)
    except NotFoundError:
        pass

    # create an index
    mapping_path: Path = Path(__file__).parent.parent.joinpath('mappings.json')
    with open(mapping_path) as fin:
        mappings: Dict = json.load(fin)
    ack: Dict = es.indices.create(
        index=index,
        body=mappings)
    logger.info(ack)

    for ok, response in streaming_bulk(es,
                                       load_corpus(),
                                       index=index,
                                       chunk_size=100):
        if not ok:
            logger.warn('Bulk insert: fails')
    es.indices.refresh()


if __name__ == '__main__':
    exit(main())
