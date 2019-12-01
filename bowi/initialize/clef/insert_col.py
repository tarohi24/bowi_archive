"""
load a bulk -> insert
"""
import logging
import json
from pathlib import Path
from typing import Dict, Generator

from elasticsearch.exceptions import NotFoundError
from elasticsearch.helpers import streaming_bulk
from tqdm import tqdm

from bowi.settings import data_dir, es


logger = logging.getLogger(__file__)


def loading(index: str) -> Generator[Dict, None, None]:
    directory: Path = data_dir.joinpath(f'clef/dump')
    for path in tqdm(list(directory.glob(f'*.bulk'))):
        with open(path) as fin:
            for line in fin.read().splitlines():
                dic: Dict = json.loads(line)
                dic['_index'] = index
                yield dic


if __name__ == '__main__':
    index: str = 'clef'

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
                                       loading(index=index),
                                       index=index,
                                       chunk_size=100):
        if not ok:
            logger.warn('Bulk insert: fails')
    es.indices.refresh()
