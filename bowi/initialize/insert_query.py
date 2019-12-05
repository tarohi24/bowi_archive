"""
Insert query docs to Elasticsearch
"""
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Generator

from elasticsearch.helpers import streaming_bulk

from bowi import settings


logger = logging.getLogger()


def load_query(dump_path: Path,
               es_index: str) -> Generator[Dict, None, None]:
    with open(dump_path) as fin:
        for line in fin:
            dic: Dict = json.loads(line)
            dic['_index'] = es_index
            yield dic


def main():
    parser = argparse.ArgumentParser(description='Insert query docs to Elasticsearch')
    parser.add_argument('dataset',
                        metavar='D',
                        type=str,
                        nargs=1,
                        help='Specify the name of the dataset')

    args = parser.parse_args()
    dataset: str = args.dataset[0]
    dump_path: Path = settings.data_dir.joinpath(f'{dataset}/query/dump.bulk')

    if not dump_path.exists():
        raise RuntimeError(f'Invalid dataset: {dataset}')

    # Create index with mappings
    es_index: str = f'{dataset}_query'
    mapping_path: Path = Path(__file__).parent.joinpath('mappings.json')
    with open(mapping_path) as fin:
        mappings: Dict = json.load(fin)
    ack: Dict = settings.es.indices.create(index=es_index, body=mappings)
    logger.info(ack)

    for ok, response in streaming_bulk(settings.es,
                                       load_query(dump_path=dump_path, es_index=es_index),
                                       index=es_index,
                                       chunk_size=100):
        if not ok:
            logger.warn('Bulk insert: fails')
    settings.es.indices.refresh()


if __name__ == '__main__':
    exit(main())
