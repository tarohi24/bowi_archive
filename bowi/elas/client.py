"""
Client module
"""
from dataclasses import dataclass
import logging
from typing import Dict, Type

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

from bowi.settings import es as ses


logger = logging.getLogger(__file__)


class IndexCreateError(Exception):
    pass


@dataclass
class EsClient:
    es_index: str
    es: Elasticsearch = ses

    def create_index(self,
                     mappings: Dict) -> None:
        ack: Dict = self.es.indices.create(
            index=self.es_index,
            body={'mappings': mappings}
        )
        logger.info(ack)

    def delete_index(self) -> None:
        try:
            self.es.indices.delete(index=self.es_index)
        except NotFoundError:
            logger.info(f'{self.es_index} does not exist')
