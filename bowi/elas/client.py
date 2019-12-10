"""
Client module
"""
from dataclasses import dataclass
import logging
from typing import Dict, List

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
import numpy as np

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

    def get_idfs(self,
                 docid: str) -> np.ndarray:
        res: Dict = self.es.termvectors(index=self.es_index,
                                        id=docid,
                                        fields=['text', ])
        idfs: List[float] = [val['doc_freq']
                             for val in res['term_vectors']['text']['terms'].values()]
        return np.array(idfs)

    def get_tokens_from_doc(self, docid: str) -> List[str]:
        res: Dict = self.es.termvectors(index=self.es_index,
                                        id=docid,
                                        fields=['text', ])
        tokens: List[str] = list(
            res['term_vectors']['text']['terms'].keys())
        return tokens
