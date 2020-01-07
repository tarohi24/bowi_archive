"""
Client module
"""
from dataclasses import dataclass
import logging
from typing import Dict, List, Tuple

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

    def isin(self, docid: str) -> bool:
        body: Dict = {'query': {'match': {'docid': docid}}}
        hits: List = self.es.search(index=self.es_index, body=body)['hits']['hits']
        return len(hits) > 0

    def get_elasid(self, docid: str) -> str:
        body: Dict = {'query': {'match': {'docid': docid}}, '_source': False}
        hit: Dict = self.es.search(index=self.es_index, body=body)['hits']['hits'][0]
        return hit['_id']

    def get_tfidfs(self,
                   docid: str) -> Dict[str, Tuple[int, float]]:
        """
        Get (TF, IDF) of each token in  a doucment
        """
        res: Dict = self.es.termvectors(index=self.es_index,
                                        id=docid,
                                        term_statistics=True,
                                        fields=['text', ])
        tfidfs: Dict[str, Tuple[int, float]] = {
            word: (val['term_freq'], np.log(1 / val['doc_freq']))
            for word, val in res['term_vectors']['text']['terms'].items()
        }
        return tfidfs

    def get_tokens_from_doc(self, docid: str) -> List[str]:
        # get id
        elasid: str = self.get_elasid(docid=docid)
        res: Dict = self.es.termvectors(index=self.es_index,
                                        id=elasid,
                                        fields=['text', ])
        tokens: List[str] = list(
            res['term_vectors']['text']['terms'].keys())
        return tokens

    def get_all_terms(self) -> List[str]:
        """
        CAUTION: too heavy.
        """
        query: Dict = {
            'aggs': {
                'texts': {
                    'terms': {
                        'field': 'text',
                        'size': 999999
                    }
                }
            }
        }
        res = self.es.search(index=self.es_index,
                             body=query,
                             request_timeout=300)
        return res

    def get_source(self, docid: str) -> Dict:
        body: Dict = {'query': {'match': {'docid': docid}}, 'size': 1}
        res: Dict = self.es.search(index=self.es_index,
                                   body=body)
        initial_hit: Dict = res['hits']['hits'][0]  # type: ignore
        return initial_hit['_source']
