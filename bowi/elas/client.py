"""
Client module
"""
from dataclasses import dataclass
import logging
from operator import itemgetter
from typing import Dict, List, Tuple

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

from bowi.utils.text import is_valid_word
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

    def get_tfs(self,
                docid: str) -> Dict[str, int]:
        elasid: str = self.get_elasid(docid=docid)
        res: Dict = self.es.termvectors(index=self.es_index,
                                        id=elasid,
                                        term_statistics=True,
                                        fields=['text', ])
        tfs: Dict[str, int] = {
            word: val['term_freq']
            for word, val in res['term_vectors']['text']['terms'].items()
        }
        return tfs

    def get_tokens_from_doc(self,
                            docid: str,
                            filtering: bool = True) -> List[str]:
        # get id
        elasid: str = self.get_elasid(docid=docid)
        res: Dict[str, Dict] = self.es.termvectors(
            index=self.es_index,
            id=elasid,
            fields=['text', ])['term_vectors']['text']['terms']
        positions: List[Tuple[str, int]] = sum(
            [
                [(word, p['position']) for p in val['tokens']
                 if (not filtering) or is_valid_word(word)]
                for word, val in res.items()
            ],
            []
        )
        tokens: List[str] = [tok for tok, _ in sorted(positions,
                                                      key=itemgetter(1))]
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

    def analyze_text(self, text: str) -> List[str]:
        chunk_size: int = 500
        words: List[str] = text.split()
        n_chunk: int = len(words) // chunk_size + 1
        tokens: List[str] = []
        for i in range(n_chunk):
            body: Dict = {
                'analyzer': 'english',
                'text': ' '.join(
                    words[i * chunk_size:(i + 1) * chunk_size])
            }
            res: List[Dict] = self.es.indices.analyze(index=self.es_index,
                                                      body=body)['tokens']
            tokens.extend([dic['token'] for dic in res])
        return tokens
