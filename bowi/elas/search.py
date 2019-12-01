"""
Module for elasticsaerch
"""
from __future__ import annotations  # noqa
from dataclasses import dataclass, field
import logging
from typing import Dict, Generator, Iterable, List, Tuple

from elasticsearch.helpers import scan

from bowi.elas.client import EsClient
from bowi.models import ColDocument
from bowi.settings import es
from bowi.models import RankItem


logger = logging.getLogger(__file__)


@dataclass
class EsResultItem:
    docid: str
    score: float
    source: Dict

    @classmethod
    def from_dict(cls,
                  data: Dict) -> EsResultItem:
        self = EsResultItem(
            docid=data['_source']['docid'],
            score=data['_score'],
            source=data['_source']
        )
        return self

    def get_id_and_tag(self) -> Tuple[str, str]:
        assert 'tags' in self.source
        return (self.docid, str(self.source['tags'][0]))

    def to_document(self) -> ColDocument:
        assert 'text' in self.source, 'text not found in source'
        assert 'tags' in self.source, 'tags not found in source'
        assert 'title' in self.source, 'title not found in source'
        return ColDocument(
            docid=self.docid,
            text=self.source['text'],
            title=self.source['title'],
            tags=self.source['tags'])


@dataclass
class EsResult:
    hits: Iterable[EsResultItem]

    @classmethod
    def from_dict(cls,
                  data: Dict) -> EsResult:
        hits: List[Dict] = data['hits']['hits']  # type: ignore
        return cls([EsResultItem.from_dict(hit) for hit in hits])

    def to_rank_item(self,
                     query_id: str) -> RankItem:
        scores: Dict[Tuple[str, str], float] = {
            hit.get_id_and_tag(): hit.score
            for hit in self.hits}
        return RankItem(query_id=query_id, scores=scores)

    def get_scores(self) -> Dict[str, float]:
        dic: Dict[str, float] = {
            hit.docid: hit.score
            for hit in self.hits
        }
        return dic

    def to_docs(self) -> List[ColDocument]:
        return [hit.to_document() for hit in self.hits]


@dataclass
class EsSearcher:
    """
    Module for search for Elasticsearch
    """
    es_index: str
    query: Dict = field(default_factory=dict)
    es: EsClient = es

    def search(self) -> EsResult:
        res: EsResult = EsResult.from_dict(
            es.search(index=self.es_index, body=self.query))
        return res

    def scroll(self) -> EsResult:

        def scanning(client: EsClient,
                     query: Dict,
                     es_index: str,
                     scroll: str = '1s') -> Generator[EsResultItem, None, None]:
            res: Generator[Dict, None, None] = scan(
                client=client,
                query=query,
                index=es_index,
                scroll=scroll)
            for dic in res:
                yield EsResultItem.from_dict(dic)

        sc: Generator[EsResultItem, None, None] = scanning(
            client=self.es,
            query=self.query,
            es_index=self.es_index)
        res: EsResult = EsResult(sc)
        return res

    def initialize_query(self) -> EsSearcher:
        self.query['_source'] = ['docid', ]
        return self

    def add_match_all(self) -> EsSearcher:
        self.query['query'] = {'match_all': {}}
        return self

    def add_query(self,
                  terms: List[str],
                  field: str = 'text',
                  condition: str = 'should') -> EsSearcher:
        """
        modify self.query
        """
        self.query['query'] = {
            'bool': {
                condition: [
                    {'match': {field: t}}
                    for t in terms
                ]
            }
        }
        return self

    def add_size(self,
                 size: int) -> EsSearcher:
        """
        modify self.size
        """
        self.query['size'] = size
        return self

    def add_source_fields(self,
                          source_fields: List[str]) -> EsSearcher:
        """
        modify self._source
        """
        self.query['_source'].extend(source_fields)
        return self

    def add_filter(self,
                   terms: List[str],
                   field: str = 'tags',
                   condition: str = 'should') -> EsSearcher:
        """
        modify self.query
        """
        self.query['query']['bool']['filter'] = {
            'bool': {
                'should': [{'match': {field: t}} for t in terms]
            }
        }
        return self
