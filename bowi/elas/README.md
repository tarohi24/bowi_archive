Chainable Elasticsearch Client
=====

This is a wrapper of `python-elasticsearch`.

## Features

### Chainable methods
This module adopts chainable style to set and execute a query.
 
```python
from bowi.elas.search import EsResult, EsSearcher

q_words: List[str] = ['hey', 'jude']  # query terms
es_index: str = 'myindex'  # index name in Elasticsearch
size: int = 100  # n_docs to be retrieved
search_field: str = 'text'  # the field to search in
source_fields:: List[str] = ['docid']  # metadata to be returned

searcher = EsSearcher(es_index=es_index)
candidates: EsResult = searcher\
    .initialize_query()\
    .add_query(terms=q_words, field=search_field)\
    .add_size(size)\
    .add_filter(terms=query_doc.tags, field='tags')\
    .add_source_fields(source_fields)\
    .search()
```
