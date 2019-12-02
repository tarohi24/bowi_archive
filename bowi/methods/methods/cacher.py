"""
extract keywords -> do search ------> embedding -----> dump embedding
                              \\----> dump texts
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Callable, ClassVar, Dict, List, Type, TypedDict, TypeVar)

import numpy as np
from nltk.tokenize import sent_tokenize
from typedflow.batch import Batch
from typedflow.exceptions import FaultItem
from typedflow.flow import Flow
from typedflow.nodes import TaskNode, DumpNode, LoaderNode

from bowi.elas.search import EsSearcher
from bowi.embedding.base import Model as EmbedModel
from bowi.embedding.fasttext import FastText
from bowi.embedding.bert import Bert
from bowi.embedding.elmo import Elmo
from bowi.methods.common.methods import Method
from bowi.methods.methods.keywords import KeywordBaseline, KeywordParam
from bowi.models import ColDocument
from bowi.settings import cache_dir


T = TypeVar('T')
K = TypeVar('K')


@dataclass
class CacheParam:
    n_words: int
    model: str


@dataclass
class Cacher(Method[CacheParam]):
    param_type: ClassVar[Type] = CacheParam
    kb: KeywordBaseline = field(init=False)
    embed_model: EmbedModel = field(init=False)

    def __post_init__(self):
        self.kb: KeywordBaseline = KeywordBaseline(
            mprop=self.mprop,
            param=KeywordParam(n_words=self.param.n_words))
        if self.param.model == 'fasttext':
            self.embed_model: FastText = FastText()
        elif self.param.model == 'elmo':
            self.embed_model: Elmo = Elmo()
        elif self.param.model == 'bert':
            self.embed_model: Elmo = Bert()
        else:
            raise KeyError()

    def get_docid(self,
                  doc: ColDocument) -> str:
        return doc.docid

    def get_filtered_docs(self,
                          doc: ColDocument) -> List[ColDocument]:
        docids: List[str] = [item.docid for item
                             in self.kb.search(doc=doc).hits]
        searcher: EsSearcher = EsSearcher(es_index=self.mprop.context['es_index'])
        docs: List[ColDocument] = searcher\
            .initialize_query()\
            .add_query(terms=docids, field='docid')\
            .add_size(len(docids))\
            .add_source_fields(['text', 'title', 'tags'])\
            .search()\
            .to_docs()
        return docs

    class IDandDocs(TypedDict):
        docid: str
        rel_docs: List[ColDocument]

    def dump_doc(self,
                 batch: Batch[IDandDocs]) -> None:
        for item in batch.data:
            if isinstance(item, FaultItem):
                continue
            if isinstance(item['rel_docs'], FaultItem):
                continue
            path: Path = cache_dir\
                .joinpath(self.mprop.context['es_index'])\
                .joinpath('text')\
                .joinpath(f"{item['docid']}.bulk")
            with open(path, 'w') as fout:
                for doc in item['rel_docs']:
                    fout.write(doc.to_json())
                    fout.write('\n')

    def dump_embedding(self,
                       batch: Batch[IDandDocs]) -> None:
        for item in batch.data:
            if isinstance(item, FaultItem):
                continue
            if isinstance(item['rel_docs'], FaultItem):
                continue
            for doc in item['rel_docs']:
                sents: List[str] = sent_tokenize(doc.text)
                embeddings: np.ndarray = self.embed_model.embed_words(words=sents)
                dirpath: Path = cache_dir\
                    .joinpath(self.mprop.context['es_index'])\
                    .joinpath(self.param.model)\
                    .joinpath(item['docid'])
                try:
                    dirpath.mkdir()
                except FileExistsError:
                    pass
                path = dirpath.joinpath(f"{doc.docid}.npy")
                np.save(str(path.resolve()), embeddings)

    def create_flow(self):
        loader: LoaderNode[ColDocument] = self.load_node
        node_getid: TaskNode[ColDocument, str] = TaskNode(func=self.get_docid)
        (node_getid < loader)('doc')

        node_get_docs: TaskNode[ColDocument, List[ColDocument]] = TaskNode(
            func=self.get_filtered_docs)
        node_dump_text: DumpNode[self.IDandDocs] = DumpNode(func=self.dump_doc)
        (node_dump_text < node_getid)('docid')
        (node_dump_text < node_get_docs)('rel_docs')
        (node_get_docs < loader)('loader')

        node_encoder: DumpNode[self.IDandDocs] = DumpNode(func=self.dump_embedding)
        (node_encoder < node_getid)('docid')
        (node_encoder < node_get_docs)('rel_docs')

        flow: Flow = Flow(dump_nodes=[node_dump_text, node_encoder])
        return flow
