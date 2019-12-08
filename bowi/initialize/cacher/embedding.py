"""
Save Embeddings of queries with the word
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, ClassVar, Type, Generator

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import DumpNode, TaskNode

from bowi import settings
from bowi.embedding.fasttext import FastText
from bowi.methods.common.types import Param
from bowi.methods.common.methods import Method
from bowi.models import Document
from bowi.utils.text import get_all_tokens


@dataclass
class EmbeddingCacheParam(Param):
    embed_model: str

    @classmethod
    def from_args(cls, args) -> Param:
        return EmbeddingCacheParam(embed_model=args.embed_model)


@dataclass
class EmbededWord:
    word: str
    vec: np.ndarray


@dataclass
class EmbeddingCacher(Method[EmbeddingCacheParam]):
    param_type: ClassVar[Type] = EmbeddingCacheParam
    fasttext: FastText = field(init=False)  # TODO: models should not be fixed

    def __post_init__(self):
        super(EmbeddingCacher, self).__post_init__()
        self.fasttext: FastText = FastText()

    def embed(self,
              doc: Document) -> Generator[EmbededWord, None, None]:
        tokens: List[str] = get_all_tokens(doc.text)
        emb_list: List[Optional[np.ndarray]] = self.fasttext.embed_words(tokens)
        for tok, vec in zip(tokens, emb_list):
            if vec is not None:
                yield EmbededWord(word=tok, vec=vec)
        return

    def dump(self,
             ews: Generator[EmbededWord, None, None],
             doc: Document) -> None:
        """
        Dump both keywords and their embeddings

        - EPXXXXX (dir)
        |- words.txt
        |- embeddings.npy
        """
        dump_dir: Path = settings.cache_dir.joinpath(
            f'{self.context.es_index}/embedding/{doc.docid}')
        dump_dir.mkdir(parents=True, exist_ok=True)
        emb_list: List[np.ndarray] = []
        with open(dump_dir.joinpath('words.txt'), 'w') as fout:
            for ew in ews:
                fout.write(f'{ew.word}\n')
                emb_list.append(ew.vec)
        embs: np.ndarray = np.array(emb_list)
        np.save(dump_dir.joinpath('embeddings.npy'), embs)

    def create_flow(self) -> Flow:
        node_embed: TaskNode = TaskNode(func=self.embed)
        (node_embed < self.load_node)('doc')
        node_dump: DumpNode = DumpNode(func=self.dump)
        (node_dump < node_embed)('ews')
        (node_dump < self.load_node)('doc')
        flow: Flow = Flow(dump_nodes=[node_dump, ])
        return flow
