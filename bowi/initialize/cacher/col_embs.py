from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, ClassVar, Type, Generator, Optional

import numpy as np
from typedflow.flow import Flow
from typedflow.nodes import DumpNode, TaskNode

from bowi import settings
from bowi.embedding.fasttext import FastText
from bowi.methods.common.types import Param
from bowi.methods.common.methods import Method
from bowi.methods.common.pre_filtering import load_cols
from bowi.models import Document
from bowi.utils.text import get_all_tokens


@dataclass
class ColEmbsParam(Param):
    embed_model: str

    @classmethod
    def from_args(cls, args) -> Param:
        return ColEmbsParam(embed_model=args.embed_model)


@dataclass
class ColEmbs(Method[ColEmbsParam]):
    param_type: ClassVar[Type] = ColEmbsParam
    fasttext: FastText = field(init=False)  # TODO: models should not be fixed

    def __post_init__(self):
        super(ColEmbs, self).__post_init__()
        self.fasttext: FastText = FastText()

    def load_col_texts(self,
                       doc: Document) -> List[Document]:
        cols: List[Document] = load_cols(docid=doc.docid,
                                         runname='100',
                                         dataset=self.context.es_index)
        return cols

    def tokenize(self, cols: List[Document]) -> Dict[str, List[str]]:
        dic: Dict[str, List[str]] = {
            col.docid: get_all_tokens(col.text)
            for col in cols
        }
        return dic

    def embed(self,
              col_dict: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        dic: Dict[str, Optional[List[np.ndarray]]] = {
            docid: self.fasttext.embed_words(tokens)
            for docid, tokens in col_dict.items()
        }
        return {docid: np.array([vec for vec in vecs if vec is not None])
                for docid, vecs in dic.items()}

    def dump(self,
             doc: Document,
             mat_dict: Dict[str, np.ndarray]) -> None:
        dirpath: Path = settings.cache_dir.joinpath(
            f'{self.context.es_index}/col_embs/{doc.docid}')
        dirpath.mkdir(exist_ok=True)
        for col_id, mat in mat_dict.items():
            np.save(dirpath / f'{col_id}.npy', mat)

    def create_flow(self) -> Flow:
        node_load_cols = TaskNode(func=self.load_col_texts)
        (node_load_cols < self.load_node)('doc')
        node_tokenize = TaskNode(func=self.tokenize)
        (node_tokenize < node_load_cols)('cols')
        node_embed = TaskNode(func=self.embed)
        (node_embed < node_tokenize)('col_dict')
        node_dump = DumpNode(self.dump)
        (node_dump < self.load_node)('doc')
        (node_dump < node_embed)('mat_dict')
        
        flow: Flow = Flow([node_dump, ])
        return flow
