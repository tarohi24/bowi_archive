"""
Loader of pre-filtered texts and embeddings
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

from bowi.models import Document
from bowi.settings import cache_dir


def load_emb(docid: str,
             dataset: str,
             model: str) -> Dict[str, np.ndarray]:
    """
    Return
    -----
    A list of matrix (n_sentences * dim)
    """
    dirpath: Path = cache_dir.joinpath(f'{dataset}/{model}/{docid}')
    dic: Dict[str, np.ndarray] = {
        p.stem: np.load(str(p.resolve()))
        for p in dirpath.glob('*.npy')
    }
    return dic


def load_cols(docid: str,
              runname: str,
              dataset: str) -> List[Document]:
    path: Path = cache_dir.joinpath(f'{dataset}/text/{runname}/{docid}.bulk')
    with open(path) as fin:
        lst: List[Document] = [Document.from_json(line)  # type: ignore
                               for line in fin.read().splitlines()]
    return lst
