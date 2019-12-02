"""
Loader of pre-filtered texts and embeddings
"""
from pathlib import Path
from typing import Dict, List

import numpy as np

from bowi.models import ColDocument
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
              dataset: str) -> List[ColDocument]:
    path: Path = cache_dir.joinpath(f'{dataset}/text/{docid}.bulk')
    with open(path) as fin:
        lst: List[ColDocument] = [ColDocument.from_json(line)  # type: ignore
                                  for line in fin.read().splitlines()]
    return lst
