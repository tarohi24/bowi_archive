from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List

from annoy import AnnoyIndex
import numpy as np

from bowi.methods.common.types import Context
from bowi import settings


@dataclass
class KeywordCacher:
    """
    Save keywords extracted in the following format:

    docid\tk1,k2,..,kn
    """
    context: Context

    def _get_dump_path(self) -> Path:
        path: Path = settings.cache_dir\
            .joinpath(f'{self.context.es_index}/keywords/{self.context.method}')\
            .joinpath(f'{self.context.runname}.keywords')
        return path

    def dump(self,
             docid: str,
             keywords: List[str]) -> None:
        path: Path = self._get_dump_path()
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        with open(path, 'a') as fout:
            formatted: str = f'{docid}\t{",".join(keywords)}'
            fout.write(formatted + '\n')

    def load(self) -> Dict[str, List[str]]:
        path: Path = self._get_dump_path()
        data: Dict[str, List[str]] = {
            docid: keywords.split(',')
            for docid, keywords
            in [line.split('\t') for line in path.read_text().splitlines()]
        }
        return data


@dataclass
class KNNCacher:
    """
    Cacher of kNN words according to fasttext
    """
    dataset: str
    dim: int = 300
    ann: AnnoyIndex = field(init=False)
    w2i: Dict[str, int] = field(init=False)
    i2w: Dict[int, str] = field(init=False)

    def __post_init__(self):
        # load Annoy index
        cdir: Path = settings.cache_dir / f'{self.dataset}/knn'
        self.ann = AnnoyIndex(self.dim, 'angular')
        self.ann.load(str((cdir / 'knn.ann').resolve()))

        # load id <--> word converters
        with open(cdir / 'w2i.json') as fin:
            self.w2i = json.load(fin)
        self.i2w = {id_: word for word, id_ in self.w2i.items()}

    def get_nn(self,
               word: str,
               threshold: float = 0.5,
               include_self: bool = False) -> Dict[str, float]:
        """
        Get words whose distance to the given word is below threshold.
        Raise KeyError when word is not in the vocab

        Return
        -----
        {word: dist}
        """
        id_: int = self.w2i[word]
        dist_threshold: float = np.sqrt(2 * (1 - threshold))
        ids, dists = self.ann.get_nns_by_item(id_,
                                              20,  # whatever you like
                                              include_distances=True)
        data: Dict[str, float] = {
            self.i2w[i]: dist for i, dist in zip(ids, dists)
            if dist < dist_threshold
        }
        if include_self:
            data[word] = 0.0
        return data


@dataclass
class DFCacher:
    dataset: str
    df_dict: Dict[str, int] = field(init=False)
    total_docs: int = field(init=False)

    def __post_init__(self):
        path: Path = settings.cache_dir / f'{self.dataset}/df.json'
        with open(path) as fin:
            self.df_dict = json.load(fin)
        self.total_docs = {
            'clef': 993910,
            'ntcir': 1315355,
            'aan': 23596,
            'cmu': 19865,
        }[self.dataset]

    def get_idf(self, word: str) -> float:
        try:
            df: int = self.df_dict[word]
        except KeyError:
            df: int = 2  # type: ignore
        # Elasticsearch-style IDF
        return np.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
