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
        cdir: Path = settings.cache_dir / self.dataset
        self.ann = AnnoyIndex(self.dim, 'angular')
        self.ann.load(cdir / 'knn.ann')

        # load id <--> word converters
        with open(cdir / 'w2i.json') as fin:
            self.w2i = json.load(fin)
        self.i2w = {id_: word for word, id_ in self.w2i.items()}

    def get_nn(self,
               word: str,
               threhsold: float = 0.5) -> List[str]:
        try:
            id_: int = self.w2i[word]
        except KeyError:
            raise RuntimeError(f'{word} is not in the vocab')
        dist_threshold: float = np.sqrt(2 * (1 - threhsold))
        ids, dists = self.ann.get_nns_by_item(id_,
                                              20,  # whatever you like
                                              include_distances=True)
        return [self.i2w[i] for i, dist in zip(ids, dists)
                if dist < dist_threshold]
