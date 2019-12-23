from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
        with open(path, 'a') as fout:
            formatted: str = f'{docid}\t{",".join(keywords)}'
            fout.write(formatted + '\n')

    def load(self) -> Dict[str, List[str]]:
        path: Path = self._get_dump_path()
        data: Dict[str, List[str]] = {
            docid: keywords.split('\n')
            for docid, keywords
            in [line.split('\t') for line in path.read_text.splitlines()]
        }
        return data
