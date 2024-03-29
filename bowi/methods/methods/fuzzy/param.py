from __future__ import annotations
from dataclasses import dataclass

from bowi.methods.common.types import Param


@dataclass
class FuzzyParam(Param):
    n_words: int

    @classmethod
    def from_args(cls, args) -> FuzzyParam:
        param: FuzzyParam = FuzzyParam(n_words=args.n_words)
        return param
