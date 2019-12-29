from __future__ import annotations
from dataclasses import dataclass

from bowi.methods.common.types import Param


@dataclass
class FuzzyParam(Param):
    n_words: int
    model: str
    prefilter_name: str
    min_tf: int

    @classmethod
    def from_args(cls, args) -> FuzzyParam:
        param: FuzzyParam = FuzzyParam(n_words=args.n_words,
                                       model=args.model,
                                       prefilter_name=str(args.prefilter_name),
                                       min_tf=args.min_tf)
        return param
