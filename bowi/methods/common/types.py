from __future__ import annotations
from dataclasses import dataclass
from operator import itemgetter
from typing import Dict, Iterable, Tuple, TypedDict, TypeVar  # type: ignore


@dataclass
class TRECResult:
    query_docid: str
    scores: Dict[str, float]

    def to_prel(self) -> str:
        sorted_scores: Iterable[Tuple[str, float]] = sorted(self.scores.items(),
                                                            key=itemgetter(1),
                                                            reverse=True)
        return '\n'.join([f"{self.query_docid} Q0 {key} {rank} {score} STANDARD"
                          for rank, (key, score)
                          in enumerate(sorted_scores, 1)])


class Param:

    @classmethod
    def from_args(cls, args) -> Param:
        ...


@dataclass
class Context:
    es_index: str
    method: str
    runname: str
    n_docs: int
