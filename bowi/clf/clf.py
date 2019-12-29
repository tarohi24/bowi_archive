"""
Modules for CLF
"""
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import cast, Dict, List, Pattern, Match


@dataclass
class KNNClassifier():
    """
    KNN classifier
    """
    n_neigh: int = 3
    prel_pat: Pattern = field(
        default_factory=lambda: re.compile(
            r'^(?P<qid>\w+)\s+Q0\s+(?P<relid>\w+)\s+(?P<rank>\d+)\s+(?P<score>[\d\.]+)\sSTANDARD$'))

    def _load_prel_file(self,
                        prel_file: Path) -> Dict[str, List[str]]:
        """
        Load prel file (trec_eval format) and extract
        relevant items
        """
        def create_str_list() -> List[str]:
            """
            This is for passing mypy tests.
            Because lambda expression doesn't allow type annotatinos,
            I prepared for this function.
            """
            return []

        knns: Dict[str, List[str]] = defaultdict(create_str_list)
        lines: List[str] = prel_file.read_text().splitlines()
        for line in lines:
            if (match := self.prel_pat.match(line)) is None:
                continue
            knns[match.group('qid')].append(match.group('relid'))
        return knns
