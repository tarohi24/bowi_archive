from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import cast, Dict, List, Set

from bowi.elas.client import EsClient


@dataclass
class Evaluator:
    res_dir: Path
    dataset: str = field(init=False)
    pred_dict: Dict[str, List[str]] = field(init=False)

    def __post_init__(self):
        # parse name of dataset
        self.dataset: str = self.res_dir.parent.stem
        # load pred
        with open(self.res_dir / 'clf.json') as fin:
            self.pred_dict: Dict[str, List[str]] = json.load(fin)

    def _get_accuracy(self, docid: str) -> float:
        try:
            pred: List[str] = self.pred_dict[docid]
        except KeyError:
            raise RuntimeError(f'{docid} is invalid')
        # this line asserts query docs and collection docs are in the same index
        es_client: EsClient = EsClient(self.dataset)
        gt: List[str] = es_client.get_source(docid)['tags']
        unionset: Set[str] = set(pred) & set(gt)
        return 1.0 if len(unionset) > 0 else 0.0
