import argparse
from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Dict, List

from bowi.elas.client import EsClient


@dataclass
class Evaluator:
    res_dir: Path
    dataset: str = field(init=False)
    pred_dict: Dict[str, str] = field(init=False)

    def __post_init__(self):
        # parse name of dataset
        self.dataset: str = self.res_dir.parent.parent.stem
        # load pred
        with open(self.res_dir / 'clf.json') as fin:
            self.pred_dict: Dict[str, str] = json.load(fin)

    def _get_accuracy(self, docid: str) -> float:
        try:
            pred: str = self.pred_dict[docid]
        except KeyError:
            raise RuntimeError(f'{docid} is invalid')
        # this line asserts query docs and collection docs are in the same index
        es_client: EsClient = EsClient(self.dataset)
        gt: List[str] = es_client.get_source(docid)['tags']
        return 1.0 if pred in gt else 0.0

    def get_accs(self) -> Dict[str, float]:
        """
        Compute accuracices of queries in pred_dict
        """
        accs: Dict[str, float] = {docid: self._get_accuracy(docid=docid)
                                  for docid in self.pred_dict.keys()}
        return accs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('resdir', type=Path)
    args = parser.parse_args()
    res_dir: Path = args.resdir
    evaluator: Evaluator = Evaluator(res_dir=res_dir)
    res: Dict[str, float] = evaluator.get_accs()
    # save results
    dump_path: Path = res_dir / 'clfres.json'
    with open(dump_path, 'w') as fout:
        json.dump(res, fout)
    return 0


if __name__ == '__main__':
    exit(main())
