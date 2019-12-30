"""
KNN Document Classifier
"""
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
from typing import cast, Counter, Dict, List, Pattern, Match

from tqdm import tqdm

from bowi.elas.client import EsClient


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
            if (match := self.prel_pat.match(line)) is None:  # noqa
                continue
            qid: str = match.group('qid')
            relid: str = match.group('relid')
            if qid == relid:
                continue
            if len(knns[qid]) < self.n_neigh:
                knns[qid].append(relid)
        return knns

    def _get_labels(self,
                    ids: List[str],
                    es_client: EsClient) -> List[List[str]]:
        if len(ids) == 0:
            return []
        else:
            head, *tail = ids
            source: Dict = es_client.get_source(docid=head)
            tags: List[str] = source['tags']
            return [tags] + self._get_labels(tail, es_client=es_client)

    def _get_repr(self,
                  tags: List[List[str]]) -> Dict[str, str]:
        counter: Counter[str] = Counter(sum(tags, []))
        return counter.most_common(1)[0][0]

    def clf(self,
            prel_file: Path,
            es_client: EsClient) -> Dict[str, str]:
        """
        For each document, choose the most popular class to which
        kNN documents belong to.
        """
        clf_res: Dict[str, str] = dict()
        rel_docs: Dict[str, List[str]] = self._load_prel_file(prel_file=prel_file)
        for qid, relids in tqdm(rel_docs.items()):
            tags_list: List[List[str]] = self._get_labels(relids,
                                                          es_client=es_client)
            clf_res[qid] = self._get_repr(tags_list)
        return clf_res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',
                        type=str)
    parser.add_argument('prel_path',
                        type=Path)
    args = parser.parse_args()
    es_client: EsClient = EsClient(args.dataset)
    knn_clf = KNNClassifier()
    prel_path: Path = args.prel_path
    res: Dict[str, str] = knn_clf.clf(prel_file=prel_path,
                                      es_client=es_client)
    # save result
    dump_dir: Path = prel_path.parent
    dump_path: Path = dump_dir / 'clf.json'
    with open(dump_path, 'w') as fout:
        json.dump(res, fout)
    return 0


if __name__ == '__main__':
    exit(main())
