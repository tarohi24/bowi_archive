"""
Save Embeddings of queries
"""
import argparse
from pathlib import Path
from typing import List

import numpy as np

from bowi import settings
from bowi.embedding.fasttext import FastText
from bowi.models import Document
from bowi.utils.text import get_all_tokens


ft_model: FastText = FastText()


def load_bulk(dataset: str) -> List[Document]:
    dump_path: Path = settings.data_dir.joinpath(f'{dataset}/query/dump.bulk')
    with open(dump_path) as fin:
        docs: List[Document] = [Document.from_json(line)  # type: ignore
                                for line in fin.read().splitlines()]
    return docs


def dump(doc: Document, dataset: str) -> None:
    dump_path: Path = settings.cache_dir.joinpath(
        f'{dataset}/embedding/{doc.docid}.bulk')
    # tokenize
    tokens: List[str] = get_all_tokens(doc.text)
    # embed
    ary: np.ndarray = np.array([
        vec for vec in ft_model.embed_words(tokens)
        if vec is not None])
    np.save(dump_path, ary)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',
                        metavar='D',
                        type=str,
                        nargs=1)
    args = parser.parse_args()
    dataset: str = args.dataset[0]
    for doc in load_bulk(dataset=dataset):
        dump(doc, dataset=dataset)
    return 0


if __name__ == '__main__':
    exit(main())
