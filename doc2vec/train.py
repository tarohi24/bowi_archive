from typing import Generator, List

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm

from bowi.elas.client import EsClient
from bowi.settings import project_root


index: str = 'clef'
TAG: str = 'A'
n_docs: int = 10000


def load_corpus() -> Generator[TaggedDocument, None, None]:
    escl: EsClient = EsClient(es_index=index)
    for _ in tqdm(list(range(n_docs))):
        docid: str = escl.get_random_id()
        tokens: List[str] = escl.get_tokens_from_doc(docid)
        yield TaggedDocument(words=tokens, tags=[TAG, ])


def main() -> int:
    model: Doc2Vec = Doc2Vec(list(load_corpus()),
                             vector_size=5,
                             window=2,
                             min_count=3,
                             workers=4)
    model.save(str((project_root / f'doc2vec/{index}.model').resolve()))
    return 0


if __name__ == '__main__':
    exit(main())
