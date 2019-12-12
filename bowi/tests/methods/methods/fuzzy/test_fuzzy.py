from pathlib import Path
from typing import Dict, List

import pytest
import numpy as np

from bowi import settings
from bowi.tests.embedding.fasttext import FTMock
from bowi.embedding.base import mat_normalize
from bowi.methods.methods.fuzzy.fuzzy import get_keyword_inds, rec_loss, offsetted_ind

from bowi.testing.patches import patch_data_dir


@pytest.fixture(autouse=True)
def mock_ft(mocker):
    mocker.patch('bowi.methods.methods.fuzzy.naive.FastText', new=FTMock)


@pytest.fixture
def sample_embeddings() -> Dict[str, np.ndarray]:
    embdir: Path = settings.data_dir.joinpath('embs')
    with open(embdir.joinpath('fasttext_vocab.txt')) as fin:
        vocabs: List[str] = fin.read().splitlines()
    mat: np.ndarray = np.load(str(embdir.joinpath('fasttext_embs.npy').resolve()))
    return {word: ary for word, ary in zip(vocabs, mat)}


def test_rec_loss(sample_embeddings):
    tokens = ['software', 'license', 'program', 'terms', 'code']
    tfs: np.ndarray = np.random.randint(5, size=len(tokens))
    idfs: np.ndarray = np.random.rand(len(tokens))
    embs = mat_normalize(np.array([sample_embeddings[w] for w in tokens]))
    assert 0 < rec_loss(embs=embs,
                        keyword_embs=None,
                        cand_emb=embs[1],
                        tfs=tfs,
                        idfs=idfs) < 4
    assert 0 < rec_loss(embs=embs,
                        keyword_embs=embs[:1],
                        cand_emb=embs[2],
                        tfs=tfs,
                        idfs=idfs) < 3


def test_get_keywords(sample_embeddings):
    tokens = ['software', 'license', 'program', 'terms', 'code']
    tfs: np.ndarray = np.random.randint(5, size=len(tokens))
    idfs: np.ndarray = np.random.rand(len(tokens))
    _embs: np.ndarray = np.array([sample_embeddings[w] for w in tokens])
    assert _embs.shape == (5, 300)
    embs = mat_normalize(_embs)
    key_inds: List[int] = get_keyword_inds(embs=embs,
                                           idfs=idfs,
                                           tfs=tfs,
                                           n_keywords=2)
    # Assert if keyword embs are chosen from the original matrix
    assert len(key_inds) == 2


def test_offset():
    offset = offsetted_ind
    assert offset(1, [0]) == 2
    assert offset(1, [1]) == 2
    assert offset(1, [2]) == 1
    assert offset(100, [1, 3, 5, 101]) == 103
