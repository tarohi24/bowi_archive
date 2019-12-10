from pathlib import Path
from typing import Dict, List

import pytest
import numpy as np

from bowi import settings
from bowi.tests.embedding.fasttext import FTMock
from bowi.embedding.base import mat_normalize
from bowi.methods.methods.fuzzy.fuzzy import get_keyword_embs, rec_loss

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
    idfs: np.ndarray = np.random.rand(len(tokens))
    embs = mat_normalize(np.array([sample_embeddings[w] for w in tokens]))
    assert 0 < rec_loss(embs=embs,
                        keyword_embs=None,
                        cand_emb=embs[1],
                        idfs=idfs) < 4
    assert 0 < rec_loss(embs=embs,
                        keyword_embs=embs[:1],
                        cand_emb=embs[2],
                        idfs=idfs) < 3


def test_get_keywords(sample_embeddings):
    tokens = ['software', 'license', 'program', 'terms', 'code']
    idfs: np.ndarray = np.random.rand(len(tokens))
    embs = mat_normalize(np.array([sample_embeddings[w] for w in tokens]))
    keyword_embs: np.ndarray = get_keyword_embs(
        embs=embs,
        keyword_embs=None,
        idfs=idfs,
        n_remains=2,
        coef=1)
    gt_embs: np.ndarray = embs[[0, 1]]
    assert np.linalg.matrix_rank(keyword_embs) == 2
    assert np.linalg.matrix_rank(keyword_embs - gt_embs) == 1
