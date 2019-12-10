from typing import Any, Dict, List, Set, Generator

import pytest
import numpy as np
from typedflow.batch import Batch
from typedflow.nodes import LoaderNode
from typedflow.exceptions import EndOfBatch

from bowi.models import Document
from bowi.embedding import fasttext
from bowi.methods.methods.fuzzy.param import FuzzyParam
from bowi.methods.methods.fuzzy import rerank
from bowi.tests.methods.methods.base import context  # noqa

from bowi.tests.embedding.fasttext import FTMock


@pytest.fixture
def param() -> FuzzyParam:
    return FuzzyParam(
        n_words=3,
        model='fasttext',
        coef=1,
        prefilter_name='100'
    )


@pytest.fixture
def model(param, context, mocker) -> rerank.FuzzyRerank:  # noqa
    mocker.patch.object(rerank, 'FastText', FTMock)
    return rerank.FuzzyRerank(param=param, context=context)


def get_tokens() -> List[str]:
    tokens: List[str] = 'hello world everyone'.split()
    return tokens
    

def test_embed_words(model):
    mat = model.embed_words(get_tokens())
    assert mat.ndim == 2
    norms = np.linalg.norm(mat, axis=1)
    # assert mat is normalized
    np.testing.assert_array_almost_equal(norms, np.ones(mat.shape[0]))


def test_get_kembs(model):
    mat = model.embed_words(get_tokens())
    embs = model.get_kembs(mat)
    assert set(model._get_nns(mat, embs)) == {0, 1, 2}


def test_fuzzy_bows(mocker, model):
    mat = model.embed_words(get_tokens())
    embs = model.get_kembs(mat)
    bow: np.ndarray = model.to_fuzzy_bows(mat, embs)
    ones: np.ndarray = np.ones(embs.shape[0])
    np.testing.assert_array_almost_equal(bow, ones / np.sum(ones))

    # 2 keywords
    mocker.patch.object(model.param, 'n_words', 2)
    embs = model.get_kembs(mat)
    assert embs.shape[0] == 2
    sorted_sims: np.ndarray = np.sort(model.to_fuzzy_bows(mat, embs))
    desired = np.sort([2 / 3, 1 / 3])
    np.testing.assert_array_almost_equal(sorted_sims, desired)


def test_match(mocker, model):
    col_bows: Dict[str, np.ndarray] = {
        'a': np.ones(3),
        'b': np.array([0.5, 0.3, 0.2]), 
    }
    col_bows = {docid: vec / np.linalg.norm(vec)  # noqa
                for docid, vec in col_bows.items()}
    qbow = col_bows['a']

    qdoc = mocker.MagicMock()
    qdoc.docid = 'query'
    res = model.match(query_doc=qdoc,
                      query_bow=qbow,
                      col_bows=col_bows)
    assert res.scores['a'] > res.scores['b']


def test_get_cols(mocker, model):
    mocker.patch.object(model.context, 'es_index', 'clef')
    qdoc = mocker.MagicMock()
    qdoc.docid = 'EP1288722A2'
    ids: Set[str] = set(d.docid for d in model.get_cols(query=qdoc))
    assert set(ids) == {'EP0762208B1', 'EP0762208A2', 'EP1096314A1'}


def test_typecheck(model):
    flow = model.create_flow()
    flow.typecheck()


def test_flow(mocker, model):

    def generate_query() -> Generator[Document, None, None]:
        yield Document(docid='query', title='', text='hey jude', tags=[])
    
    def generate_cols(docid: str, dataset: str) -> List[Document]:
        return [Document(docid=str(i), title='', text=str(i), tags=[])
                for i in range(2)]

    def accept(self,
               batch_id: int) -> Batch[Dict[str, Any]]:
        """
        merge all the arguments items into an instance of T (=arg_type)
        """
        materials: Dict[str, Batch] = dict()
        for key, prec in self.precs.items():
            try:
                materials[key] = prec.get_or_produce_batch(batch_id=batch_id)
            except EndOfBatch:
                print(key)
        # check lengths of batches
        merged_batch: Batch[Dict[str, Any]] = self._merge_batches(materials=materials)
        return merged_batch

    mocker.patch('typedflow.nodes.ConsumerNode.accept', accept)
    mocker.patch.object(model, 'load_node', LoaderNode(generate_query, batch_size=1))
    mocker.patch('bowi.methods.methods.fuzzy.rerank.load_cols', generate_cols)
    flow = model.create_flow()
    assert flow.get_loader_nodes()[0].cache_table.life == 3
    flow.dump_nodes[0].accept(batch_id=0)
