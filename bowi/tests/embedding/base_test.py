import numpy as np
import pytest

from bowi.embedding.base import return_vector, return_matrix, InvalidShape


@pytest.fixture
def vec() -> np.ndarray:
    vec: np.ndarray = np.arange(6)
    return vec


@pytest.fixture
def mat() -> np.ndarray:
    mat: np.ndarray = np.arange(6).reshape(2, 3)
    return mat


@return_vector
def identity_vec_mapping(vec):
    return vec


@return_matrix
def identity_mat_mapping(mat):
    return mat
