import logging
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from bowi.embedding.base import return_matrix


logger = logging.getLogger(__file__)


@return_matrix
def _get_new_kemb_cand(cand_emb: np.ndarray,
                       keyword_embs: Optional[np.ndarray]) -> np.ndarray:
    if keyword_embs is None:
        return cand_emb.reshape(1, -1)
    else:
        mod_cand_emb: np.ndarray = cand_emb.reshape(1, -1)
        assert keyword_embs.ndim == 2
        assert mod_cand_emb.ndim == 2
        return np.concatenate([keyword_embs, mod_cand_emb])


def rec_loss(embs: np.ndarray,
             keyword_embs: Optional[np.ndarray],
             cand_emb: np.ndarray) -> float:
    """
    Reconstruct error. In order to enable unittests, two errors are
    implemented individually.
    """
    dims: np.ndarray = _get_new_kemb_cand(cand_emb=cand_emb,
                                          keyword_embs=keyword_embs)
    assert dims.ndim == 2
    assert embs.shape[1] == dims.shape[1]
    maxes: np.ndarray = np.amax(np.dot(embs, dims.T), axis=1)
    val: float = (1 - maxes).mean()
    return val


def cent_sim_loss(keyword_embs: np.ndarray,
                  cand_emb: np.ndarray) -> float:
    """
    Loss penalizing for similar dimensions
    """
    val: float = np.dot(keyword_embs, cand_emb).mean()
    return val


def calc_error(embs: np.ndarray,
               keyword_embs: Optional[np.ndarray],
               cand_emb: np.ndarray,
               coef: float) -> float:
    rec_error: float = rec_loss(embs, keyword_embs, cand_emb)
    if keyword_embs is not None and coef != 0:
        assert keyword_embs.ndim == 2
        cent_sim_error: float = cent_sim_loss(keyword_embs, cand_emb)
        logger.debug(
            f'rec: {str(rec_error)}, cent: {str(cent_sim_error)}, kemb: {str(keyword_embs.shape)} cand: {str(cand_emb.shape)}')
    else:
        cent_sim_error: float = 0  # type: ignore
    return rec_error + coef * cent_sim_error


@return_matrix
def get_keyword_embs(embs: np.ndarray,
                     keyword_embs: Optional[np.ndarray],
                     n_remains: int,
                     coef: float,
                     pbar=None) -> np.ndarray:
    if pbar is None:
        pbar = tqdm(total=n_remains)  # noqa
    uniq_vecs: np.ndarray = np.unique(embs, axis=0)
    errors: List[float] = [calc_error(embs=embs,
                                      keyword_embs=keyword_embs,
                                      cand_emb=cand,
                                      coef=coef)
                           for cand in uniq_vecs]
    argmin: int = np.argmin(errors)
    new_keyword_emb = uniq_vecs[argmin]
    residual_inds: np.ndarray = np.array([not np.array_equal(vec, new_keyword_emb) for vec in embs])
    new_dims: np.ndarray = _get_new_kemb_cand(cand_emb=new_keyword_emb,
                                              keyword_embs=keyword_embs)
    pbar.update(1)
    if n_remains == 1:
        return new_dims
    else:
        res_dims: np.ndarray = embs[residual_inds]
        return get_keyword_embs(embs=res_dims,
                                keyword_embs=new_dims,
                                n_remains=(n_remains - 1),
                                coef=coef,
                                pbar=pbar)
