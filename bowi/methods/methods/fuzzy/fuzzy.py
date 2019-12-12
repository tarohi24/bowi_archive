import logging
import warnings
from typing import List, Optional

import numpy as np
from tqdm import tqdm

from bowi.embedding.base import return_matrix


logger = logging.getLogger(__file__)


def offsetted_ind(ind: int, ranges: List[int]) -> int:
    """
    Calculate the original index of given ind

    >>> offset(1, [0])
    2
    >>> offset(1, [1])
    2
    >>> offset(1, [2])
    1
    >>> offset(100, [1, 3, 5, 101])
    103
    """
    if len(ranges) == 0:
        return ind
    else:
        head, *tail = ranges
        if ind < head:
            return ind
        else:
            return offsetted_ind(ind, tail) + 1


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
             cand_emb: np.ndarray,
             tfs: np.ndarray,
             idfs: np.ndarray) -> float:
    """
    Reconstruct error. In order to enable unittests, two errors are
    implemented individually.
    """
    dims: np.ndarray = _get_new_kemb_cand(cand_emb=cand_emb,
                                          keyword_embs=keyword_embs)
    assert dims.ndim == 2
    assert tfs.ndim == 1
    assert idfs.ndim == 1
    assert embs.shape[1] == dims.shape[1]
    assert embs.shape[0] == len(tfs) == len(idfs)

    with warnings.catch_warnings():
        try:
            tfidfs: np.ndarray = tfs * idfs
            maxes: np.ndarray = np.amax(np.dot(embs, dims.T), axis=1)
        except Warning as w:
            raise RuntimeError(str(w))
    val: float = ((1 - maxes) * tfidfs).mean()
    return val


def calc_error(embs: np.ndarray,
               keyword_embs: Optional[np.ndarray],
               cand_emb: np.ndarray,
               tfs: np.ndarray,
               idfs: np.ndarray) -> float:
    rec_error: float = rec_loss(embs=embs,
                                keyword_embs=keyword_embs,
                                cand_emb=cand_emb,
                                tfs=tfs,
                                idfs=idfs)
    logger.debug(f'recerror: {str(rec_error)}')
    return rec_error


def get_keyword_inds(embs: np.ndarray,
                     tfs: np.ndarray,
                     idfs: np.ndarray,
                     n_keywords: int,
                     keyword_embs: Optional[np.ndarray] = None,
                     prev_key_inds: List[int] = [],
                     pbar=None) -> List[int]:
    """
    To prevent from being confused, you have to specify args with keywords.

    Returns
    -----
    indices of keywords in the original embedding

    Parameters
    -----
    embs
        word embeddings (each two vectors should be different)

    tfs
        Term Frequencies of each word

    idfs
        Inverted Document Frequencies of each word

    keyword_embs
        You don't have to set this value (only inside this function this arg is used).

    prev_key_inds
        You don't need to set

    pbar
        Progress bar. If none, this method automatically creates a new one.
    """
    assert len(embs) == len(tfs) == len(idfs)
    # Create a progress bar.
    if pbar is None:
        pbar = tqdm(total=n_keywords)  # noqa

    # Compute errors
    errors: List[float] = [calc_error(embs=embs,
                                      cand_emb=cand_vec,
                                      tfs=tfs,
                                      idfs=idfs,
                                      keyword_embs=keyword_embs) for cand_vec in embs]

    # Choose words which marked least error as a new keyword
    argmin: int = np.argmin(errors)
    offsetted: int = offsetted_ind(argmin, prev_key_inds)

    # Stop selection if all words are selected
    # or if # keywords are equal to specified one
    if (len(embs) == 1) or (n_keywords == 1):
        return [offsetted]
    else:
        new_dims: np.ndarray = _get_new_kemb_cand(cand_emb=embs[argmin],
                                                  keyword_embs=keyword_embs)
        pbar.update(1)
        inds: np.ndarray = np.ones(len(embs), bool)
        inds[argmin] = False
        res_embs: np.ndarray = embs[inds, :]
        res_tfs: np.ndarray = tfs[inds]
        res_idfs: np.ndarray = idfs[inds]
        return [offsetted] + get_keyword_inds(embs=res_embs,
                                              keyword_embs=new_dims,
                                              n_keywords=(n_keywords - 1),
                                              tfs=res_tfs,
                                              idfs=res_idfs,
                                              prev_key_inds=prev_key_inds + [offsetted],
                                              pbar=pbar)
