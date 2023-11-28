"""Coordinate/bit-flip descent optimizer for Doppler PRN codes."""
import pickle
import numpy as np
from tqdm import tqdm
from numba import float64, njit
from numba.typed import List

from .core import (
    delta_xcors_mag2,
    xcors_mag2,
    precompute_terms,
    update_terms,
)


@njit
def step(a, b, m, n):
    """Update a, b to next bit to test."""
    if a == m - 1:
        a = 0
        if b == n - 1:
            b = 0
        else:
            b += 1
    else:
        a += 1

    return a, b


def optimize(
    codes,
    weights,
    n_iter=10,
    patience=-1,
    compute_initial_obj=True,
    log_freq=1000,
    log_path="",
):
    """Optimize initial code using bit flip descent. Test bits sequentially,
    columnwise."""
    codes = codes.copy()
    m, n = codes.shape

    # by default, patience is the number of bits
    if patience < 0:
        patience = m * n

    # precompute terms
    extended_weights, codes_fft, codes_padded_fft = precompute_terms(codes, weights)

    # save changes in objective
    obj = List.empty_list(float64)

    # initial objective value
    if compute_initial_obj:
        curr_obj = xcors_mag2(codes, weights)
    else:
        curr_obj = 0.0
    obj.append(curr_obj)

    iters_not_improved = 0
    a, b = 0, 0
    for iter in tqdm(range(n_iter)):
        # check effect of flipping code[a, b]
        delta = delta_xcors_mag2(
            a, b, codes, weights, extended_weights, codes_fft, codes_padded_fft
        )

        if delta <= 0:
            # flip the bit, update precomputable terms
            update_terms(a, b, codes, codes_fft, codes_padded_fft)

            # update logs
            curr_obj += delta
            obj.append(curr_obj)
            iters_not_improved = 0
        else:
            # do not flip the bit
            obj.append(curr_obj)
            iters_not_improved += 1

        # stop if no improvement possible
        if iters_not_improved == patience:
            break

        # write to log
        if log_path != "" and iter % log_freq == 0:
            pickle.dump(
                {
                    "codes": codes,
                    "obj": np.asarray(obj),
                },
                open(log_path + ".pkl", "wb"),
            )

        # update a, b
        a, b = step(a, b, m, n)

    # write to log
    if log_path != "" and iter % log_freq == 0:
        pickle.dump(
            {
                "codes": codes,
                "obj": np.asarray(obj),
            },
            open(log_path + ".pkl", "wb"),
        )

    return codes, np.asarray(obj)
