from tqdm import tqdm
import numpy as np
from numpy.fft import fft
from numba import float64
from numba.typed import List

from rocket_fft import numpy_like

numpy_like()

from .core import flip_extend, delta_xcors_mag2, xcors_mag2


def optimize(codes, weights, n_iter=10, patience=-1, compute_initial_obj=True):
    """Optimize initial code using bit flip descent. Test bits sequentially,
    columnwise."""
    codes = codes.copy()
    m, n = codes.shape

    # by default, patience is the number of bits
    if patience < 0:
        patience = m * n

    # precompute terms
    extended_weights = flip_extend(weights)
    codes_fft = fft(codes)
    codes_padded_fft = fft(np.hstack((codes, np.zeros((m, n)))))

    # initial objective value
    curr_obj = xcors_mag2(codes, weights) if compute_initial_obj else 0.0

    # save changes in objective
    obj = List.empty_list(float64)

    iters_not_improved = 0
    a, b = 0, 0
    for _ in tqdm(range(n_iter)):
        # check effect of flipping code[a, b]
        delta = delta_xcors_mag2(
            a, b, codes, weights, extended_weights, codes_fft, codes_padded_fft
        )

        if delta <= 0:
            # flip the bit, update precomputable terms
            codes[a, b] *= -1
            codes_fft[a, :] = fft(codes[a, :])
            codes_padded_fft[a, :] = fft(np.hstack((codes[a, :], np.zeros(n))))

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

        # update a, b
        if a == m - 1:
            a = 0
            if b == n - 1:
                b = 0
            else:
                b += 1
        else:
            a += 1

    return codes, obj
