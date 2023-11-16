""" Data structures and functions for PRN code families """
import numpy as np
from numba import njit, prange, int8, int32, float32, float64
from numba.experimental import jitclass
from numba.typed import List

spec = [
    ('family_size', int32),
    ('length', int32),
    ('code', int8[:,:]),
    ('correlations', int32[:, :, :])
]

@jitclass(spec)
class PRN(object):
    def __init__(self, family_size : int, length : int) -> None:
        assert family_size > 0 and length > 0, "Family size and length must be positive integers"
        assert length <= 2 ** 31 - 1, "Length must be less than 2^31 - 1"

        self.m, self.n = family_size, length

        # code is a m x n matrix of 1s and -1s, once initialized
        self.code = np.empty((family_size, length), dtype=int8)

        # track auto and cross correlation values
        self.correlations = np.empty((family_size, family_size, length), dtype=int32)

    @property
    def size(self):
        return self.m, self.n


    def __getitem__(self, key):
        return self.code[key]

    def __setitem__(self, key, value):
        # TODO
        self.code[key] = value

    def compute_correlations(self):
        """ Compute all auto- and cross-correlations """
        pass

def compute_correlations(x):
    """ Compute all auto- and cross-correlations """
    spectra = List()
    for i in range(x.shape[0]):
        spectra.append(np.fft.fft(x[i]))


def fill_random(prn : PRN):
    """ Fill a PRN code with random values """
    prn.code = np.random.choice(np.array([-1, 1], dtype=np.int8), size=prn.size)

    # @staticmethod
    # def randb(family_size, length):
    #     """ Generate a random binary code of length `length` and family size `family_size`"""
    #     return np.random.choice(np.array([-1, 1], dtype=np.int8), size=(family_size, length))


@njit
def dsos_fast(x, i, s, current_correlations, doppler_f=0, doppler_t=0):
    """ Change in sum of squared correlation due to flipping a bit x[i, s]
        Convention: current_correlations[i, j] for i <= j is vector of
        cross-correlation between codes i and j at all shifts
    """
    doppler_periods = doppler_f * doppler_t
    delta = 0.
    m, n = x.shape
    for j in range(m):
        _i, _j = min(i, j), max(i, j)
        for k in range(1 if _j == _i else 0, n):
            if _j == _i:
                # update autocorrelation values at nonzero shift
                delta_correlation_ijk = 2 * x[_i, s] * (x[_i, (s + k) % n] + x[_i, (s - k) % n])
            else:
                # update cross-correlation values at all shifts
                delta_correlation_ijk = 2 * x[_i, s] * x[_j, (s - k) % n]

            if doppler_periods:
                pass
            else:
                delta += delta_correlation_ijk ** 2 - 2 * delta_correlation_ijk * current_correlations[_i, _j, k]

    return delta

@njit
def dcorr(x, i, s, doppler_f=0, doppler_t=0):
    doppler_periods = doppler_f * doppler_t
    m, n = x.shape
    inds = np.empty((m * n - 1, 3), dtype=int32)
    deltas = np.empty(m * n - 1, dtype=float32)

    idx = 0
    for j in range(m):
        _i, _j = min(i, j), max(i, j)
        for k in range(1 if _j == _i else 0, n):
            if _j == _i:
                # update autocorrelation values at nonzero shift
                deltas[idx] = 2 * x[_i, s] * (x[_i, (s + k) % n] + x[_i, (s - k) % n])
            else:
                # update cross-correlation values at all shifts
                deltas[idx] = 2 * x[_i, s] * x[_j, (s - k) % n]

            inds[idx, 0], inds[idx, 1], inds[idx, 2] = _i, _j, k
            idx += 1

    return inds, deltas

@njit
def dsos(x, i, s, current_correlations):
    inds, deltas = dcorr(x, i, s)
    delta = 0.
    for i in range(len(deltas)):
        delta += deltas[i] ** 2 - 2 * deltas[i] * current_correlations[inds[i,0], inds[i,1], inds[i,2]]
    return delta

if __name__ == "__main__":
    x = np.random.choice((-1,1), size=(127, 257))
    corrs = np.empty((x.shape[0], x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            full_corr = np.fft.ifft(np.fft.fft(x[i, :]) * np.conj(np.fft.fft(x[j, :]))).real
            if i == j: full_corr[0] = 0
            corrs[i, j, :] = full_corr

    curr_sos = np.sum([corrs[i, j] ** 2 for i in range(x.shape[0]) for j in range(i, x.shape[0])])

    i, s = 0, 2
    import time

    pred_delta = dsos(x, i, s, corrs)
    elapsed = 0.
    for _ in range(100):
        start = time.perf_counter()
        dsos(x, i, s, corrs)
        elapsed += time.perf_counter() - start
    print("predicted delta", pred_delta, "time: ", elapsed / 100)

    # flip bits
    x[i, s] *= -1
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            full_corr = np.fft.ifft(np.fft.fft(x[i, :]) * np.conj(np.fft.fft(x[j, :]))).real
            if i == j: full_corr[0] = 0
            corrs[i, j, :] = full_corr

    new_sos = np.sum([corrs[i, j] ** 2 for i in range(x.shape[0]) for j in range(i, x.shape[0])])

    print("actual delta", new_sos - curr_sos)
