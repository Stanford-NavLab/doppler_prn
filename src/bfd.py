""" Bit flip descent for minimizing the expected sum of squared auto and
cross-correlations over random Doppler shift """
import numpy as np
from numba import njit, prange, int32, float64


@njit(fastmath=True)
def doppler_weights(f, t, n):
    """Weights matrix for Doppler frequency f and chip period t"""
    M = np.empty((n, n), dtype=float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = np.cos(2 * np.pi * f * t * (i - j))
    return M


@njit(fastmath=True)
def unif_expected_doppler_weights(f_max, t, n, n_grid_points=0):
    """Weights matrix for uniformly distributed Doppler frequency [-f_max,
    f_max] and chip period t"""
    if n_grid_points > 0:
        # discrete approximation
        M = np.zeros((n, n), dtype=float64)
        for f in np.linspace(-f_max, f_max, n_grid_points):
            M += doppler_weights(f, t, n) / n_grid_points
    else:
        # compute exact expected value
        M = np.empty((n, n), dtype=float64)
        for i in range(n):
            for j in range(n):
                M[i, j] = np.sinc(2 * f_max * t * (i - j))
    return M


@njit(fastmath=True)
def weighted_cc_sq_mag(x, y, weights):
    """Squared magnitudes of weighted cross-correlation of two vectors"""
    x, y = x.astype(float64), y.astype(float64)
    n = len(x)
    z = np.empty(n, dtype=float64)
    for k in range(n):
        xy = x * np.roll(y, k)
        z[k] = xy.T @ weights @ xy
    return z


@njit(fastmath=True, parallel=True)
def delta_weighted_ac_sq_mag(i, x, weights):
    """Change in weighted squared magnitude of weighted auto-correlation of a
    vector after flipping x[i]

    # A brute force implementation:
    #
    # for m1 in range(n):
    #     for m2 in range(m1, n):
    #         match_count = (
    #             (i == m1) + (i == m2) + (i == (m1 - k) % n) + (i == (m2 - k) % n)
    #         )
    #         if match_count == 1 or match_count == 3:
    #             delta[k] -= (
    #                 (4 if m1 != m2 else 2)
    #                 * x[m1]
    #                 * x[m2]
    #                 * x[(m1 - k) % n]
    #                 * x[(m2 - k) % n]
    #                 * weights[m1, m2]
    #             )
    # The four terms 1,2,3,4 are given by:
    # i == m1, i == m2, i == (m1 - k) % n, i == (m2 - k) % n
    # There are 8 combinations: 1,2,3,4,123,124,134,234
    # that lead to match_count == 1 or 3
    # case 123 cannot happen, since then i == m1 and i == (m1 - k) % n for k != 0
    # cases 124, 134, 234 also cannot happen for same reason --
    # case 5: i == m1, i == m2, i == (m1 - k) % n, i != (m2 - k) % n
    # case 6: i == m1, i == m2, i != (m1 - k) % n, i == (m2 - k) % n
    # case 7: i == m1, i != m2, i == (m1 - k) % n, i == (m2 - k) % n
    # case 8: i != m1, i == m2, i == (m1 - k) % n, i == (m2 - k) % n
    """
    n = len(x)
    delta = np.zeros(n, dtype=float64)
    for k in prange(1, n):
        ipk = (i + k) % n
        imk = (i - k) % n
        change = 0
        for m in range(n):
            mmk = (m - k) % n
            if m != i and m != ipk:
                # case 1: i == m1, i != m2, i != (m2 - k) % n
                change += 2 * x[i] * x[m] * x[imk] * x[mmk] * weights[i, m]
                # case 2: i == m2, i != m1, i != (m1 - k) % n
                change += 2 * x[m] * x[i] * x[mmk] * x[imk] * weights[m, i]
                # case 3: i != m1, i != m2, i == (m1 - k) % n, i != (m2 - k) % n
                # m1 = (i + k) % n, i != m2, i != (m2 - k) % n
                change += 2 * x[ipk] * x[m] * x[i] * x[mmk] * weights[ipk, m]
                # case 4: i != m1, i != m2, i != (m1 - k) % n, i == (m2 - k) % n
                # m2 = (i + k) % n, i != m1, i != (m1 - k) % n
                change += 2 * x[m] * x[ipk] * x[mmk] * x[i] * weights[m, ipk]
        delta[k] -= change
    return delta


@njit(fastmath=True, parallel=True)
def left_delta_weighted_cc_sq_mag(i, x, y, weights):
    """Change in weighted squared magnitude of weighted cross-correlation of two
    vectors after flipping x[i]"""
    n = len(x)
    delta = np.zeros(n, dtype=float64)
    for k in prange(n):
        for m in range(i):
            delta[k] -= (
                4 * x[i] * x[m] * y[(i - k) % n] * y[(m - k) % n] * weights[i, m]
            )
        for m in range(i + 1, n):
            delta[k] -= (
                4 * x[i] * x[m] * y[(i - k) % n] * y[(m - k) % n] * weights[i, m]
            )
    return delta


@njit(fastmath=True, parallel=True)
def right_delta_weighted_cc_sq_mag(i, x, y, weights):
    """Change in weighted squared magnitude of weighted cross-correlation of two
    vectors after flipping y[i]"""
    n = len(x)
    delta = np.zeros(n, dtype=float64)
    for k in prange(n):
        m1 = (i + k) % n
        for m2 in range(m1):
            delta[k] -= 4 * x[m1] * x[m2] * y[i] * y[(m2 - k) % n] * weights[m1, m2]
        for m2 in range(m1 + 1, n):
            delta[k] -= 4 * x[m1] * x[m2] * y[i] * y[(m2 - k) % n] * weights[m1, m2]
    return delta


if __name__ == "__main__":
    n = 1000
    f, t = 3 * np.random.rand(), np.random.rand()
    # M = doppler_weights(f, t, n)
    M = unif_expected_doppler_weights(f, t, n, n_grid_points=0)

    x = np.random.choice((-1, 1), n)
    y = np.random.choice((-1, 1), n)

    i = np.random.choice(n)
    x2 = x.copy()
    x2[i] *= -1
    y2 = y.copy()
    y2[i] *= -1

    delta_weighted_ac_sq_mag(i, x, M)
    left_delta_weighted_cc_sq_mag(i, x, y, M)
    right_delta_weighted_cc_sq_mag(i, x, y, M)

    import time

    avg = 0
    for _ in range(10):
        start = time.perf_counter()
        # delta_weighted_ac_sq_mag(i, x, M)
        np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(y)))
        # left_delta_weighted_cc_sq_mag(i, x, y, M)
        # right_delta_weighted_cc_sq_mag(i, x, y, M)
        avg += (time.perf_counter() - start) / 10
    print(avg)

    true = (weighted_cc_sq_mag(x2, x2, M) - weighted_cc_sq_mag(x, x, M)).sum()
    pred = delta_weighted_ac_sq_mag(i, x, M).sum()
    print("ac pass", np.allclose(true, pred))

    true = (weighted_cc_sq_mag(x2, y, M) - weighted_cc_sq_mag(x, y, M)).sum()
    pred = left_delta_weighted_cc_sq_mag(i, x, y, M).sum()
    print("left pass", np.allclose(true, pred))

    true = (weighted_cc_sq_mag(x, y2, M) - weighted_cc_sq_mag(x, y, M)).sum()
    pred = right_delta_weighted_cc_sq_mag(i, x, y, M).sum()

    print("right pass", np.allclose(true, pred))
