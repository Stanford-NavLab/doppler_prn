"""Core functions for Doppler PRN optimization"""
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange, float64, complex128

from rocket_fft import numpy_like

numpy_like()


def randb(m, n):
    """Generate a random m x n matrix of bits"""
    return np.random.choice((-1, 1), size=(m, n))


@njit(fastmath=True)
def unif_expected_doppler_weights(f_max, t, n, n_grid_points=0):
    """Weights matrix for uniformly distributed Doppler frequency [-f_max,
    f_max] and chip period t"""
    if n_grid_points > 0:
        # discrete approximation
        weights = np.zeros(n, dtype=float64)
        for f in np.linspace(-f_max, f_max, n_grid_points):
            weights += doppler_weights(f, t, n) / n_grid_points
        return weights

    # compute exact expected value
    return np.sinc(2 * f_max * t * np.arange(n))


@njit(fastmath=True)
def doppler_weights(f, t, n):
    """Weights matrix for Doppler frequency f and chip period t"""
    return np.cos(2 * np.pi * f * t * np.arange(n))


@njit(fastmath=True)
def xcor(x, y, weights):
    """Weighted cross-correlation of x and y:
    f(x,y)[k] = \sum_{m=0}^{n-1} x[m] y[(m - k)%n] w[m - k]
    """
    n = len(x)
    X = fft(np.hstack((x, np.zeros(n, dtype=float64))))
    Y = fft(np.hstack((y, y)) * weights)
    return ifft(X * Y.conj())[-n:].conj()


@njit(fastmath=True)
def toeplitz(c):
    """A clone of scipy.linalg.toeplitz, which creates a Toeplitz matrix from
    vector c."""
    c = np.asarray(c).ravel()
    r = c.conjugate()
    vals = np.concatenate((c[::-1], r[1:]))
    out_shp = len(c), len(r)
    n = vals.strides[0]
    return as_strided(vals[len(c) - 1 :], shape=out_shp, strides=(-n, n)).copy()


@njit(fastmath=True, parallel=True)
def calc_weights_toeplitz(f, t, n):
    """Weights matrix for Doppler frequency f and chip period t"""
    return np.cos(2 * np.pi * f * t * np.arange(n))


@njit(fastmath=True)
def xcor_mag2(x, y, weights):
    """Squared magnitude weighted cross-correlation of real x and y
    f(x,y)[k] = \sum_{m=0}^{n-1} \sum_{l=0}^{n-1} x[m] x[(m-k)%n] y[l]
    y[(l-k)%n] weights[m, l]"""
    X = np.outer(x, x) * toeplitz(weights)
    Y = np.outer(y, y)
    return np.diag(ifft2(fft2(X) * fft2(Y).conj()).real)


@njit(fastmath=True)
def xcor2_fft(X, Y):
    """2D cross-correlation of real x and y, given their 2D FFTs X and Y"""
    return ifft2(X * Y.conj()).real


@njit(fastmath=True, parallel=True)
def xcors_mag2(codes, weights):
    """Sum of squared magnitude weighted cross-correlations of codes, which is
    an m x n matrix of codes, where m is the number of codes and n is the code length.
    """
    m, n = codes.shape

    weights_matrix = toeplitz(weights)
    outer_ffts = np.empty((m, n, n), dtype=complex128)
    weighted_outer_ffts = np.empty((m, n, n), dtype=complex128)
    for i in prange(m):
        outer_prod = np.outer(codes[i, :], codes[i, :])
        outer_ffts[i, ...] = fft2(outer_prod)
        weighted_outer_ffts[i, ...] = fft2(outer_prod * weights_matrix)

    mag2 = np.empty((m, m))
    for i in prange(m):
        for j in prange(m):
            mag2[i, j] = np.trace(
                xcor2_fft(weighted_outer_ffts[i, ...], outer_ffts[j, ...])
            )
    return mag2.sum()


@njit(fastmath=True)
def flip_extend(w):
    """Extend w by flipping it and appending it to itself"""
    return np.hstack((np.array([w[0]]), np.flip(w[1:]), w))


@njit(fastmath=True)
def delta_acor_mag2(b, x, weights, extended_weights, x_fft, x0_fft):
    """Change in magnitude squared of weighted autocorrelation of x after
    flipping x[b].
    # # precomputable terms
    # extended_weights = flip_extend(weights)
    # x_fft = fft(x)
    # x0_fft = fft(np.hstack((x, np.zeros(n, dtype=float64))))
    """
    n = len(x)

    weights_b = np.array([weights[np.abs(b - m)] for m in range(n)])
    xb = np.roll(x, -b).astype(float64)
    corr1 = ifft(fft(x * weights_b) * x_fft.conj()).real[1:]

    yyd = np.hstack((xb, xb)) * extended_weights
    corr2 = np.roll(ifft(x0_fft * fft(yyd).conj()).real[-n:], -b)[1:]

    xbmk = np.roll(np.flip(x), b + 1).astype(float64)
    wb = np.roll(weights_b, -b)[1:]

    v1 = x[b] * xbmk[1:]
    v2 = x[b] * xb[1:]

    delta1 = xbmk[1:] @ (corr1 - v1 - v2 * wb)
    delta2 = xb[1:] @ (corr2 - v1 * wb - v2)
    return -4 * x[b] * (delta1 + delta2)


@njit(fastmath=True)
def delta_lxcor_mag2(b, x, y, weights, y_fft):
    """Change in magnitude squared of weighted crosscorrelation of x with y after
    flipping x[b].
    # # precomputable terms
    # y_fft = fft(y)
    """
    n = len(x)
    weights_b = np.array([weights[np.abs(b - m)] for m in range(n)])
    corr = ifft(fft(x * weights_b) * y_fft.conj()).real
    yb = np.flip(np.roll(y, -b - 1)).astype(float64)
    return -4 * x[b] * yb @ (corr - x[b] * yb)


@njit(fastmath=True)
def delta_rxcor_mag2(b, x, y, extended_weights, x0_fft):
    """Change in magnitude squared of weighted crosscorrelation of x with y after
    flipping y[b].
    # # precomputable terms
    # extended_weights = flip_extend(weights)
    # x0 = np.hstack((x, np.zeros(n)))
    # x0_fft = fft(x0)
    """
    n = len(x)

    yb = np.roll(y, -b)
    yyd = np.hstack((yb, yb)) * extended_weights
    corr = np.roll(ifft(x0_fft * fft(yyd).conj()).real[-n:], -b)
    xb = np.roll(x, -b).astype(float64)
    return -4 * y[b] * xb @ (corr - y[b] * xb)


@njit(fastmath=True, parallel=True)
def delta_xcors_mag2(
    a, b, codes, weights, extended_weights, codes_fft, codes_padded_fft
):
    """Change in sum of squared magnitude weighted cross-correlations of codes
    after flipping bit codes[a,b].
    # # precomputable terms
    # extended_weights = flip_extend(weights)
    # codes_fft = fft(codes)
    # codes_padded_fft = fft(np.hstack((codes, np.zeros(codes.shape)))
    """
    m = codes.shape[0]
    delta = np.empty(m)
    for i in prange(m):
        if i == a:
            delta[a] = delta_acor_mag2(
                b,
                codes[a, :],
                weights,
                extended_weights,
                codes_fft[a, :],
                codes_padded_fft[a, :],
            )
        else:
            delta[i] = delta_lxcor_mag2(
                b, codes[a, :], codes[i, :], weights, codes_fft[i, :]
            ) + delta_rxcor_mag2(
                b,
                codes[i, :],
                codes[a, :],
                extended_weights,
                codes_padded_fft[i, :],
            )

    return delta.sum()


@njit(fastmath=True, parallel=True)
def precompute_terms(codes, weights):
    """Precompute terms needed for bit flip descent."""
    extended_weights = flip_extend(weights)
    codes_fft = fft(codes)
    codes_padded_fft = fft(np.hstack((codes, np.zeros(codes.shape))))
    return extended_weights, codes_fft, codes_padded_fft


@njit(fastmath=True, parallel=True)
def update_terms(a, b, codes, codes_fft, codes_padded_fft):
    """Update precomputable terms after flipping codes[a, b]"""
    codes[a, b] *= -1
    codes_fft[a, :] = fft(codes[a, :])
    codes_padded_fft[a, :] = fft(np.hstack((codes[a, :], np.zeros(codes.shape[1]))))


@njit(fastmath=True)
def xcors_mag2_large(codes, weights):
    """Sum of squared magnitude weighted cross-correlations of codes, which is
    an m x n matrix of codes, where m is the number of codes and n is the code
    length. Start with code of all ones, and bit flip to get desired objective value."""
    m, n = codes.shape
    weights_matrix = toeplitz(weights)

    # start with codes of all ones, and then bit flip to get desired objective
    x = np.ones((m, n))
    obj = weights_matrix.sum() * n * m**2

    # precompute terms
    extended_weights, x_fft, x_padded_fft = precompute_terms(x, weights)

    for a in range(m):
        for b in range(n):
            if codes[a, b] != x[a, b]:
                obj += delta_xcors_mag2(
                    a, b, x, weights, extended_weights, x_fft, x_padded_fft
                )
                update_terms(a, b, x, x_fft, x_padded_fft)
    return obj
