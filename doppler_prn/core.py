"""Core functions for Doppler PRN optimization"""
import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2
from numpy.lib.stride_tricks import as_strided
from numba import njit, prange, float64, complex128


def randb(m, n):
    """Generate a random m x n matrix of bits"""
    return np.random.choice((-1, 1), size=(m, n))


@njit(fastmath=True)
def unif_expected_doppler_weights(f_max, t, n, n_grid_points=0):
    """Weights matrix for uniformly distributed relative Doppler
    frequency [-f_max, f_max] and chip period t"""
    if n_grid_points > 0:
        # discrete approximation
        weights = np.zeros(n, dtype=float64)
        for f in np.linspace(-f_max, f_max, n_grid_points):
            weights += doppler_weights(f, t, n) / n_grid_points
    else:
        # compute exact expected value
        weights = np.sinc(2 * f_max * t * np.arange(n))

    return weights


@njit(fastmath=True)
def triangle_expected_doppler_weights(f_max, t, n, n_grid_points=0, normalize=False):
    """Weights matrix for triangularly distributed relative Doppler
    frequency [-f_max, f_max] and chip period t"""
    if normalize:
        assert n_grid_points > 0
        weights = np.zeros(n, dtype=float64)
        for f_dop in np.linspace(-f_max, f_max, n_grid_points):
            weights_f_dop = np.zeros(n, dtype=float64)
            for f_rec in np.linspace(-f_max, f_max, n_grid_points):
                weights_f_dop += doppler_weights(f_rec - f_dop, t, n) / n_grid_points
            weights_f_dop /= toeplitz(weights_f_dop).sum()
            weights += weights_f_dop / n_grid_points
    else:
        if n_grid_points > 0:
            # discrete approximation
            weights = np.zeros(n, dtype=float64)
            for f in np.linspace(-f_max, f_max, n_grid_points):
                for f2 in np.linspace(-f_max, f_max, n_grid_points):
                    weights += doppler_weights(f - f2, t, n) / n_grid_points**2
        else:
            # compute exact expected value
            weights = np.sinc(2 * f_max * t * np.arange(n)) ** 2

    return weights


@njit(fastmath=True)
def doppler_weights(f, t, n):
    """Weights vector for relative Doppler frequency f and chip period t"""
    return np.cos(2 * np.pi * f * t * np.arange(n))


@njit(fastmath=True)
def expected_doppler_weights(f, f_max, t, n, n_grid_points=0, normalize=False):
    """Weights matrix for Doppler frequency f, chip period t, and receiver sweep
    frequency [-f_max, f_max]"""
    if n_grid_points > 0:
        weights = np.zeros(n)
        for f1 in np.linspace(-f_max, f_max, n_grid_points):
            weights += doppler_weights(f1 - f, t, n) / n_grid_points
    else:
        weights = np.ones(n)
        num = np.sin(2 * np.pi * (f_max - f) * t * np.arange(n)) + np.sin(
            2 * np.pi * (f_max + f) * t * np.arange(n)
        )
        den = 4 * np.pi * f_max * t * np.arange(n)
        weights[den != 0] = num[den != 0] / den[den != 0]

    if normalize:
        return weights / toeplitz(weights).sum()

    return weights


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
def xcors_mag2_direct(codes, weights):
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


@njit(fastmath=True, parallel=True)
def xcor_mag2_at_doppler(codes, f, t):
    """Sum of squared magnitude weighted cross-correlations of codes, which is
    an m x n matrix of codes, where m is the number of codes and n is the code length.
    Evaluated at relative Doppler frequency f and chip period t"""
    m, n = codes.shape
    d = np.array([np.exp(-2j * np.pi * f * t * k) for k in range(-n, n)])
    obj = np.empty((m, m))
    for i in prange(m):
        for j in prange(m):
            x0 = np.hstack((codes[i, :], np.zeros(n)))
            yyd = np.hstack((codes[j, :], codes[j, :])) * d
            xcor_val = ifft(fft(x0) * fft(yyd).conj())[-n:].conj()
            obj[i, j] = np.sum(np.abs(xcor_val) ** 2)

    return obj.sum()


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

    delta1 = xbmk[1:] @ (corr1 - v1 * weights[0] - v2 * wb)
    delta2 = xb[1:] @ (corr2 - v1 * wb - v2 * weights[0])
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
    return -4 * x[b] * yb @ (corr - x[b] * yb * weights[0])


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
    return -4 * y[b] * xb @ (corr - y[b] * xb * extended_weights[0])


@njit(fastmath=True, parallel=True)
def alt_delta_xcors_mag2(
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


@njit(inline="always")
def reverse_roll(a, left_shift):
    """Equivalent to np.roll(a, -left_shift), with left_shift > 0"""
    b = np.empty_like(a)
    n = len(a)
    b[0 : n - left_shift] = a[left_shift:n]
    b[n - left_shift : n] = a[0:left_shift]
    return b


@njit(parallel=True, inline="always")
def get_toeplitz_column(weights, b):
    """Equivalent to toeplitz(weights)[:, b]"""
    c = np.empty_like(weights)
    for i in prange(len(weights)):
        c[i] = weights[np.abs(b - i)]
    return c


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
    m, n = codes.shape

    weights_b = get_toeplitz_column(weights, b)
    weights_b_rolled = np.roll(weights_b, -b)[1:]
    weighted_codes = codes * weights_b

    xab = codes[a, b]
    xa_rolled = reverse_roll(codes[a, :], b).astype(float64)
    x_weighted = weighted_codes[a, :]
    weighted_xx_rolled_fft_conj = fft(
        np.hstack((xa_rolled, xa_rolled)) * extended_weights
    ).conj()

    delta = np.empty(m)
    for i in prange(m):
        if i == a:
            corr1 = ifft(fft(x_weighted) * codes_fft[a, :].conj()).real[1:]
            corr2 = reverse_roll(
                ifft(codes_padded_fft[a, :] * weighted_xx_rolled_fft_conj).real[-n:], b
            )[1:]

            v1 = np.roll(np.flip(codes[a, :]), b + 1)[1:].astype(float64)
            v2 = xa_rolled[1:]
            v3 = v1 * v2 * weights_b_rolled

            delta1 = v1 * corr1 - xab * (weights[0] * v1**2 + v3)
            delta2 = v2 * corr2 - xab * (weights[0] * v2**2 + v3)

            delta[a] = -4 * xab * np.sum(delta1 + delta2)
        else:
            y_fl_b = reverse_roll(codes[i, :], b + 1)[::-1].astype(float64)
            y_rb = reverse_roll(codes[i, :], b).astype(float64)

            corr1 = ifft(fft(x_weighted) * codes_fft[i, :].conj()).real
            corr2 = reverse_roll(
                ifft(codes_padded_fft[i, :] * weighted_xx_rolled_fft_conj).real[-n:], b
            )

            ip1 = y_fl_b * (corr1 - xab * y_fl_b * weights[0])
            ip2 = y_rb * (corr2 - xab * y_rb * weights[0])

            delta[i] = -4 * xab * np.sum(ip1 + ip2)

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
def xcors_mag2(codes, weights, normalize=False):
    """Sum of squared magnitude weighted cross-correlations of codes, which is
    an m x n matrix of codes, where m is the number of codes and n is the code
    length. Start with code of all ones, and bit flip to get desired objective value."""
    m, n = codes.shape

    # start with codes of all ones, and then bit flip to get desired objective
    x = np.ones((m, n))
    obj = (1.0 if normalize else toeplitz(weights).sum()) * n * m**2

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
