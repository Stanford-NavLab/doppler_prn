import time
import numpy as np
from doppler_prn import *

np.random.seed(0)

m, n = 31, 1023
a, b = np.random.choice(m), np.random.choice(n)

# weights = triangle_expected_doppler_weights(6e3, 1.0 / 1.023e6, n)
x = randb(m, n)

f, t = 6e3, 1.0 / 1.023e6
print(xcor_mag2_at_doppler(x, f, t))

weights = doppler_weights(f, t, n)
print(xcors_mag2_direct(x, weights))


# xcors_mag2(x, weights)
# start = time.perf_counter()
# xcors_mag2(x, weights)
# elapsed = time.perf_counter() - start
# print("obj time", elapsed)


# x2 = x.copy()
# x2[a, b] *= -1
# print(xcors_mag2(x2, weights) - xcors_mag2(x, weights))


# extended_weights, codes_fft, codes_padded_fft = precompute_terms(x, weights)
# print(
#     alt_delta_xcors_mag2(
#         a, b, x, weights, extended_weights, codes_fft, codes_padded_fft
#     )
# )
# print(delta_xcors_mag2(a, b, x, weights, extended_weights, codes_fft, codes_padded_fft))


# alt_time = 0.0
# for _ in range(100):
#     start = time.perf_counter()
#     alt_delta_xcors_mag2(
#         a, b, x, weights, extended_weights, codes_fft, codes_padded_fft
#     )
#     elapsed = time.perf_counter() - start
#     alt_time += elapsed / 100
# print("alt it/s:", 1.0 / alt_time, alt_time)


# orig_time = 0.0
# for _ in range(100):
#     start = time.perf_counter()
#     delta_xcors_mag2(a, b, x, weights, extended_weights, codes_fft, codes_padded_fft)
#     elapsed = time.perf_counter() - start
#     orig_time += elapsed / 100
# print("ori it/s:", 1.0 / orig_time, orig_time)
