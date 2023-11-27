import numpy as np
import time
from doppler_prn import unif_expected_doppler_weights, optimize

np.random.seed(0)

f, t = 5e3, 1.0 / 1.023e6
m, n = 31, 1023
max_iters = 10_000_000
weights = unif_expected_doppler_weights(f, t, n)
initial_codes = np.random.choice((-1, 1), size=(m, n))

start = time.perf_counter()
optimized_codes, obj_vals = optimize(
    initial_codes, weights, n_iter=max_iters, compute_initial_obj=True
)
elapsed = time.perf_counter() - start
print("elapsed:", elapsed)
print("objective:", obj_vals[-1])

np.save("optimized_codes.npy", optimized_codes)
np.save("obj_vals.npy", obj_vals)

# import matplotlib.pyplot as plt

# plt.plot(obj_vals)
# plt.show()
