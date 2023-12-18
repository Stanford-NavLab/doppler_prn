import os
import pickle
from doppler_prn import *


if __name__ == "__main__":
    results = {}

    # gold codes, 31 x 1023
    Xg = gold_codes(1023)
    weights = triangle_expected_doppler_weights(6e3, 9.77517107e-7, 1023)
    codes = Xg[:31, :]
    results["gold_31_1023_codes"] = codes
    results["gold_31_1023_obj"] = xcors_mag2(codes, weights)

    # gold codes, 300 x 1023
    weights = triangle_expected_doppler_weights(29.6e3, 2e-7, 1023)
    codes = Xg[:300, :]
    results["gold_300_1023_codes"] = codes
    results["gold_300_1023_obj"] = xcors_mag2(codes, weights)

    # extended weil codes, 31 x 10230
    weights = triangle_expected_doppler_weights(4.5e3, 9.77517107e-8, 10230)
    Xw = weil_codes(10230 - 7)
    np.random.seed(0)
    codes = np.hstack((Xw[:31, :], randb(31, 7)))
    results["weil_31_10230_codes"] = codes
    results["weil_31_10230_obj"] = xcors_mag2(codes, weights)

    pickle.dump(results, open(os.path.join("results", "gold_weil.pkl"), "wb"))
