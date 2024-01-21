import os
import pickle
from doppler_prn import *


if __name__ == "__main__":
    results = {}

    # gold codes, 31 x 1023
    f, t = 4.4e3, 9.77517107e-7
    Xg = gold_codes(1023)
    weights = unif_expected_doppler_weights(2 * f, t, 1023)
    codes = Xg[:31, :]
    results["gold_31_1023_codes"] = codes
    results["gold_31_1023_obj"] = xcors_mag2(codes, weights)
    results["gold_31_1023_obj0"] = xcors_mag2_at_reldop(codes, 0, 0)
    results["gold_31_1023_rel_freqs"] = np.linspace(-f, f, 3000) * 3
    results["gold_31_1023_obj_vs_rel_freqs"] = []
    for freq in results["gold_31_1023_rel_freqs"]:
        results["gold_31_1023_obj_vs_rel_freqs"].append(
            xcors_mag2_at_reldop(codes, freq, t)
        )

    # gold codes, 300 x 1023
    f, t = 29.6e3, 2e-7
    weights = unif_expected_doppler_weights(2 * f, t, 1023)
    codes = Xg[:300, :]
    results["gold_300_1023_codes"] = codes
    results["gold_300_1023_obj"] = xcors_mag2(codes, weights)
    results["gold_300_1023_obj0"] = xcors_mag2_at_reldop(codes, 0, 0)
    results["gold_300_1023_rel_freqs"] = np.linspace(-f, f, 3000) * 3
    results["gold_300_1023_obj_vs_rel_freqs"] = []
    for freq in results["gold_300_1023_rel_freqs"]:
        results["gold_300_1023_obj_vs_rel_freqs"].append(
            xcors_mag2_at_reldop(codes, freq, t)
        )

    # weil codes
    f, t = 9.5e3, 1.941747572815534e-7
    weights = unif_expected_doppler_weights(2 * f, t, 5113)
    codes = weil_codes(5113)[:8, :]
    results["weil_8_5113_codes"] = codes
    results["weil_8_5113_obj"] = xcors_mag2(codes, weights)
    results["weil_8_5113_obj0"] = xcors_mag2_at_reldop(codes, 0, 0)
    results["weil_8_5113_rel_freqs"] = np.linspace(-f, f, 3000) * 3
    results["weil_8_5113_obj_vs_rel_freqs"] = []
    for freq in results["weil_8_5113_rel_freqs"]:
        results["weil_8_5113_obj_vs_rel_freqs"].append(
            xcors_mag2_at_reldop(codes, freq, t)
        )

    # extended weil codes, 31 x 10230
    f, t = 3.3e3, 9.77517107e-8
    weights = unif_expected_doppler_weights(2 * f, t, 10230)
    Xw = weil_codes(10230 - 7)
    np.random.seed(0)
    codes = np.hstack((Xw[:31, :], randb((31, 7))))
    results["weil_31_10230_codes"] = codes
    results["weil_31_10230_obj"] = xcors_mag2(codes, weights)
    results["weil_31_10230_obj0"] = xcors_mag2_at_reldop(codes, 0, 0)
    results["weil_31_10230_rel_freqs"] = np.linspace(-f, f, 3000) * 3
    results["weil_31_10230_obj_vs_rel_freqs"] = []
    for freq in results["weil_31_10230_rel_freqs"]:
        results["weil_31_10230_obj_vs_rel_freqs"].append(
            xcors_mag2_at_reldop(codes, freq, t)
        )

    pickle.dump(results, open(os.path.join("results", "gold_weil.pkl"), "wb"))
