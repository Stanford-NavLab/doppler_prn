import os
import pickle
from doppler_prn import *


if __name__ == "__main__":
    results = {}
    gold_weil_data = pickle.load(open(os.path.join("results", "gold_weil.pkl"), "rb"))

    # 31 x 1023
    freqs = np.linspace(-6e3, 6e3, 50)
    opt = pickle.load(open(os.path.join("results", "gps_exact_seed=0.pkl"), "rb"))
    gold_31_1023 = gold_weil_data["gold_31_1023_codes"]
    results["gps"] = {
        "freqs": freqs,
        "opt": np.array(
            [xcor_mag2_at_doppler(opt["codes"], f, 9.77517107e-7) for f in freqs]
        ),
        "bench": np.array(
            [
                xcor_mag2_at_doppler(gold_31_1023["codes"], f, 9.77517107e-7)
                for f in freqs
            ]
        ),
    }

    # 31 x 10230
    freqs = np.linspace(-4.5e3, 4.5e3, 50)
    opt = pickle.load(open(os.path.join("results", "gpsl5_10_seed=0.pkl"), "rb"))
    weil_300_10230 = gold_weil_data["weil_31_10230_codes"]
    results["gpsl5"] = {
        "freqs": freqs,
        "opt": np.array(
            [xcor_mag2_at_doppler(opt["codes"], f, 9.77517107e-8) for f in freqs]
        ),
        "bench": np.array(
            [
                xcor_mag2_at_doppler(weil_300_10230["codes"], f, 9.77517107e-8)
                for f in freqs
            ]
        ),
    }

    # 300 x 1023
    freqs = np.linspace(-29.6e3, 29.6e3, 50)
    opt = pickle.load(open(os.path.join("results", "leo_1023_exact_seed=0.pkl"), "rb"))
    gold_300_1023 = gold_weil_data["gold_300_1023_codes"]
    results["leo_1023"] = {
        "freqs": freqs,
        "opt": np.array([xcor_mag2_at_doppler(opt["codes"], f, 2e-7) for f in freqs]),
        "bench": np.array(
            [xcor_mag2_at_doppler(gold_300_1023["codes"], f, 2e-7) for f in freqs]
        ),
    }

    pickle.dump(results, open(os.path.join("results", "obj_v_freqs.pkl"), "wb"))
