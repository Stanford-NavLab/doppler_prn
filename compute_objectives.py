import os
import pickle
from doppler_prn import xcors_mag2, precompute_terms


if __name__ == "__main__":
    for fname in os.listdir("results"):
        if not fname.endswith(".pkl"):
            continue

        # read file
        data = pickle.load(open(os.path.join("results", fname), "rb"))

        # compute true final objective
        if "exact" in fname:
            # no need to compute true objective
            data["final_obj_true"] = data["obj"][-1]
        elif "seed=0" in fname:
            # compute true objective
            codes, weights = data["codes"], data["weights"]
            extended_weights, codes_fft, codes_padded_fft = precompute_terms(
                codes, weights
            )
            data["final_obj_true"] = xcors_mag2(codes, weights)

        # rewrite file
        pickle.dump(data, open(os.path.join("results", fname), "wb"))
