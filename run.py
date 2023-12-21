import argparse
import pickle
import numpy as np
from doppler_prn import (
    triangle_expected_doppler_weights,
    expected_doppler_weights,
    optimize,
    randb,
    xcors_mag2,
    toeplitz,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", help="Random seed", type=int, default=0)
    parser.add_argument("--f", help="Doppler frequency", type=float, default=6e3)
    parser.add_argument("--t", help="Doppler period", type=float, default=1.0 / 1.023e6)
    parser.add_argument("--m", help="Number of codes", type=int, default=31)
    parser.add_argument("--n", help="Code length", type=int, default=1023)
    parser.add_argument(
        "--gs",
        help="Grid size for approximating expected value. Zero for exact expression when available",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--maxit", help="Maximum iterations", type=int, default=10_000_000
    )
    parser.add_argument(
        "--name", help="Base name for output files", type=str, default="gps_l1"
    )
    parser.add_argument("--log", help="Log write frequency", type=int, default=10_000)
    parser.add_argument(
        "--obj",
        help="Calculate initial objective",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--obj_v_freq",
        help="Calculate objective vs observed Doppler frequency",
        action=argparse.BooleanOptionalAction,
    )
    args = parser.parse_args()

    # experiment name
    exp_name = args.name
    if exp_name != "":
        exp_name += "_seed=%d" % args.s

    # weights defining cross-correlation with Doppler
    weights = triangle_expected_doppler_weights(
        args.f, args.t, args.n, n_grid_points=args.gs, normalize=True
    )

    # random initial codes
    np.random.seed(args.s)
    initial_codes = randb(args.m, args.n)
    log = optimize(
        initial_codes,
        weights,
        n_iter=args.maxit,
        patience=-1,
        compute_initial_obj=args.obj,
        log_freq=args.log,
        log_path=exp_name,
    )

    # objective vs observed Doppler frequency
    if args.obj_v_freq:
        freqs = np.linspace(-args.f, args.f, 50)
        objs = []
        for freq in freqs:
            weights = expected_doppler_weights(
                freq, args.f, args.t, args.n, normalize=True
            )
            objs.append(xcors_mag2(log["codes"], weights, normalize=True))
        log["doppler_freq"] = freqs
        log["obj_vs_freq"] = np.array(objs)

        if exp_name != "":
            pickle.dump(log, open(exp_name + ".pkl", "wb"))
