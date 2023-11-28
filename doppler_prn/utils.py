import numpy as np
import matplotlib.pyplot as plt
import itertools
import os, pickle
from types import SimpleNamespace
import objectives


def get_rand_codeset_bin01(code_shape, p_vals=None):
    if p_vals is None:
        p_vals = 0.5 * np.ones(code_shape)
    assert chk_shape(p_vals, code_shape), (
        "p_vals must have same shape as input shape. p_vals shape: "
        + str(p_vals.shape)
        + " input shape: "
        + str(code_shape)
    )
    s_unif = np.random.uniform(size=code_shape)
    return 1.0 * (s_unif < p_vals)


def get_rand_codeset_bin01_samples(code_shape, popsize, p_vals=None):
    if p_vals is None:
        p_vals = 0.5 * np.ones(code_shape)
    rep_pvals = np.repeat(p_vals[np.newaxis, :, :], popsize, axis=0)
    return get_rand_codeset_bin01((popsize, code_shape[0], code_shape[1]), rep_pvals)


def convert_bin01_to_pm1(seq):
    # converts elements in seq from 0/1 to +1/-1
    assert chk_allbin01(seq)
    return -2 * seq + 1


def convert_pm1_to_bin01(seq):
    # converts elements in seq from +1/-1 to 0/1
    assert chk_allpm1(seq)
    return np.abs(-0.5 * (seq - 1))


def correlate_two_seq_sets(seq_set1, seq_set2):
    # will correlate along 2nd dimension (horizontal),
    # assumes sequences are horizontal and different sequences are stacked vertically:
    #     [ ---- seq 1 ----
    #       ---- seq 2 ----
    #              :
    #       ---- seq N ---- ]

    # check seq_set1 and seq_set2 must be the same size
    assert chk_samesize(
        seq_set1, seq_set2
    ), "Both sequence sets to correlate must be the same size"

    # check both sequences are +/- 1 arrays
    assert chk_allpm1(seq_set1) and chk_allpm1(
        seq_set2
    ), "All elements of input sequences must be +/- 1"

    corr = np.real(np.fft.ifft(np.fft.fft(seq_set1) * np.conj(np.fft.fft(seq_set2))))
    return corr


def naive_correlate_two_seq_sets(seq_set1, seq_set2):
    # does same thing as correlate_two_seq_sets, but in naive manner (non-FFT)

    # check seq_set1 and seq_set2 must be the same size
    assert chk_samesize(
        seq_set1, seq_set2
    ), "Both sequence sets to correlate must be the same size"

    # check both sequences are +/- 1 arrays
    assert chk_allpm1(seq_set1) and chk_allpm1(
        seq_set2
    ), "All elements of input sequences must be +/- 1"

    n_seq = seq_set1.shape[0]
    n_codelength = seq_set1.shape[1]
    corr_result = np.zeros((n_seq, n_codelength))
    for i_seq in range(n_seq):
        for i_code in range(n_codelength):
            corr_result[i_seq, i_code] = np.correlate(
                seq_set1[i_seq, :], np.roll(seq_set2[i_seq, :], i_code), mode="valid"
            )
    return corr_result


def get_pairs_info(ncodes):
    pairs = list(itertools.combinations(range(ncodes), 2))
    npairs = len(pairs)
    return pairs, npairs


def naive_getpairs_info(ncodes):
    pairs = []
    for i in range(ncodes):
        j_range = range((i + 1), ncodes)
        for j in j_range:
            pairs.append((i, j))
    npairs = int((ncodes) * (ncodes - 1) / 2)
    return pairs, npairs


def get_auto_cross_corr_summary(
    codeset,
    popsize,
    ncodes,
    nbits,
    pairs,
    npairs,
    auto_corr_summ_fcn,
    cross_corr_summ_fcn=None,
):
    # this function returns a "summary" of the auto- and cross-correlation performance for a population of binary codes
    #
    # codeset -- must be +/- 1 binary sequence
    # corr_summ_fcn "summarizes" the results from a particular correlation
    #      (considers only sidelobes of an auto or cross-correlation)
    #      e.g. take mean abs sidelobes in this correlation, or take max abs, or mean sqr
    #      corr_summ_fcn takes in correlation result of [popsize, dim_sidelobes] and outputs a scalar value
    # Note: can have separate summary functions (if only 1 provided, uses same one for cross-corr)
    #
    # Returns auto_comp and cross_comp
    #   auto_comp is [popsize x ncodes] -- gives summary for each population element, for each code sequence
    #   cross_comp is [popsize x npairs] -- gives summary for each population element, for each pair of codes

    # if no cross-corr summary fcn provided, use the same function as the auto-corr one
    if cross_corr_summ_fcn is None:
        cross_corr_summ_fcn = auto_corr_summ_fcn

    # Check that codeset is of size [popsize, ncodes, nbits] and of values 0/1
    assert chk_shape(codeset, (popsize, ncodes, nbits))
    assert chk_allpm1(codeset)

    # save auto/cross corr components
    cross_comp = np.zeros((popsize, npairs))
    auto_comp = np.zeros((popsize, ncodes))

    # Compute cross-correlation component
    for i_pair, cur_pair in enumerate(pairs):
        first_seqs = codeset[:, cur_pair[0], :]
        second_seqs = codeset[:, cur_pair[1], :]

        corr = correlate_two_seq_sets(first_seqs, second_seqs)
        cross_comp[:, i_pair] = cross_corr_summ_fcn(corr)

    # Compute cross-correlation component
    for i_seq in range(ncodes):
        cur_seqs = codeset[:, i_seq, :]
        corr = correlate_two_seq_sets(cur_seqs, cur_seqs)
        corr_nozerolag = corr[:, 1:]
        auto_comp[:, i_seq] = auto_corr_summ_fcn(corr_nozerolag)

    return auto_comp, cross_comp


def get_all_auto_cross_correlations_singlecodeset(
    codeset, ncodes, nbits, pairs, npairs
):
    # this function returns a "summary" of the auto- and cross-correlation performance for a population of binary codes
    #
    # codeset -- must be +/- 1 binary sequence (just 1 code element)
    #
    # Returns auto_comp and cross_comp
    #   auto_corr_vals is [ncodes x (nbits-1)] -- gives all auto-correlation values, for all code sequences
    #   cross_corr_vals is [npairs x nbits] -- gives all cross-correlation values, for all pairs of codes

    # Check that codeset is of size [popsize, ncodes, nbits] and of values 0/1
    assert chk_shape(codeset, (ncodes, nbits))
    assert chk_allpm1(codeset)

    # save auto/cross corr components
    auto_corr_vals = np.zeros((ncodes, nbits - 1))
    cross_corr_vals = np.zeros((npairs, nbits))

    # Compute cross-correlation component
    for i_pair, cur_pair in enumerate(pairs):
        first_seqs = codeset[cur_pair[0], :]
        second_seqs = codeset[cur_pair[1], :]
        cross_corr_vals[i_pair, :] = correlate_two_seq_sets(first_seqs, second_seqs)

    # Compute cross-correlation component
    for i_seq in range(ncodes):
        cur_seqs = codeset[i_seq, :]
        corr = correlate_two_seq_sets(cur_seqs, cur_seqs)
        auto_corr_vals[i_seq, :] = corr[1:]

    return auto_corr_vals, cross_corr_vals


def mean_abs_corr_power(corr_results, powval):
    assert powval >= 1, "power value must be >=1, but is currently: " + str(powval)
    return np.mean(np.power(np.absolute(corr_results), powval), axis=1)


def mean_sqr_corr(corr_results):
    return np.mean(corr_results * corr_results, axis=1)


def max_abs_corr(corr_results):
    return np.max(np.abs(corr_results), axis=1)


def naive_get_auto_cross_corr_summary(
    codeset,
    popsize,
    ncodes,
    nbits,
    pairs,
    npairs,
    auto_corr_summ_fcn,
    cross_corr_summ_fcn=None,
):
    # if no cross-corr summary fcn provided, use the same function as the auto-corr one
    if cross_corr_summ_fcn is None:
        cross_corr_summ_fcn = auto_corr_summ_fcn

    # Check that codeset is of size [popsize, ncodes, nbits] and of values 0/1
    assert chk_shape(codeset, (popsize, ncodes, nbits))
    assert chk_allpm1(codeset)

    # create output arrays
    test_auto_summ = np.zeros([popsize, ncodes])
    test_cross_summ = np.zeros([popsize, npairs])
    for i in range(popsize):
        auto_corr = correlate_two_seq_sets(codeset[i, :, :], codeset[i, :, :])
        test_auto_summ_i = auto_corr_summ_fcn(auto_corr[:, 1:])
        test_auto_summ[i, :] = test_auto_summ_i
        for j, cur_pair in enumerate(pairs):
            cross_corr = correlate_two_seq_sets(
                codeset[i, cur_pair[0], :], codeset[i, cur_pair[1], :]
            )
            cross_corr = cross_corr.reshape([1, -1])
            test_cross_summ_i = cross_corr_summ_fcn(cross_corr)
            test_cross_summ[i, j] = test_cross_summ_i
    return test_auto_summ, test_cross_summ


#############################################     EVALUATE/EXTRACT INFO     #############################################


def evaluate_current_codes(
    i, pop, popsize, nelite, ncodes, nbits, pairs, npairs, objective_fcn
):
    # compute objective (keep auto / cross correlation parts)
    obj, auto_comp, cross_comp = objective_fcn(
        convert_bin01_to_pm1(pop), popsize, ncodes, nbits, pairs, npairs
    )

    # get the elite population
    sorted_idc = np.argsort(obj)
    # top_perf_pop = pop[sorted_idc[0:nelite], :, :]

    best_obj = round(obj[sorted_idc[0]], 1)
    worst_obj = round(obj[sorted_idc[-1]], 1)
    print(
        "Iteration "
        + str(i)
        + " finished. Best/worst objective perf: "
        + str(best_obj)
        + ", "
        + str(worst_obj)
    )


def print_and_add_line_to_log(print_str, log_path=None):
    print(print_str)
    if log_path is not None:
        with open(log_path, "a") as log_file:
            log_file.writelines(print_str + "\n")


def extract_and_plot_run_info(run_path, show_plot=False):
    run_params_pkl_path = run_path + "/run_params.pkl"
    final_results_pkl_path = run_path + "/final_results.pkl"
    with open(run_params_pkl_path, "rb") as rp_f:
        run_params_dict = pickle.load(rp_f)
    with open(final_results_pkl_path, "rb") as fr_f:
        final_results_dict = pickle.load(fr_f)

    # extract from init and results params dictionary
    # params include: "nbits", "ncodes", "popsize", "objective_fcn", "nelite", "alpha", "objective_name",
    #                 "max_iter", "save_iter", "plot_iter"
    ni = SimpleNamespace(**run_params_dict)

    # params include: "tot_time_sec", "tot_iter", "final_pop", "final_pvalues", "best_obj", "best_obj_autocomp",
    #                 "best_obj_crosscomp", "plot_idx", "iter_arr", "obj_all", "obj_perf_bestpt", "auto_all",
    #                 "auto_perf_bestpt", "auto_perf_bestauto", "auto_perf_bestcross", "cross_all",
    #                 "cross_perf_bestpt", "cross_perf_bestcross", "cross_perf_bestauto", "rand_bin01", "converged"
    nr = SimpleNamespace(**final_results_dict)

    pairs, npairs = get_pairs_info(ni.ncodes)

    # print info from run:
    print_run_info(
        ni.nbits,
        ni.ncodes,
        ni.popsize,
        ni.nelite,
        ni.alpha,
        ni.max_iter,
        ni.objective_fcn,
        ni.objective_name,
    )

    # go through intermediate points
    if ni.save_iter is not None:
        for i in range(0, nr.tot_iter, ni.save_iter):
            interm_results_pkl_path = (
                run_path + "/intermediate_results_iter" + str(i) + ".pkl"
            )
            with open(interm_results_pkl_path, "rb") as ir_f:
                interm_results_dict = pickle.load(ir_f)

            # params include: ["curr_pop", "best_code", "pvalues"]
            evaluate_current_codes(
                i,
                interm_results_dict["curr_pop"],
                ni.popsize,
                ni.nelite,
                ni.ncodes,
                ni.nbits,
                pairs,
                npairs,
                ni.objective_fcn,
            )
            if i != 0:
                plot_final_histogram(
                    nr.rand_bin01,
                    np.round(interm_results_dict["pvalues"]),
                    False,
                    ni.ncodes,
                    ni.nbits,
                )

    # print and plot final results
    print_and_plot_obj_auto_cross_performance(
        nr.tot_time_sec,
        nr.tot_iter,
        nr.best_obj,
        nr.best_obj_autocomp,
        nr.best_obj_crosscomp,
        nr.plot_idx,
        nr.iter_arr,
        nr.obj_all,
        nr.obj_perf_bestpt,
        nr.auto_all,
        nr.auto_perf_bestpt,
        nr.auto_perf_bestauto,
        nr.auto_perf_bestcross,
        nr.cross_all,
        nr.cross_perf_bestpt,
        nr.cross_perf_bestcross,
        nr.cross_perf_bestauto,
        ni.objective_name,
        show_plot=show_plot,
    )

    # print final histograms
    plot_final_histogram(
        nr.rand_bin01,
        nr.final_pvalues,
        nr.converged,
        ni.ncodes,
        ni.nbits,
        show_plot=show_plot,
    )


def extract_info_from_prev_run(run_path):
    run_params_pkl_path = run_path + "/run_params.pkl"
    assert os.path.exists(run_path), "specified path (" + run_path + ") does not exist"
    assert os.path.exists(run_params_pkl_path), (
        "specified folder ("
        + run_path
        + ") does not have a run_params.pkl file (should exist, with initialization parameters)"
    )

    with open(run_params_pkl_path, "rb") as rp_f:
        run_params_dict = pickle.load(rp_f)

    # extract from init and results params dictionary
    # params include: "nbits", "ncodes", "popsize", "objective_fcn", "nelite", "alpha", "objective_name",
    #                 "max_iter", "save_iter", "plot_iter"
    ni = SimpleNamespace(**run_params_dict)

    # go through intermediate points
    assert (
        ni.save_iter is not None
    ), "Cannot extract run info (save_iter for this run is None)"

    found_next_iter_info = True
    i = 0
    assert os.path.exists(run_path + "/intermediate_results_iter0.pkl"), (
        "Path does not have intermediate results for iteration 0. "
        + "\n"
        + "Cannot extract run information from this path."
    )
    while found_next_iter_info:
        i += ni.save_iter
        interm_results_pkl_path = (
            run_path + "/intermediate_results_iter" + str(i) + ".pkl"
        )
        found_next_iter_info = os.path.exists(interm_results_pkl_path)

    # go back to the last i where path existed
    i = i - ni.save_iter
    interm_results_pkl_path = run_path + "/intermediate_results_iter" + str(i) + ".pkl"
    with open(interm_results_pkl_path, "rb") as ir_f:
        interm_results_dict = pickle.load(ir_f)

    # get current p-values and current population
    cur_pop = interm_results_dict["curr_pop"]
    cur_pval = interm_results_dict["pvalues"]

    # specify current iteration (whatever the last performed iteration was, plus 1)
    cur_iter = i + 1

    # return iteration, current population, current pvalues
    return cur_iter, cur_pop, cur_pval


def print_correlation_properties(final_codeset_pm1, ncodes, nbits):
    assert chk_allpm1(final_codeset_pm1), "Codeset should be all plus-or-minus 1"
    assert chk_shape(final_codeset_pm1, (ncodes, nbits))
    pairs, npairs = get_pairs_info(ncodes)
    final_codeset_3d = final_codeset_pm1[np.newaxis, :, :]

    # max average absolute corr
    obj, auto_comp, cross_comp = objectives.objective_max_meancorrpower_auto_cross(
        (final_codeset_3d), 1, ncodes, nbits, pairs, npairs, 1
    )
    print(
        "   max mean abs overall (auto, cross): "
        + str(round(obj[0], 2))
        + " ("
        + str(round(auto_comp[0], 2))
        + ", "
        + str(round(cross_comp[0], 2))
        + ")"
    )
    # max 2-norm performance
    obj, auto_comp, cross_comp = objectives.objective_max_ms_auto_cross(
        (final_codeset_3d), 1, ncodes, nbits, pairs, npairs
    )
    print(
        "   max 2-norm overall (auto, cross): "
        + str(round(obj[0] ** 0.5, 2))
        + " ("
        + str(round(auto_comp[0] ** 0.5, 2))
        + ", "
        + str(round(cross_comp[0] ** 0.5, 2))
        + ")"
    )
    # max 4-norm performance
    obj, auto_comp, cross_comp = objectives.objective_max_meancorrpower_auto_cross_4(
        (final_codeset_3d), 1, ncodes, nbits, pairs, npairs
    )
    print(
        "   max 4-norm overall (auto, cross): "
        + str(round(obj[0] ** 0.25, 2))
        + " ("
        + str(round(auto_comp[0] ** 0.25, 2))
        + ", "
        + str(round(cross_comp[0] ** 0.25, 2))
        + ")"
    )
    # max 6-norm performance
    obj, auto_comp, cross_comp = objectives.objective_max_meancorrpower_auto_cross_6(
        (final_codeset_3d), 1, ncodes, nbits, pairs, npairs
    )
    print(
        "   max 6-norm overall (auto, cross): "
        + str(round(obj[0] ** (1.0 / 6.0), 2))
        + " ("
        + str(round(auto_comp[0] ** (1.0 / 6.0), 2))
        + ", "
        + str(round(cross_comp[0] ** (1.0 / 6.0), 2))
        + ")"
    )
    # max 8-norm performance
    obj, auto_comp, cross_comp = objectives.objective_max_meancorrpower_auto_cross_8(
        (final_codeset_3d), 1, ncodes, nbits, pairs, npairs
    )
    print(
        "   max 8-norm overall (auto, cross): "
        + str(round(obj[0] ** (1.0 / 8.0), 2))
        + " ("
        + str(round(auto_comp[0] ** (1.0 / 8.0), 2))
        + ", "
        + str(round(cross_comp[0] ** (1.0 / 8.0), 2))
        + ")"
    )
    # max performance
    obj, auto_comp, cross_comp = objectives.objective_max_peak_auto_cross(
        (final_codeset_3d), 1, ncodes, nbits, pairs, npairs
    )
    print(
        "   max peak overall (auto, cross): "
        + str(round(obj[0]))
        + " ("
        + str(round(auto_comp[0]))
        + ", "
        + str(round(cross_comp[0]))
        + ")"
    )


def extract_and_plot_corr_diffobjectives(
    run_path_list,
    list_labels,
    list_colors,
    list_rwidths,
    nbits,
    ncodes,
    popsize,
    nelite,
    alpha,
    include_rand=True,
    save_path=None,
    show_plot=False,
):
    pairs, npairs = get_pairs_info(ncodes)
    len_list = len(run_path_list)
    if include_rand:
        assert len_list + 1 == len(list_labels), (
            "length of run_path_list + 1 and length of list_labels must be the same (because including rand). run_path_list is length of "
            + str(len_list)
            + ", while list_labels is length of "
            + str(len(list_labels) + " but should be: " + str(len_list + 1))
        )
        assert len_list + 1 == len(list_colors), (
            "length of run_path_list + 1 and length of list_colors must be the same (because including rand). run_path_list is length of "
            + str(len_list)
            + ", while list_colors is length of "
            + str(len(list_colors) + " but should be: " + str(len_list + 1))
        )
        assert len_list + 1 == len(list_rwidths), (
            "length of run_path_list + 1 and length of list_rwidths must be the same (because including rand). run_path_list is length of "
            + str(len_list)
            + ", while list_rwidths is length of "
            + str(len(list_rwidths) + " but should be: " + str(len_list + 1))
        )
    else:
        assert (
            len_list > 0
        ), "length of run_path_list must be greater than zero, but is: " + str(len_list)
        assert len_list + 1 == len(list_labels), (
            "length of run_path_list and list_labels must be the same. run_path_list is length of "
            + str(len_list)
            + ", while list_labels is length of "
            + str(len(list_labels))
        )
        assert len_list + 1 == len(list_colors), (
            "length of run_path_list and list_colors must be the same. run_path_list is length of "
            + str(len_list)
            + ", while list_colors is length of "
            + str(len(list_colors))
        )
        assert len_list + 1 == len(list_rwidths), (
            "length of run_path_list and list_rwidths must be the same. run_path_list is length of "
            + str(len_list)
            + ", while list_rwidths is length of "
            + str(len(list_rwidths))
        )

    list_codesets = [None] * len_list
    # abs_cross_corrs_list = [None]*len_list

    for i, run_path in enumerate(run_path_list):
        run_params_pkl_path = run_path + "/run_params.pkl"
        final_results_pkl_path = run_path + "/final_results.pkl"
        with open(run_params_pkl_path, "rb") as rp_f:
            # params include: "nbits", "ncodes", "popsize", "objective_fcn", "nelite", "alpha", "objective_name",
            #                 "max_iter", "save_iter", "plot_iter"
            run_params_dict = pickle.load(rp_f)
            # check that ncodes, nbits, popsize, nelite, alpha are same
            assert (
                run_params_dict["nbits"] == nbits
            ), "nbits differs from specified argument: " + str(nbits)
            assert (
                run_params_dict["ncodes"] == ncodes
            ), "ncodes differs from specified argument: " + str(ncodes)
            assert (
                run_params_dict["popsize"] == popsize
            ), "popsize differs from specified argument: " + str(popsize)
            assert (
                run_params_dict["nelite"] == nelite
            ), "nelite differs from specified argument: " + str(nelite)
            assert (
                run_params_dict["alpha"] == alpha
            ), "alpha differs from specified argument: " + str(alpha)
        with open(final_results_pkl_path, "rb") as fr_f:
            # params include: "tot_time_sec", "tot_iter", "final_pop", "final_pvalues", "best_obj", "best_obj_autocomp",
            #                 "best_obj_crosscomp", "plot_idx", "iter_arr", "obj_all", "obj_perf_bestpt", "auto_all",
            #                 "auto_perf_bestpt", "auto_perf_bestauto", "auto_perf_bestcross", "cross_all",
            #                 "cross_perf_bestpt", "cross_perf_bestcross", "cross_perf_bestauto", "rand_bin01", "converged"
            final_results_dict = pickle.load(fr_f)

            # evaluate current codes

            final_codeset = np.round(final_results_dict["final_pvalues"])
            list_codesets[i] = final_codeset
            # bin_codes_pm1 = convert_bin01_to_pm1(final_codeset)
            # final_codeset_3d = final_codeset[np.newaxis, :, :]

            # get absolute correlations:
            # auto_corrs, cross_corrs = get_all_auto_cross_correlations_singlecodeset(bin_codes_pm1, ncodes, nbits, pairs, npairs)
            # abs_auto_corrs_list[i] = np.round(np.abs(auto_corrs))
            # abs_cross_corrs_list[i] = np.round(np.abs(cross_corrs))

            # print info from run:
            if include_rand:
                print("Objective: " + list_labels[i + 1])
            else:
                print("Objective: " + list_labels[i])
            print("   total iter: " + str(final_results_dict["tot_iter"]))

            print_correlation_properties(
                convert_bin01_to_pm1(final_codeset), ncodes, nbits
            )

            print("   converged? " + str(final_results_dict["converged"]))
            rand_bin01 = final_results_dict["rand_bin01"]

    # if adding random code set, do so at beginning
    if include_rand:
        # rand_bin_codes_pm1 = convert_bin01_to_pm1( rand_bin01 )
        # auto_corrs, cross_corrs = get_all_auto_cross_correlations_singlecodeset(rand_bin_codes_pm1, ncodes, nbits, pairs, npairs)
        # abs_auto_corrs_list.insert(0, np.round(np.abs(auto_corrs)))
        # abs_cross_corrs_list.insert(0, np.round(np.abs(cross_corrs)))
        list_codesets.insert(0, rand_bin01)
        len_list = len(list_labels)
        print("Random codes ")
        print_correlation_properties(convert_bin01_to_pm1(rand_bin01), ncodes, nbits)

    plot_absolute_correlation_histogram_rwidths(
        list_codesets,
        list_labels,
        list_colors,
        ncodes,
        nbits,
        list_rwidths,
        save_path=save_path,
        show_plot=show_plot,
    )


#########################################################################################################################


##################################################     SAVING INFO     ##################################################
def create_and_save_params_info(
    nbits,
    ncodes,
    popsize,
    objective_fcn,
    nelite,
    alpha,
    save_path,
    objective_name,
    max_iter,
    save_iter,
    plot_iter,
    init_pop_path,
):
    # create init params dictionary
    run_info_dict = _create_run_params_dict(
        nbits,
        ncodes,
        popsize,
        objective_fcn,
        nelite,
        alpha,
        objective_name,
        max_iter,
        save_iter,
        plot_iter,
        init_pop_path,
    )
    # Note: old versions did not save init_pop_path

    # create a binary pickle file, write the python object (dict), and close
    try:
        f = open(save_path + "/run_params.pkl", "wb")
        pickle.dump(run_info_dict, f)
    finally:
        f.close()

    return run_info_dict


def _create_run_params_dict(
    nbits,
    ncodes,
    popsize,
    objective_fcn,
    nelite,
    alpha,
    objective_name,
    max_iter,
    save_iter,
    plot_iter,
    init_pop_path,
):
    run_info_dict = {}
    run_info_var_names = [
        "nbits",
        "ncodes",
        "popsize",
        "objective_fcn",
        "nelite",
        "alpha",
        "objective_name",
        "max_iter",
        "save_iter",
        "plot_iter",
        "init_pop_path",
    ]
    # Note: old versions did not save init_pop_path
    for var_str in run_info_var_names:
        run_info_dict[var_str] = eval(var_str)
    return run_info_dict


def save_intermediate_results(save_path, curr_pop, best_code, pvalues, iteridx):
    intermediate_results_dict = {}
    intermediate_results_var_names = ["curr_pop", "best_code", "pvalues"]
    for var_str in intermediate_results_var_names:
        intermediate_results_dict[var_str] = eval(var_str)
    # create a binary pickle file, write the python object (dict), and close
    try:
        f = open(save_path + "/intermediate_results_iter" + str(iteridx) + ".pkl", "wb")
        pickle.dump(intermediate_results_dict, f)
    finally:
        f.close()


def get_pop_pvalues_from_interm_results(init_pop_path):
    with open(init_pop_path, "rb") as ir_f:
        interm_results_dict = pickle.load(ir_f)

    curr_pop = interm_results_dict["curr_pop"]
    cur_pval = interm_results_dict["pvalues"]
    return curr_pop, cur_pval


def get_save_path_with_subfolders(
    base_save_folder, nbits, ncodes, popsize, objective_fcn, nelite, alpha, runidx=None
):
    save_path = (
        base_save_folder
        + "/"
        + str(nbits)
        + "bits"
        + "/"
        + str(ncodes)
        + "codes"
        + "/"
        + objective_fcn.__name__
        + "/"
        + str(popsize)
        + "popsize_"
        + str(nelite)
        + "nelite"
        + "/"
        + str(alpha)
        + "alpha"
    )
    if runidx is not None:
        assert (
            runidx >= 0 and type(runidx) == int
        ), "Run index needs to be a non-negative integer, but is: " + str(runidx)
        save_path += "/run" + str(runidx)
    return save_path


def create_save_path_with_subfolders(
    base_save_folder,
    nbits,
    ncodes,
    popsize,
    objective_fcn,
    nelite,
    alpha,
    cont_from_prev_run_folderstr=None,
):
    if base_save_folder is None:
        return None

    # if not None, append these subfolders to the path
    save_path_base = get_save_path_with_subfolders(
        base_save_folder, nbits, ncodes, popsize, objective_fcn, nelite, alpha
    )

    # if continuing from previous run, use the run string to create the save path
    if cont_from_prev_run_folderstr is not None:
        final_save_path = save_path_base + "/" + cont_from_prev_run_folderstr
        return final_save_path

    # if not continuing from previous run, create a new folder
    created_path = False
    idx = 0
    while not created_path:
        final_save_path = save_path_base + "/run" + str(idx)
        # create the path (and return path name)
        if not os.path.exists(final_save_path):
            # If it doesn't exist, create it
            os.makedirs(final_save_path)
            created_path = True
        else:
            idx += 1

    return final_save_path


def save_final_results(
    tot_time_sec,
    tot_iter,
    final_pop,
    final_pvalues,
    best_obj,
    best_obj_autocomp,
    best_obj_crosscomp,
    plot_idx,
    iter_arr,
    obj_all,
    obj_perf_bestpt,
    auto_all,
    auto_perf_bestpt,
    auto_perf_bestauto,
    auto_perf_bestcross,
    cross_all,
    cross_perf_bestpt,
    cross_perf_bestcross,
    cross_perf_bestauto,
    rand_bin01,
    converged,
    save_path,
):
    final_results_dict = {}
    final_results_var_names = [
        "tot_time_sec",
        "tot_iter",
        "final_pop",
        "final_pvalues",
        "best_obj",
        "best_obj_autocomp",
        "best_obj_crosscomp",
        "plot_idx",
        "iter_arr",
        "obj_all",
        "obj_perf_bestpt",
        "auto_all",
        "auto_perf_bestpt",
        "auto_perf_bestauto",
        "auto_perf_bestcross",
        "cross_all",
        "cross_perf_bestpt",
        "cross_perf_bestcross",
        "cross_perf_bestauto",
        "rand_bin01",
        "converged",
    ]
    for var_str in final_results_var_names:
        final_results_dict[var_str] = eval(var_str)

    # create a binary pickle file, write the python object (dict), and close
    try:
        f = open(save_path + "/final_results.pkl", "wb")
        pickle.dump(final_results_dict, f)
    finally:
        f.close()

    return final_results_dict


#############################################################################################################################


##################################################     CHECK FUNCTIONS     ##################################################
def chk_arrays_same(arr1, arr2, eps=1e-8):
    return np.sum(np.abs(arr1 - arr2)) < eps


def chk_allbin01(seq, eps=1e-8):
    # checks all elements of seq are 0 or 1
    return np.sum(np.abs(seq * (1 - seq))) < eps


def chk_allpm1(seq, eps=1e-8):
    # checks all elements of seq are -1 or +1
    return np.sum(np.abs(np.abs(seq) - 1)) < eps


def chk_all_neg1(seq, eps=1e-8):
    # checks all elements of seq are -1 or +1
    return np.sum(seq + 1) < eps


def chk_all_fromlist(seq, arr_vals, eps=1e-8):
    seq_flatten = seq.flatten()
    return all((seq_flatten[:, np.newaxis] == arr_vals).any(axis=1))


def chk_samesize(nparr1, nparr2):
    # checks two numpy arrays are the same size
    return nparr1.shape == nparr2.shape


def chk_shape(nparr, nparr_shape):
    # checks that numpy array (nparr) has expected shape (nparr_shape)
    return nparr.shape == nparr_shape


# def chk_runparams_match_cont_runparams(run_param_dict, run_param_path):

#############################################################################################################################


##################################################     PLOTTING & PRINTING     ##################################################
import numpy as np


def plot_absolute_correlation_histogram(
    list_codesets,
    list_labels,
    list_colors,
    ncodes,
    nbits,
    save_path=None,
    num_bins_auto=None,
    num_bins_cross=None,
    title_line2="",
    show_plot=False,
):
    assert len(list_codesets) == len(
        list_labels
    ), "List of code sets has to be the same length as the codeset labels."
    assert len(list_codesets) == len(
        list_colors
    ), "List of colors has to be the same length as the list of codesets."
    if num_bins_auto is not None:
        assert type(num_bins_auto) == int and num_bins_auto > 0
    if num_bins_cross is not None:
        assert type(num_bins_cross) == int and num_bins_cross > 0

    num_codesets = len(list_codesets)
    pairs, npairs = get_pairs_info(ncodes)

    list_abs_auto_corrs = [None] * num_codesets
    list_abs_cross_corrs = [None] * num_codesets
    max_auto_corrs = np.nan * np.ones([num_codesets])
    max_cross_corrs = np.nan * np.ones([num_codesets])

    # save absolute auto/cross correlations
    for i in range(num_codesets):
        bin_codes_01 = list_codesets[i]
        assert chk_allbin01(bin_codes_01), (
            "Incoming codes need to be binary 0/1 values. Element "
            + str(i)
            + " is not binary 0/1."
        )
        assert chk_shape(bin_codes_01, (ncodes, nbits)), (
            "Incoming codes need to be of shape (ncodes, nbits), "
            + "provided as input arguments. (ncodes, nbits) is: ("
            + str(ncodes)
            + ", "
            + str(nbits)
            + "), but the shape of code "
            + str(i)
            + " is: "
            + str(bin_codes_01.shape)
        )

        bin_codes_pm1 = convert_bin01_to_pm1(np.round(bin_codes_01))
        auto_corrs, cross_corrs = get_all_auto_cross_correlations_singlecodeset(
            bin_codes_pm1, ncodes, nbits, pairs, npairs
        )
        abs_auto_corrs = np.round(np.abs(auto_corrs)).reshape([-1])
        abs_cross_corrs = np.round(np.abs(cross_corrs)).reshape([-1])
        list_abs_auto_corrs[i] = abs_auto_corrs
        list_abs_cross_corrs[i] = abs_cross_corrs
        max_auto_corrs[i] = np.max(abs_auto_corrs)
        max_cross_corrs[i] = np.max(abs_cross_corrs)

    # get 2D arrays
    arr_abs_auto_corrs = np.vstack(list_abs_auto_corrs)
    arr_abs_cross_corrs = np.vstack(list_abs_cross_corrs)

    # plot auto-correlation plots
    fig = plt.figure(figsize=(15, 3))
    if num_bins_auto is None:
        overall_max_auto = np.max(max_auto_corrs)
        if nbits % 2 != 0:
            bins_arr = np.arange(1, overall_max_auto + 2, 2)
        else:
            bins_arr = np.arange(0, overall_max_auto + 2, 2)
        plt.hist(
            arr_abs_auto_corrs.T,
            bins=bins_arr,
            histtype="bar",
            color=list_colors,
            label=list_labels,
        )
    else:
        plt.hist(
            arr_abs_auto_corrs.T,
            bins=num_bins_auto,
            histtype="bar",
            color=list_colors,
            label=list_labels,
        )
    plt.title("Absolute auto-correlation histogram\n" + title_line2 + "\n")
    plt.xlabel("\n absolute correlation")
    plt.ylabel("occurences\n")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if show_plot:
        plt.show()
    # if save path is not none, save plots
    if save_path is not None:
        fig.savefig(
            save_path + "/auto_hist.svg", format="svg", dpi=1200, bbox_inches="tight"
        )
        # fig.savefig(save_path+'/auto_hist.png', format='png', dpi=1200, bbox_inches='tight')

    fig = plt.figure(figsize=(15, 3))
    if num_bins_auto is None:
        overall_max_cross = np.max(max_auto_corrs)
        if nbits % 2 != 0:
            bins_arr = np.arange(1, overall_max_cross + 2, 2)
        else:
            bins_arr = np.arange(0, overall_max_cross + 2, 2)
        plt.hist(
            arr_abs_cross_corrs.T,
            bins=bins_arr,
            histtype="bar",
            color=list_colors,
            label=list_labels,
        )
    else:
        plt.hist(
            arr_abs_cross_corrs.T,
            bins=num_bins_cross,
            histtype="bar",
            color=list_colors,
            label=list_labels,
        )
    plt.title("Absolute cross-correlation histogram\n" + title_line2 + "\n")
    plt.xlabel("\n absolute correlation")
    plt.ylabel("occurences\n")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if show_plot:
        plt.show()
    # if save path is not none, save plots
    if save_path is not None:
        fig.savefig(
            save_path + "/cross_hist.svg", format="svg", dpi=1200, bbox_inches="tight"
        )
        # fig.savefig(save_path+'/cross_hist.png', format='png', dpi=1200, bbox_inches='tight')


def plot_absolute_correlation_histogram_rwidths(
    list_codesets,
    list_labels,
    list_colors,
    ncodes,
    nbits,
    rwidths,
    save_path=None,
    num_bins_auto=None,
    num_bins_cross=None,
    title_line2="",
    show_plot=False,
):
    assert len(list_codesets) == len(
        list_labels
    ), "List of code sets has to be the same length as the codeset labels."
    assert len(list_codesets) == len(
        list_colors
    ), "List of colors has to be the same length as the list of codesets."
    if num_bins_auto is not None:
        assert type(num_bins_auto) == int and num_bins_auto > 0
    if num_bins_cross is not None:
        assert type(num_bins_cross) == int and num_bins_cross > 0

    num_codesets = len(list_codesets)
    pairs, npairs = get_pairs_info(ncodes)

    list_abs_auto_corrs = [None] * num_codesets
    list_abs_cross_corrs = [None] * num_codesets
    max_auto_corrs = np.nan * np.ones([num_codesets])
    max_cross_corrs = np.nan * np.ones([num_codesets])

    # save absolute auto/cross correlations
    for i in range(num_codesets):
        bin_codes_01 = list_codesets[i]
        assert chk_allbin01(bin_codes_01), (
            "Incoming codes need to be binary 0/1 values. Element "
            + str(i)
            + " is not binary 0/1."
        )
        assert chk_shape(bin_codes_01, (ncodes, nbits)), (
            "Incoming codes need to be of shape (ncodes, nbits), "
            + "provided as input arguments. (ncodes, nbits) is: ("
            + str(ncodes)
            + ", "
            + str(nbits)
            + "), but the shape of code "
            + str(i)
            + " is: "
            + str(bin_codes_01.shape)
        )

        bin_codes_pm1 = convert_bin01_to_pm1(np.round(bin_codes_01))
        auto_corrs, cross_corrs = get_all_auto_cross_correlations_singlecodeset(
            bin_codes_pm1, ncodes, nbits, pairs, npairs
        )
        abs_auto_corrs = np.round(np.abs(auto_corrs)).reshape([-1])
        abs_cross_corrs = np.round(np.abs(cross_corrs)).reshape([-1])
        list_abs_auto_corrs[i] = abs_auto_corrs
        list_abs_cross_corrs[i] = abs_cross_corrs
        max_auto_corrs[i] = np.max(abs_auto_corrs)
        max_cross_corrs[i] = np.max(abs_cross_corrs)

    # plot auto-correlation plots
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_codesets):
        if num_bins_auto is None:
            overall_max_auto = max_auto_corrs[i]
            if nbits % 2 != 0:
                bins_arr = np.arange(1, overall_max_auto + 2, 2)
            else:
                bins_arr = np.arange(0, overall_max_auto + 2, 2)
            plt.hist(
                list_abs_auto_corrs[i],
                bins=bins_arr,
                histtype="bar",
                color=list_colors[i],
                label=list_labels[i],
                rwidth=rwidths[i],
            )
        else:
            plt.hist(
                list_abs_auto_corrs[i],
                bins=int(num_bins_auto),
                histtype="bar",
                color=list_colors[i],
                label=list_labels[i],
                rwidth=rwidths[i],
            )
    plt.title("Absolute auto-correlation histogram\n" + title_line2 + "\n")
    plt.xlabel("\n absolute auto-correlation")
    plt.ylabel("occurences\n")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if show_plot:
        plt.show()
    # if save path is not none, save plots
    if save_path is not None:
        fig.savefig(
            save_path + "/auto_hist.svg", format="svg", dpi=1200, bbox_inches="tight"
        )

    fig = plt.figure(figsize=(15, 3))
    for i in range(num_codesets):
        if num_bins_cross is None:
            overall_max_cross = max_cross_corrs[i]
            if nbits % 2 != 0:
                bins_arr = np.arange(1, overall_max_cross + 2, 2)
            else:
                bins_arr = np.arange(0, overall_max_cross + 2, 2)
            plt.hist(
                list_abs_cross_corrs[i],
                bins=bins_arr,
                histtype="bar",
                color=list_colors[i],
                label=list_labels[i],
                rwidth=rwidths[i],
            )
        else:
            plt.hist(
                list_abs_cross_corrs[i],
                bins=int(num_bins_cross),
                histtype="bar",
                color=list_colors[i],
                label=list_labels[i],
                rwidth=rwidths[i],
            )
    plt.title("Absolute cross-correlation histogram\n" + title_line2 + "\n")
    plt.xlabel("\n absolute cross-correlation")
    plt.ylabel("occurences\n")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if show_plot:
        plt.show()

    # if save path is not none, save plots
    if save_path is not None:
        fig.savefig(
            save_path + "/cross_hist.svg", format="svg", dpi=1200, bbox_inches="tight"
        )


def print_and_plot_obj_auto_cross_performance(
    tot_time_sec,
    tot_iter,
    best_obj,
    best_obj_autocomp,
    best_obj_crosscomp,
    plot_idx,
    iter_arr,
    obj_all,
    obj_perf_bestpt,
    auto_all,
    auto_perf_bestpt,
    auto_perf_bestauto,
    auto_perf_bestcross,
    cross_all,
    cross_perf_bestpt,
    cross_perf_bestcross,
    cross_perf_bestauto,
    objective_name=None,
    save_path=None,
    show_plot=False,
    log_path=None,
):
    to_print = (
        "\n"
        + "Completed "
        + str(tot_iter)
        + " iterations! \n"
        + "Final best cost: "
        + str(best_obj)
        + "\n"
    )
    to_print += (
        "    auto-corr part: "
        + str(best_obj_autocomp)
        + "\n   cross-corr part: "
        + str(best_obj_crosscomp)
    )
    to_print += "\n" + "Total time run: " + str(tot_time_sec) + " sec" + "\n"
    print_and_add_line_to_log(to_print, log_path)
    best_perf_pt_color = "g"  # np.array([153, 160, 83])/255
    obj_pop_color = 166 * np.array([1, 1, 1]) / 255
    best_sameobjcomp_color = "c"
    best_otherobjcomp_color = "r"

    # main objective
    fig = plt.figure(1)
    plt.plot(
        iter_arr,
        obj_all[0:plot_idx, 0],
        "*",
        color=obj_pop_color,
        label="population performance",
    )
    plt.plot(iter_arr, obj_all[0:plot_idx, :], "*", color=obj_pop_color)
    plt.plot(
        iter_arr,
        obj_perf_bestpt[0:plot_idx],
        "*",
        color=best_perf_pt_color,
        label="best-performing code set",
    )
    plt.xlabel("\n iterations")
    if objective_name is not None:
        plt.ylabel("correlation cost \n (" + objective_name + ")\n")
    else:
        plt.ylabel("correlation cost\n")
    plt.title(
        "CEM objective performance\n (Total: "
        + str(tot_iter)
        + " iterations, Final best objective: "
        + str(best_obj)
        + ")\n"
    )
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if show_plot:
        plt.show()

    # if save path is not none, save plots
    if save_path is not None:
        fig.savefig(
            save_path + "/obj_perf.svg", format="svg", dpi=1200, bbox_inches="tight"
        )
        # fig.savefig(save_path+'/obj_perf.png', format='png', dpi=1200, bbox_inches='tight')

    # auto-corr performance
    fig = plt.figure(2)
    plt.plot(
        iter_arr,
        auto_all[0:plot_idx, 0],
        "*",
        color=obj_pop_color,
        label="auto-corr of all pts",
    )
    plt.plot(iter_arr, auto_all[0:plot_idx, :], "*", color=obj_pop_color)
    plt.plot(
        iter_arr,
        auto_perf_bestpt[0:plot_idx],
        "*",
        color=best_perf_pt_color,
        label="overall best-performing pt",
    )
    plt.plot(
        iter_arr,
        auto_perf_bestauto[0:plot_idx],
        "*",
        color=best_sameobjcomp_color,
        label="best auto-corr pt",
    )
    plt.plot(
        iter_arr,
        auto_perf_bestcross[0:plot_idx],
        "*",
        color=best_otherobjcomp_color,
        label="best cross-corr pt",
    )
    plt.xlabel("\n iterations")
    if objective_name is not None:
        plt.ylabel(
            "auto-correlation component \n (Overall obj: " + objective_name + ")\n"
        )
    else:
        plt.ylabel("auto-correlation component\n")
    plt.title(
        "CEM auto-correlation component performance\n (Total: "
        + str(tot_iter)
        + " iterations, Final best objective: "
        + str(best_obj)
        + ")\n"
    )
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if show_plot:
        plt.show()

    # if save path is not none, save plots
    if save_path is not None:
        fig.savefig(
            save_path + "/auto_perf.svg", format="svg", dpi=1200, bbox_inches="tight"
        )
        # fig.savefig(save_path+'/auto_perf.png', format='png', dpi=1200, bbox_inches='tight')

    # auto-corr performance
    fig = plt.figure(3)
    plt.plot(
        iter_arr,
        cross_all[0:plot_idx, 0],
        "*",
        color=obj_pop_color,
        label="cross-corr of all pts",
    )
    plt.plot(iter_arr, cross_all[0:plot_idx, :], "*", color=obj_pop_color)
    plt.plot(
        iter_arr,
        cross_perf_bestpt[0:plot_idx],
        "*",
        color=best_perf_pt_color,
        label="overall best-performing pt",
    )
    plt.plot(
        iter_arr,
        cross_perf_bestcross[0:plot_idx],
        "*",
        color=best_sameobjcomp_color,
        label="best cross-corr pt",
    )
    plt.plot(
        iter_arr,
        cross_perf_bestauto[0:plot_idx],
        "*",
        color=best_otherobjcomp_color,
        label="best auto-corr pt",
    )
    plt.xlabel("\n iterations")
    if objective_name is not None:
        plt.ylabel(
            "cross-correlation component \n (Overall obj: " + objective_name + ")\n"
        )
    else:
        plt.ylabel("cross-correlation component\n")
    plt.title(
        "CEM cross-correlation component performance\n (Total: "
        + str(tot_iter)
        + " iterations, Final best objective: "
        + str(best_obj)
        + ")\n"
    )
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    if show_plot:
        plt.show()

    # if save path is not none, save plots
    if save_path is not None:
        fig.savefig(
            save_path + "/cross_perf.svg", format="svg", dpi=1200, bbox_inches="tight"
        )
        # fig.savefig(save_path+'/cross_perf.png', format='png', dpi=1200, bbox_inches='tight')


def plot_final_histogram(
    rand_bin01,
    final_pvalues,
    converged,
    ncodes,
    nbits,
    save_path=None,
    title_line2="",
    show_plot=False,
):
    list_codesets = [final_pvalues, rand_bin01]
    if not converged:
        final_pvalues = np.round(final_pvalues)
        list_labels = ["CEM optimized (rounded pvalues)", "random"]
    else:
        list_labels = ["CEM optimized", "random"]
    list_colors = ["b", "r"]
    plot_absolute_correlation_histogram(
        list_codesets,
        list_labels,
        list_colors,
        ncodes,
        nbits,
        save_path,
        num_bins_auto=None,
        num_bins_cross=None,
        title_line2=title_line2,
        show_plot=show_plot,
    )


def print_run_info(
    nbits,
    ncodes,
    popsize,
    nelite,
    alpha,
    max_iter,
    objective_fcn,
    objective_name=None,
    log_path=None,
    init_pop_path=None,
):
    to_print = (
        "Running CEM for code families of: "
        + str(nbits)
        + " bits, "
        + str(ncodes)
        + " codes \n"
    )
    if objective_name is not None:
        to_print += (
            "Objective function: "
            + objective_fcn.__name__
            + " ("
            + objective_name
            + ")\n"
        )
    else:
        to_print += "Objective function: " + objective_fcn.__name__ + "\n"
    to_print += (
        "Population size = "
        + str(popsize)
        + "; number elite = "
        + str(nelite)
        + "; alpha = "
        + str(alpha)
        + "; maximum iterations = "
        + str(max_iter)
        + "\n"
    )
    if init_pop_path is not None:
        to_print += "\n"
        to_print += (
            "Starting from previous population located at the following intermediate results .pkl: "
            + "\n"
        )
        to_print += init_pop_path
        to_print += "\n"

    print_and_add_line_to_log(to_print, log_path)


def plot_comp_nes_ga(plot_save_path=None):
    figsize_tuple = (3, 4)
    ga_col = "b"
    nes_col = "orange"
    cem_col = "g"
    num_codes = np.array([3, 5, 7, 10, 13, 15, 18, 20, 25, 31])
    xmin_val = np.min(num_codes) - 1
    xmax_val = np.max(num_codes) + 1

    ga_1023 = np.array(
        [30.42, 31.03, 31.39, 31.5, 31.66, 31.68, 31.76, 31.76, 31.81, 31.85]
    )
    nes_1023 = np.array(
        [26.69, 28.78, 29.76, 30.51, 30.8, 31.03, 31.15, 31.26, 31.41, 31.5]
    )
    cem_1023 = np.array(
        [26.98, 28.88, 29.76, 30.42, 30.78, 30.93, 31.11, 31.19, 31.35, 31.47]
    )
    codelength = 1023
    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, ga_1023, "o", color=ga_col, label="GA")
    plt.plot(num_codes, nes_1023, "+", color=nes_col, label="NES")
    plt.plot(num_codes, cem_1023, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, ga_1023, "--", color=ga_col)
    plt.plot(num_codes, nes_1023, "--", color=nes_col)
    plt.plot(num_codes, cem_1023, "--", color=cem_col)
    plt.xlim([xmin_val, xmax_val])
    plt.hlines(
        y=np.sqrt(codelength),
        xmin=xmin_val,
        xmax=xmax_val,
        linestyles="--",
        color="gray",
        label="average for random codes",
    )
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with NES and GA \n for length-" + str(codelength) + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/comp_nes_ga_1023b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # plot NES, GA, CEM results
    ga_1031 = np.array(
        [30.56, 31.24, 31.5, 31.65, 31.77, 31.83, 31.88, 31.88, 31.93, 31.97]
    )
    nes_1031 = np.array(
        [26.79, 28.89, 29.82, 30.63, 30.97, 31.1, 31.26, 31.36, 31.53, 31.63]
    )
    cem_1031 = np.array(
        [27.02, 29.03, 29.87, 30.53, 30.89, 31.06, 31.23, 31.31, 31.47, 31.6]
    )
    codelength = 1031
    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, ga_1031, "o", color=ga_col, label="GA")
    plt.plot(num_codes, nes_1031, "+", color=nes_col, label="NES")
    plt.plot(num_codes, cem_1031, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, ga_1031, "--", color=ga_col)
    plt.plot(num_codes, nes_1031, "--", color=nes_col)
    plt.plot(num_codes, cem_1031, "--", color=cem_col)
    plt.xlim([xmin_val, xmax_val])
    plt.hlines(
        y=np.sqrt(codelength),
        xmin=xmin_val,
        xmax=xmax_val,
        linestyles="--",
        color="gray",
        label="average for random codes",
    )
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with NES and GA \n for length-" + str(codelength) + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/comp_nes_ga_1031b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )


def plot_comp_gold_weil_aug2023(plot_save_path=None):
    figsize_tuple = (3, 4)

    weil_col = np.array([0, 171, 196]) / 255
    cem_col = "g"
    gold_col = np.array([183, 158, 86]) / 255
    num_codes = np.array([3, 10, 18, 31])
    xmin_val = np.min(num_codes) - 1
    xmax_val = np.max(num_codes) + 1

    # Gold 8191 results
    codelength = 8191
    cem_8191ac = np.sqrt(np.array([6095.942, 7463.775, 7769.696, 7939.9896]))
    cem_8191cc = np.sqrt(np.array([6095.96, 7463.821, 7769.85, 7940.0411]))
    cem_8191ob = np.maximum(cem_8191ac, cem_8191cc)

    gold_8191ac = np.sqrt(np.array([8056.3416, 8149.3894, 8157.6407, 8145.192]))
    gold_8191cc = np.sqrt(np.array([8058.6919, 8149.2391, 8166.3454, 8178.6062]))
    gold_8191ob = np.maximum(gold_8191ac, gold_8191cc)
    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_8191ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_8191ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_8191cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, gold_8191ob, ".", color=gold_col, label="Gold")
    plt.plot(num_codes, gold_8191ac, ":", color=gold_col, label="Gold (auto)")
    plt.plot(num_codes, gold_8191cc, "--", color=gold_col, label="Gold (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Gold codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/gold_8191b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # Weil 10223 results
    codelength = 10223
    # cem_10223ac = np.sqrt(np.array([7695.358,8589.175,8993.721,9331.933,9522.652,9609.784,9706.604,9752.151,9843.325,9914.28,10027.4]))
    cem_10223ac = np.sqrt(np.array([7695.358, 9331.933, 9706.604, 9914.28]))
    cem_10223cc = np.sqrt(np.array([7695.191, 9332.053, 9706.753, 9914.328]))
    cem_10223ob = np.maximum(cem_10223ac, cem_10223cc)

    weil_10223ac = np.sqrt(np.array([10270.8024, 10270.8024, 10270.8024, 10270.8024]))
    weil_10223cc = np.sqrt(np.array([10160.6155, 10201.017, 10224.9246, 10215.5068]))
    weil_10223ob = np.maximum(weil_10223ac, weil_10223cc)
    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_10223ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_10223ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_10223cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, weil_10223ob, ".", color=weil_col, label="Weil")
    plt.plot(num_codes, weil_10223ac, ":", color=weil_col, label="Weil (auto)")
    plt.plot(num_codes, weil_10223cc, "--", color=weil_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Weil codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/weil_10223b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )


def plot_comp_gold_weil_sep2023(plot_save_path=None):
    figsize_tuple = (3, 4)

    weil_col = np.array([0, 171, 196]) / 255
    cem_col = "g"
    gold_col = np.array([183, 158, 86]) / 255
    num_codes = np.array([3, 5, 7, 10, 13, 15, 18, 20, 25, 31, 50])
    xmin_val = np.min(num_codes) - 1
    xmax_val = np.max(num_codes) + 1

    # Gold 1023 results
    codelength = 1023
    cem_1023ac = np.sqrt(
        np.array(
            [
                727.883,
                833.745,
                885.632,
                925.125,
                947.073,
                956.776,
                967.518,
                972.944,
                982.744,
                990.489,
                1002.762,
            ]
        )
    )
    cem_1023cc = np.sqrt(
        np.array(
            [
                728.059,
                834.017,
                885.683,
                925.453,
                947.109,
                956.891,
                967.555,
                973.022,
                982.848,
                990.546,
                1002.762,
            ]
        )
    )
    cem_1023ob = np.maximum(cem_1023ac, cem_1023cc)

    gold_1023ac = np.sqrt(
        np.array(
            [
                929.9062,
                975.7037,
                974.2938,
                993.3131,
                1001.1264,
                1001.3872,
                1002.0942,
                1001.3538,
                1003.2676,
                1006.8603,
                1011.9395,
            ]
        )
    )
    gold_1023cc = np.sqrt(
        np.array(
            [
                935.913,
                971.1474,
                984.6868,
                994.701,
                1003.2801,
                1004.8136,
                1005.9176,
                1009.2478,
                1011.3367,
                1013.5044,
                1017.0192,
            ]
        )
    )
    gold_1023ob = np.maximum(gold_1023ac, gold_1023cc)

    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_1023ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_1023ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_1023cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, gold_1023ob, ".", color=gold_col, label="Weil")
    plt.plot(num_codes, gold_1023ac, ":", color=gold_col, label="Weil (auto)")
    plt.plot(num_codes, gold_1023cc, "--", color=gold_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Gold codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/gold_" + str(codelength) + "b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # Weil 1031 results
    codelength = 1031
    cem_1031ac = np.sqrt(
        np.array(
            [
                729.834,
                842.945,
                891.747,
                932.174,
                954.345,
                964.388,
                975.083,
                980.613,
                990.516,
                998.258,
                1010.597,
            ]
        )
    )
    cem_1031cc = np.sqrt(
        np.array(
            [
                730.084,
                843.016,
                892.111,
                932.217,
                954.36,
                964.435,
                975.159,
                980.629,
                990.543,
                998.276,
                1010.61,
            ]
        )
    )
    cem_1031ob = np.maximum(cem_1031ac, cem_1031cc)

    weil_1031ac = np.sqrt(
        np.array(
            [
                977.2796,
                977.2796,
                977.2796,
                977.2796,
                977.2796,
                977.2796,
                977.2796,
                977.2796,
                977.2796,
                977.2796,
                977.2796,
            ]
        )
    )
    weil_1031cc = np.sqrt(
        np.array(
            [
                974.7653,
                996.1938,
                1001.9249,
                1014.215,
                1017.3619,
                1018.6403,
                1020.4302,
                1022.4047,
                1024.2797,
                1024.9384,
                1027.1948,
            ]
        )
    )
    weil_1031ob = np.maximum(weil_1031ac, weil_1031cc)

    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_1031ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_1031ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_1031cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, weil_1031ob, ".", color=weil_col, label="Weil")
    plt.plot(num_codes, weil_1031ac, ":", color=weil_col, label="Weil (auto)")
    plt.plot(num_codes, weil_1031cc, "--", color=weil_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Weil codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/weil_" + str(codelength) + "b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # Gold 2047 results
    codelength = 2047
    cem_2047ac = np.sqrt(
        np.array(
            [
                1464.1867,
                1676.722,
                1776.308,
                1853.3347,
                1896.737,
                1915.605,
                1936.8262,
                1947.703,
                1967.097,
                1982.1726,
                2006.55,
            ]
        )
    )
    cem_2047cc = np.sqrt(
        np.array(
            [
                1464.8528,
                1676.854,
                1776.503,
                1853.4513,
                1896.752,
                1915.698,
                1937.0051,
                1947.755,
                1967.13,
                1982.2154,
                2006.575,
            ]
        )
    )
    cem_2047ob = np.maximum(cem_2047ac, cem_2047cc)

    gold_2047ac = np.sqrt(
        np.array(
            [
                1965.5852,
                2004.5566,
                2015.5209,
                2021.3855,
                2022.657,
                2031.7665,
                2026.1026,
                2024.182,
                2038.1844,
                2039.1436,
                2038.1894,
            ]
        )
    )
    gold_2047cc = np.sqrt(
        np.array(
            [
                1974.6304,
                2007.1796,
                2017.8717,
                2025.9123,
                2032.4438,
                2030.2218,
                2035.6152,
                2032.851,
                2037.8562,
                2038.9526,
                2041.4755,
            ]
        )
    )
    gold_2047ob = np.maximum(gold_2047ac, gold_2047cc)

    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_2047ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_2047ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_2047cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, gold_2047ob, ".", color=gold_col, label="Weil")
    plt.plot(num_codes, gold_2047ac, ":", color=gold_col, label="Weil (auto)")
    plt.plot(num_codes, gold_2047cc, "--", color=gold_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Gold codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/gold_" + str(codelength) + "b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # Weil 2053 results
    codelength = 2053
    cem_2053ac = np.sqrt(
        np.array(
            [
                1473.4029,
                1686.561,
                1780.148,
                1858.6717,
                1901.79,
                1921.162,
                1942.426,
                1953.58,
                1972.719,
                1988.0767,
                2012.402,
            ]
        )
    )
    cem_2053cc = np.sqrt(
        np.array(
            [
                1473.7677,
                1686.692,
                1780.471,
                1858.8056,
                1902.218,
                1921.188,
                1942.4318,
                1953.627,
                1972.742,
                1988.0754,
                2012.418,
            ]
        )
    )
    cem_2053ob = np.maximum(cem_2053ac, cem_2053cc)

    weil_2053ac = np.sqrt(
        np.array(
            [
                2061.7251,
                2061.7251,
                2061.7251,
                2061.7251,
                2061.7257,
                2061.7251,
                2061.726,
                2061.7259,
                2061.7261,
                2061.7267,
                2061.727,
            ]
        )
    )
    weil_2053cc = np.sqrt(
        np.array(
            [
                2016.196,
                2061.5592,
                2060.4429,
                2059.296,
                2046.6429,
                2056.9883,
                2056.7358,
                2056.7369,
                2051.152,
                2052.2813,
                2049.1576,
            ]
        )
    )
    weil_2053ob = np.maximum(weil_2053ac, weil_2053cc)

    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_2053ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_2053ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_2053cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, weil_2053ob, ".", color=weil_col, label="Weil")
    plt.plot(num_codes, weil_2053ac, ":", color=weil_col, label="Weil (auto)")
    plt.plot(num_codes, weil_2053cc, "--", color=weil_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Weil codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/weil_" + str(codelength) + "b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # Weil 4099 results
    codelength = 4099
    cem_4099ac = np.sqrt(
        np.array(
            [
                2981.413,
                3378.181,
                3571.195,
                3717.809,
                3802.546,
                3840.178,
                3881.818,
                3902.333,
                3940.844,
                3970.515,
                4018.607,
            ]
        )
    )
    cem_4099cc = np.sqrt(
        np.array(
            [
                2981.438,
                3378.333,
                3571.394,
                3717.906,
                3802.687,
                3840.326,
                3881.837,
                3902.566,
                3940.845,
                3970.592,
                4018.616,
            ]
        )
    )
    cem_4099ob = np.maximum(cem_4099ac, cem_4099cc)

    weil_4099ac = np.sqrt(
        np.array(
            [
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
                4011.7916,
            ]
        )
    )
    weil_4099cc = np.sqrt(
        np.array(
            [
                4005.9983,
                4022.2989,
                4049.6494,
                4062.1412,
                4070.1913,
                4073.3662,
                4080.0611,
                4078.5595,
                4085.1795,
                4086.3906,
                4090.8476,
            ]
        )
    )
    weil_4099ob = np.maximum(weil_4099ac, weil_4099cc)

    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_4099ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_4099ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_4099cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, weil_4099ob, ".", color=weil_col, label="Weil")
    plt.plot(num_codes, weil_4099ac, ":", color=weil_col, label="Weil (auto)")
    plt.plot(num_codes, weil_4099cc, "--", color=weil_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Weil codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/weil_" + str(codelength) + "b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # Gold 8191 results
    codelength = 8191
    cem_8191ac = np.sqrt(
        np.array(
            [
                6095.942,
                6836.619,
                7181.13,
                7463.775,
                7621.58,
                7690.463,
                7769.696,
                7810.60,
                7883.234,
                7939.9896,
                8032.743,
            ]
        )
    )
    cem_8191cc = np.sqrt(
        np.array(
            [
                6095.96,
                6836.81,
                7181.427,
                7463.821,
                7621.574,
                7690.561,
                7769.85,
                7810.624,
                7883.249,
                7940.0411,
                8032.747,
            ]
        )
    )
    cem_8191ob = np.maximum(cem_8191ac, cem_8191cc)

    weil_8191ac = np.sqrt(
        np.array(
            [
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
                8355.0396,
            ]
        )
    )
    weil_8191cc = np.sqrt(
        np.array(
            [
                8175.2426,
                8192.7781,
                8199.8104,
                8175.652,
                8178.346,
                8186.1477,
                8186.8362,
                8192.1611,
                8193.0089,
                8186.7185,
                8188.9932,
            ]
        )
    )
    weil_8191ob = np.maximum(weil_8191ac, weil_8191cc)
    gold_8191ac = np.sqrt(
        np.array(
            [
                8056.3416,
                8106.979,
                8120.7056,
                8149.3894,
                8133.4662,
                8163.5095,
                8157.6407,
                8156.6005,
                8158.8292,
                8145.192,
                8172.4362,
            ]
        )
    )
    gold_8191cc = np.sqrt(
        np.array(
            [
                8058.6919,
                8113.1903,
                8120.381,
                8149.2391,
                8158.8356,
                8164.1322,
                8166.3454,
                8167.5326,
                8169.8347,
                8178.6062,
                8181.1051,
            ]
        )
    )
    gold_8191ob = np.maximum(gold_8191ac, gold_8191cc)
    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_8191ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_8191ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_8191cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, gold_8191ob, ".", color=gold_col, label="Gold")
    plt.plot(num_codes, gold_8191ac, ":", color=gold_col, label="Gold (auto)")
    plt.plot(num_codes, gold_8191cc, "--", color=gold_col, label="Gold (cross)")
    plt.plot(num_codes, weil_8191ob, ".", color=weil_col, label="Weil")
    plt.plot(num_codes, weil_8191ac, ":", color=weil_col, label="Weil (auto)")
    plt.plot(num_codes, weil_8191cc, "--", color=weil_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Gold & Weil codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/gold_weil_8191b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )

    # Weil 10223 results
    codelength = 10223
    cem_10223ac = np.sqrt(
        np.array(
            [
                7695.358,
                8589.175,
                8993.721,
                9331.933,
                9522.652,
                9609.784,
                9706.604,
                9752.151,
                9843.325,
                9914.28,
                10027.4,
            ]
        )
    )
    # cem_10223ac = np.sqrt(np.array([7695.358,9331.933,9706.604,9914.28]))
    cem_10223cc = np.sqrt(
        np.array(
            [
                7695.191,
                8589.236,
                8993.788,
                9332.053,
                9522.71,
                9609.901,
                9706.753,
                9752.209,
                9843.341,
                9914.328,
                10027.4,
            ]
        )
    )
    # cem_10223cc = np.sqrt(np.array([7695.191,9332.053,9706.753,9914.328]))
    cem_10223ob = np.maximum(cem_10223ac, cem_10223cc)

    weil_10223ac = np.sqrt(
        np.array(
            [
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
                10270.8024,
            ]
        )
    )
    weil_10223cc = np.sqrt(
        np.array(
            [
                10160.6155,
                10263.8235,
                10270.8024,
                10201.017,
                10233.3964,
                10216.0319,
                10224.9246,
                10230.3955,
                10222.1853,
                10225.0385,
                10224.3052,
            ]
        )
    )
    weil_10223ob = np.maximum(weil_10223ac, weil_10223cc)
    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(num_codes, cem_10223ob, "*", color=cem_col, label="CEM")
    plt.plot(num_codes, cem_10223ac, ":", color=cem_col, label="CEM (auto)")
    plt.plot(num_codes, cem_10223cc, "--", color=cem_col, label="CEM (cross)")
    plt.plot(num_codes, weil_10223ob, ".", color=weil_col, label="Weil")
    plt.plot(num_codes, weil_10223ac, ":", color=weil_col, label="Weil (auto)")
    plt.plot(num_codes, weil_10223cc, "--", color=weil_col, label="Weil (cross)")
    plt.xlim([xmin_val, xmax_val])
    # plt.hlines(y=np.sqrt(codelength), xmin=xmin_val, xmax=xmax_val, linestyles='--', color='gray', label='average for random codes')
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with Weil codes (best of 10,000 random families) \n for length-"
        + str(codelength)
        + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/weil_10223b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )


def plot_comp_nes(plot_save_path=None):
    figsize_tuple = (3, 3)
    # ga_col = 'b'
    nes_col = "b"
    # cem_col ='g'
    num_codes = np.array([3, 5, 7, 10, 13, 15, 18, 20, 25, 31])
    xmin_val = np.min(num_codes) - 1
    xmax_val = np.max(num_codes) + 1

    nes_1023 = np.array(
        [26.69, 28.78, 29.76, 30.51, 30.8, 31.03, 31.15, 31.26, 31.41, 31.5]
    )
    codelength = 1023
    fig = plt.figure(figsize=figsize_tuple)
    # plt.plot(num_codes, ga_1023, 'o', color=ga_col, label='GA')
    plt.plot(num_codes, nes_1023**2, "+", color=nes_col, label="NES")
    # plt.plot(num_codes, cem_1023, '*', color=cem_col, label='CEM')
    # plt.plot(num_codes, ga_1023, '--', color=ga_col)
    plt.plot(num_codes, nes_1023**2, "--", color=nes_col)
    # plt.plot(num_codes, cem_1023, '--', color=cem_col)
    plt.xlim([xmin_val, xmax_val])
    plt.hlines(
        y=(codelength),
        xmin=xmin_val,
        xmax=xmax_val,
        linestyles="--",
        color="orange",
        label="average for random codes",
    )
    plt.xlabel("\n number of codes")
    plt.ylabel(
        "maximum norm auto- and cross-corr (MNAC) \n with $p=2$ (Euclidean norm)  \n"
    )
    plt.title(
        "Comparison with NES and GA \n for length-" + str(codelength) + " codes \n"
    )
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/comp_nes_1023b.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )


def plot_welch_bound_wrt_familysize(
    codelength=1023,
    familysize_max=1025,
    code_fam_size_vlines=None,
    code_fam_size_vline_labels=None,
    plot_save_path=None,
):
    figsize_tuple = (3, 4)

    # get array of different family sizes to plot for
    famsize_step = 0.5
    famsize_arr = np.arange(2, familysize_max + famsize_step, famsize_step)
    welch_bd_arr = (
        np.sqrt(codelength)
        * np.sqrt(
            (codelength * famsize_arr - codelength) / (famsize_arr * codelength - 1)
        )
        / codelength
    )
    npairs = (famsize_arr) * (famsize_arr - 1) / 2
    # otherbd_ave = np.sqrt(famsize_arr*famsize_arr*(codelength**3)/(codelength*npairs + (codelength-1)*famsize_arr))
    ymin_val = np.min(welch_bd_arr) - 1
    ymax_val = np.max(welch_bd_arr) + 1

    # Gold 8191 results
    fig = plt.figure(figsize=figsize_tuple)
    plt.plot(famsize_arr, welch_bd_arr, label="Welch bound ")
    # plt.plot(famsize_arr, otherbd_ave, label='Bound from Xia et al.')
    # plt.xlim([xmin_val, xmax_val])
    if code_fam_size_vlines is not None:
        for i, curr_vline in enumerate(code_fam_size_vlines):
            curr_lbl = code_fam_size_vline_labels[i]
            if curr_lbl is not None:
                plt.vlines(
                    x=curr_vline,
                    ymin=ymin_val,
                    ymax=ymax_val,
                    linestyles="--",
                    color="gray",
                    label=curr_lbl,
                )
            else:
                plt.vlines(
                    x=curr_vline,
                    ymin=ymin_val,
                    ymax=ymax_val,
                    linestyles="--",
                    color="gray",
                )
    plt.xlabel("\n code family size")
    plt.ylabel("Welch bound (on maximum peak correlation level)  \n")
    plt.title("Welch bound for families of length-" + str(codelength) + " codes \n")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.grid()
    plt.show()
    if plot_save_path is not None:
        fig.savefig(
            plot_save_path + "/welch_bound.svg",
            format="svg",
            dpi=1200,
            bbox_inches="tight",
        )


def main_plot_welch_bound_wrt_familysize():
    codelength = 10223
    familysize_max = 5120
    code_family_size_vlines = [420, 5111]
    code_family_size_vline_labels = ["num codes used in ICD", "total num Weil codes"]
    plot_save_path = "H:\My Drive\Stanford_Documents\Documents\Research\Presentations\ION_GNSS_2023_Codes\presentation_images"
    plot_welch_bound_wrt_familysize(
        codelength,
        familysize_max,
        code_family_size_vlines,
        code_family_size_vline_labels,
        plot_save_path,
    )


#############################################################################################################################
