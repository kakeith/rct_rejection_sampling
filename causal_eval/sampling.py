"""
This contains implementations for the various sampling from RCTs methods 

Specifically: 
- Gentzel et al. 2021 OSRCT algorithm 
- (Ours) RCT rejection sampler 
"""
import os
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from scipy.special import expit
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt


def osrct_algorithm(data, rng, confound_func_params={"para_form": "linear"}):
    """
    Gentzel et al. 2021 OSRCT algorithm

    Inputs:
        - data: pd.DataFrame with columns "C" (covariates), "T" (treatment), "Y" (outcome)
        - f_function: the type of function we want to pass in as the researcher-specified P(S|C)
        - rng: numpy random number generator (so we have the same seed throughout)
        - confound_func_params : Dictionary with the parameters for the function that creates p_SC
            from researcher_specified_function_for_confounding()

    Returns sampled data to induce confounding
    """
    # generate probabilities of S (selection) as a function of C (covariates)
    p_SC = researcher_specified_function_for_confounding(data, confound_func_params=confound_func_params)

    bernoulli = rng.binomial(1, p_SC, len(data))

    # accept rows where (bernoulli == T)
    data_resampled = data[bernoulli == data["T"]]

    # return the resampled data
    data_resampled.reset_index()
    return data_resampled


def rejection_sampler(data, weights, rng, M=2, return_accepted_rows=False):
    """
    Our new proposed sampling algorithm

    Inputs:
        - data: pd.DataFrame with columns "C" (covariates), "T" (treatment), "Y" (outcome)
    """
    # draw N samples from a Uniform(0, 1)
    uniform_vector = rng.uniform(0, 1, len(data))

    # accept rows based on the condition U < weights/M
    accepted_rows = uniform_vector < (weights / M)
    data_resampled = data[accepted_rows]

    data_resampled.reset_index()

    # return data
    if return_accepted_rows:
        return data_resampled, accepted_rows
    return data_resampled


def weights_for_rejection_sampler(data, confound_func_params={"para_form": "linear"}):
    """
    We first specify weights as p(T|C)/p(T)

    """
    T = data["T"]
    pT = np.mean(T) * np.ones(len(data))
    pT[data["T"] == 0] = 1 - pT[data["T"] == 0]
    p_TC = researcher_specified_function_for_confounding(data, confound_func_params=confound_func_params)
    p_TC[data["T"] == 0] = 1 - p_TC[data["T"] == 0]
    weights = p_TC / pT
    return weights, p_TC, pT


def researcher_specified_function_for_confounding(
    data, confound_func_params={"para_form": "binary_piecewise", "zeta0": 0.2, "zeta1": 0.8}
):
    """
    Same biasing function for both sampling algorithms
    """
    # check that the `confund_func_params` has what it needs
    assert confound_func_params.get("para_form") != None
    if confound_func_params["para_form"] == "binary_piecewise":
        assert confound_func_params.get("zeta0") != None
        assert confound_func_params.get("zeta1") != None

    # linear function
    if confound_func_params["para_form"] == "linear":
        p_TC = expit(-1 + 2.5 * data["C"])  # TODO: could pass in the weight and intercept if we want to as well
    # for binary C, specify a piecewise function with zeta0 and zeta1
    elif confound_func_params["para_form"] == "binary_piecewise":
        p_TC = np.array([confound_func_params["zeta1"] if c == 1 else confound_func_params["zeta0"] for c in data["C"]])

    elif confound_func_params["para_form"] == "nonlinear":
        p_TC = expit(
            confound_func_params["C1"] * data["C1"]
            + confound_func_params["C2"] * data["C2"]
            + confound_func_params["C3"] * data["C3"]
            + confound_func_params["C4"] * data["C4"]
            + confound_func_params["C5"] * data["C5"]
            + confound_func_params["C1"] * data["C1"] * data["C2"]
        )

    return p_TC


def parametric_backdoor(data, Y, T, C, state_space="continuous"):
    """
    Compute the average causal effect E[Y(T=1)] - E[Y(T=0)] via backdoor adjustment

    Y: string corresponding variable name of the outcome
    T: string corresponding variable name
    C: list of variable names to be included in backdoor adjustment set
    """

    formula = Y + "~" + T
    if len(C) > 0:
        formula += " + " + "+".join(C)

    if state_space == "continuous":
        model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Gaussian()).fit()
    else:
        model = sm.GLM.from_formula(formula=formula, data=data, family=sm.families.Binomial()).fit()
    # print(model.params)
    data_T0 = data.copy()
    data_T1 = data.copy()
    data_T0[T] = 0
    data_T1[T] = 1
    return np.mean(model.predict(data_T1)) - np.mean(model.predict(data_T0))


def random_forest_backdoor(data, Y, A, C, state_space="continuous"):
    """
    Compute the average causal effect E[Y(A=1)] - E[Y(A=0)] via backdoor adjustment
    Y: string corresponding variable name of the outcome
    A: string corresponding variable name
    C: list of variable names to be included in backdoor adjustment set
    """

    Yvec = (
        data[Y]
        .to_numpy()
        .reshape(
            len(data),
        )
    )
    Ypredictors = data[[A] + C].to_numpy()

    # fit model for Y
    if state_space == "continuous":
        model_Y = RandomForestRegressor()
    elif state_space == "binary":
        model_Y = RandomForestClassifier()

    model_Y.fit(Ypredictors, Yvec)

    data_A0 = data.copy()
    data_A1 = data.copy()
    data_A0[A] = 0
    data_A1[A] = 1
    YpredictorsA1 = data_A1[[A] + C].to_numpy()
    YpredictorsA0 = data_A0[[A] + C].to_numpy()
    return np.mean(model_Y.predict(YpredictorsA1)) - np.mean(model_Y.predict(YpredictorsA0))


def iptw_estimate(data, propensity_scores):
    """
    Gives the IPTW estimate given the propensity scores
    """
    T = data["T"].to_numpy()
    Y = data["Y"].to_numpy()
    assert propensity_scores.shape[0] == T.shape[0]

    ate = np.mean(T * Y / propensity_scores - (1 - T) * Y / (1 - propensity_scores))
    return ate


def check_invalid_sample(data_resampled):
    """
    Checks through a few cases in which we cannot estimate the ACE. For all units,
        - T*C == T, or
        - T == C

    Returns True if the sample is invalid
    """
    tc = data_resampled["T"] * data_resampled["C"]
    if np.array_equal(tc, data_resampled["T"]):
        print("Invalid sample, T*C==T for all units")
        return True
    elif np.array_equal(data_resampled["T"], data_resampled["C"]):
        print("Invalid sample, T==C for all units")
        return True
    else:
        return False


def sampling_one_set_params(data, obs_dataset_config):
    """
    Inputs:
        - data: pd.DataFrame with T, C, Y as keys
        - obs_dataset_config: dict with the configurations for the sampling

    Output:
        - id2obs_sampl : dict with keys as the ids of the observational samples
        and the values are the data itself
    """
    assert "sampling_num_repeats" in obs_dataset_config.keys()
    assert "sampling_type" in obs_dataset_config.keys()
    assert "confound_func_params" in obs_dataset_config.keys()
    assert set(["C", "T", "Y"]) == set(data.columns)

    sampling_type = obs_dataset_config["sampling_type"]
    confound_func_params = obs_dataset_config["confound_func_params"]

    id2obs_sampl = {}
    for id in range(obs_dataset_config["sampling_num_repeats"]):
        # random seed different for each run
        rng = np.random.default_rng(id)

        if sampling_type == "rohit_rejection":
            weights, p_TC, pT = weights_for_rejection_sampler(data, confound_func_params=confound_func_params)
            sampl = rejection_sampler(data, weights, rng, M=np.max(p_TC) / np.min(pT))
        elif sampling_type == "osrct":
            sampl = osrct_algorithm(data, rng)

        id2obs_sampl[id] = sampl

    return id2obs_sampl


def one_hyperparam_setting(data, zeta0, zeta1, num_seeds_one_setting):
    """
    x-axis : Confounding in Sample: NaiveATE - RCT_ATE
    y-axis: Sample error bound: BackdoorATE - RCT_ATE
    """
    confound_func_params = {"para_form": "binary_piecewise", "zeta0": zeta0, "zeta1": zeta1}

    x_all_seeds = []
    y_all_seeds = []
    data_resampled_all_seeds = []

    rct_ace = parametric_backdoor(data, "Y", "T", [])

    for seed in range(num_seeds_one_setting):
        rng = np.random.default_rng(seed + 1000)

        # Run rejection sampler
        weights, p_TC, pT = weights_for_rejection_sampler(data, confound_func_params=confound_func_params)
        M = np.max(p_TC) / np.min(pT)
        data_resampled = rejection_sampler(data, weights, rng, M=M)
        data_resampled_all_seeds.append(data_resampled)

        # Collect results
        naive_ate = parametric_backdoor(data_resampled, "Y", "T", [])
        backdoor_adjusted_ate = parametric_backdoor(data_resampled, "Y", "T", ["C", "T*C"])
        x = naive_ate - rct_ace
        y = backdoor_adjusted_ate - rct_ace
        x_all_seeds.append(x)
        y_all_seeds.append(y)

    return np.array(x_all_seeds), np.array(y_all_seeds), data_resampled_all_seeds


def plot_results_one_setting(ax, x_all_seeds, y_all_seeds, zeta0, zeta1, title=None, abs=False, axes_lims=None):
    """
    Creates the diagnostic plots

    x_all_seeds : array for x-values of the plot which is "Sample Confounding: Abs(NaiveATE - GoldATE)"
    y_all_seeds: array for y-values of the plot which is "Sample Error Bound: \n Abs(BackdoorATE - GoldATE)"

    If title is not None, I can override the zeta0, zeta1 for different parameterizations of P*(T=1|C)
    """
    # print("blah")
    # return

    # change to the absolute values of x and y
    if abs:
        # print('x-axis mean = ', np.mean(np.abs(x_all_seeds)))
        # print('y-axis mean =', np.mean(np.abs(y_all_seeds)))
        ax.scatter(np.abs(x_all_seeds), np.abs(y_all_seeds))
        xlabel = "Sample Confounding: \n Abs(NaiveATE - GoldATE)"
        ylabel = "Sample Error Bound: \n Abs(BackdoorATE - GoldATE)"

    else:
        ax.scatter(x_all_seeds, y_all_seeds)
        xlabel = "Sample Confounding: NaiveATE - GoldATE"
        ylabel = "Sample Error Bound: BackdoorATE - GoldATE"

    if axes_lims is None:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
    else:
        lims = axes_lims

    ax.axvline(x=0, linestyle="--", alpha=0.2)
    ax.axhline(y=0, linestyle="--", alpha=0.2)

    # Plot a y=x line
    ax.plot(lims, lims, "r-", alpha=0.75, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # change the x, y labels
    if title is None:
        zeta0_pretty = np.round(zeta0, 3)
        zeta1_pretty = np.round(zeta1, 3)
        title = r"$\zeta_0$=" + str(zeta0_pretty) + r", $\zeta_1$=" + str(zeta1_pretty)

    ax.set_title(title, fontsize=30)


def diagnostic_plot(data, zeta0_zeta1_list, savefig_name, num_seeds_one_setting=100):
    num_grid_values = len(zeta0_zeta1_list)
    x_results = np.zeros((num_grid_values, num_seeds_one_setting))
    y_results = np.zeros((num_grid_values, num_seeds_one_setting))

    # get results
    for i, (zeta0, zeta1) in enumerate(zeta0_zeta1_list):
        x, y, _ = one_hyperparam_setting(data, zeta0, zeta1, num_seeds_one_setting)
        x_results[i] = x
        y_results[i] = y

    # Plot results
    plt.clf()
    plt.close()

    fig, axs = plt.subplots(1, num_grid_values, sharex=True, sharey=True, figsize=(25, 5))
    for i, (zeta0, zeta1) in enumerate(zeta0_zeta1_list):
        ax = axs[i]
        plot_results_one_setting(ax, x_results[i], y_results[i], zeta0, zeta1, abs=True, axes_lims=[0, 0.05])

    xlabel = "Sample Confounding: Abs(UnadjustedATE - GoldATE)"
    ylabel = "Sample Error Bound: \n Abs(BackdoorATE - GoldATE)"
    # common xlabel
    fig.text(0.5, -0.1, xlabel, ha="center", fontsize=30)
    # common ylabel
    fig.text(-0.03, -0.1, ylabel, ha="center", fontsize=30, rotation="vertical")

    fig.tight_layout()
    fig.savefig(savefig_name, format="png", bbox_inches="tight")
    print("saved to:: ", savefig_name)
    return fig


def many_seeds(data, rct_ace, confound_func_params, is_linear=True, num_seeds=1000):
    all_abs_error_obrct = []
    all_abs_error_rejection = []
    all_data_resampled_rejection = []

    for seed in range(num_seeds):
        rng = np.random.default_rng(seed)

        # OSRCT
        data_resampled = osrct_algorithm(data, rng, confound_func_params=confound_func_params)
        if is_linear:
            if check_invalid_sample(data_resampled): continue 
            sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C", "T*C"])
        else:
            sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C4", "T*C1*C2", "C2*C3", "C5"])
        abs_error = np.abs(sample_ace - rct_ace)
        all_abs_error_obrct.append(abs_error)

        # Rejection
        weights, p_TC, pT = weights_for_rejection_sampler(data, confound_func_params=confound_func_params)
        M = np.max(p_TC) / np.min(pT)
        data_resampled = rejection_sampler(data, weights, rng, M=M)
        if is_linear:
            if check_invalid_sample(data_resampled): continue 
            sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C", "T*C"])
        else:
            sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C4", "T*C1*C2", "C2*C3", "C5"])
        abs_error = np.abs(sample_ace - rct_ace)
        all_abs_error_rejection.append(abs_error)
        all_data_resampled_rejection.append(data_resampled)

    print("OBRCT num invalid samples=", num_seeds - len(all_abs_error_obrct))
    print("Rejection num invalid samples=", num_seeds - len(all_abs_error_rejection))
    print()

    print(f"OBSRCT: MAE (over{num_seeds} random seeds)= ", np.mean(all_abs_error_obrct))
    print(f"\tOBSRCT: std AE (over{num_seeds} random seeds)= ", np.std(all_abs_error_obrct))
    print(f"OBSRCT: Relative MAE (over{num_seeds} random seeds)= ", np.mean(all_abs_error_obrct / np.abs(rct_ace)))
    print(f"\tOBSRCT: Relative AE std (over{num_seeds} random seeds)= ", np.std(all_abs_error_obrct / np.abs(rct_ace)))

    print()
    print(f"Rejection: MAE (over {num_seeds} random seeds)= ", np.mean(all_abs_error_rejection))
    print(f"\tRejection: std AE (over {num_seeds} random seeds)= ", np.std(all_abs_error_rejection))
    print(
        f"Rejection: Relative MAE (over{num_seeds} random seeds)= ", np.mean(all_abs_error_rejection / np.abs(rct_ace))
    )
    print(
        f"\tRejection: Relative AE std (over{num_seeds} random seeds)= ",
        np.std(all_abs_error_rejection / np.abs(rct_ace)),
    )
    return all_abs_error_obrct, all_abs_error_rejection, all_data_resampled_rejection

if __name__ == "__main__":
    """
    Example usage of the RCT rejection sampling algorithm 
    """

    #load a dataset 
    fname = 'data/subpopA_physics_medicine.csv'
    data = pd.read_csv(fname)

    # Specify the parametric function of P(T|C) you want
    confound_func_params = {"para_form": "binary_piecewise", "zeta0": 0.15, "zeta1": 0.85} #binary piecewise function
    #some other options
    #confound_func_params={"para_form": "linear"} 
    #confound_func_params = {"para_form": "nonlinear", "C1": 1.5, "C2": -0.7, "C3": 1.2, "C4": 1.5, "C5": -1.2}

    # Run rejection sampler
    weights, p_TC, pT = weights_for_rejection_sampler(data, confound_func_params=confound_func_params)
    M = np.max(p_TC) / np.min(pT)
    rng = np.random.default_rng(10)
    data_resampled = rejection_sampler(data, weights, rng, M=M)

    print("Original data num. examples=", len(data))
    print("Downsampled data num. examples=", len(data_resampled))