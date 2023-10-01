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

def synthetic_dgp(setting=1, num_samples=100000): 
    """
    Returns the data and RCT ACE for one of our three synthetic DGPs
    """
    #keep the random seed for the DGP distinct from the sampling seed 
    rng_dgp = np.random.default_rng(0)

    #######################
    # DGP for RCT T->Y<-C
    ######################

    # Setting 1 
    # |C| = 1, P (T = 1) = 0.3
    if setting == 1: 
        C = rng_dgp.binomial(1, 0.5, num_samples)
        T = rng_dgp.binomial(1, 0.3, num_samples)
        Y = 0.5*C + 1.5*T + 2*T*C + rng_dgp.normal(0, 1, num_samples)
        data = pd.DataFrame({"C": C, "T": T, "Y": Y})

    # Setting 2 
    # |C| = 1, P (T = 1) = 0.5
    elif setting == 2: 
        C = rng_dgp.binomial(1, 0.5, num_samples)
        T = rng_dgp.binomial(1, 0.5, num_samples) #Set P(T=1)=0.5
        Y = 0.5*C + 1.5*T + 2*T*C + rng_dgp.normal(0, 1, num_samples)
        data = pd.DataFrame({"C": C, "T": T, "Y": Y}) 

    # Setting 3 
    # |C| = 5, Nonlinear
    elif setting == 3: 
        C1 = rng_dgp.binomial(1, 0.5, num_samples)
        C2 = C1 + rng_dgp.uniform(-0.5, 1, num_samples)
        C3 = rng_dgp.normal(0, 1, num_samples)
        C4 = rng_dgp.normal(0, 1, num_samples)
        C5 = C3 + C4 + rng_dgp.normal(0, 1, num_samples)
        T = rng_dgp.binomial(1, 0.3, num_samples)
        Y = 0.5*C4 + 2*T*C1*C2 - 1.5*T + C2*C3 + C5 + rng_dgp.normal(0, 1, num_samples)
        data = pd.DataFrame({"C1": C1, "C2": C2, "C3": C3, "C4": C4, "C5": C5, "T": T, "Y": Y})
        print("RCT ACE", np.mean(data[data['T'] == 1]["Y"]) - np.mean(data[data['T'] == 0]["Y"])) 

    # perform some sanity checks
    # in the RCT data we should see unadjusted == adjusted 
    rct_ace =  parametric_backdoor(data, "Y", "T", [])

    print("Sanity check")
    print("RCT ACE unadjusted: ", rct_ace)
    if setting != 3: print("RCT ACE adjusting for C parametric backdoor: ", parametric_backdoor(data, "Y", "T", ["C", "T*C"]))
    return data, rct_ace 


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

def bootstrapping_with_ace(data_resampled, is_rct=False, is_linear=True, num_bootstrap_samples=1000): 
    """
    Boostrap resample
    Calculate parameteric backdoor within

    is_rct : True if its from the RCT data so we just use the difference in means 
        estimator 

    is_linear : indicates the DGP that we pass into the parametric backdoor 

    num_bootstrap_samples: number of times to do bootstrap resampling 
    """
    # Step 1: Use sample data from RCT 
    # results in data_resampled

    # Step 2: Bootstrapping: resample $S$ with replacement $b$ times.
    def bootstrap_sample(dataframe):
        return dataframe.sample(n=len(dataframe), replace=True)

    all_sample_ace = []
    for i in range(num_bootstrap_samples): 
        boot_sample_df =  bootstrap_sample(data_resampled)

        # Step 3: Calculate the ACE for each bootstrap sample. 
        # RCT is just difference in means 
        if is_rct: sample_ace = parametric_backdoor(boot_sample_df, "Y", "T", [])
        # Otherwise, the parametric backdoor 
        elif is_linear:
            if check_invalid_sample(boot_sample_df): continue 
            sample_ace = parametric_backdoor(boot_sample_df, "Y", "T", ["C", "T*C"])
        else: #non-linear case 
            sample_ace = parametric_backdoor(boot_sample_df, "Y", "T", ["C4", "T*C1*C2", "C2*C3", "C5"])
        all_sample_ace.append(sample_ace)

    # Step 4: Use the percentile method to obtain 95% confidence intervals. 
    all_sample_ace = np.array(all_sample_ace)
    assert len(all_sample_ace) == num_bootstrap_samples
    low = np.percentile(all_sample_ace, 2.5)
    high = np.percentile(all_sample_ace, 97.5)

    return {"boostrap_all_sample_ace": np.array(all_sample_ace), 
            "ci_low_95": low, 
            "ci_high_95": high, 
            "ci_mean": np.mean(all_sample_ace)}

def cacluate_ci_coverage(boostrap_out, rct_ace): 
    """
    Returns 1 if the true (RCT) ACE is in the confidence interval 
    Returns 0 otherwise 

    bootstrap_out = {"boostrap_all_sample_ace": np.array(all_sample_ace), 
            "ci_low_95": low, 
            "ci_high_95": high, 
            "ci_mean": np.mean(all_sample_ace)}
    """
    if boostrap_out["ci_low_95"] <= rct_ace <= boostrap_out["ci_high_95"]: 
        return 1
    else: return 0 

def bootstrapping_three_methods_linear(data, rct_ace, confound_func_params): 
    """
    Does bootstrap resampling for 
    1. Original RCT data
    2. RCT rejection sampling
    3. Gentzel et al 

    one *one* sample 

    Then creates a boxplot 
    """
    rng = np.random.default_rng(100) 
    # RCT data only 
    rct_out = bootstrapping_with_ace(data, is_rct=True, is_linear=True)

    # Gentzel et al 
    data_resampled = osrct_algorithm(data, rng, confound_func_params=confound_func_params)
    gentzel_out =  bootstrapping_with_ace(data_resampled, is_linear=True)

    # RCT rejection sampling 
    weights, p_TC, pT = weights_for_rejection_sampler(data, confound_func_params=confound_func_params)
    M = np.max(p_TC) / np.min(pT)
    data_resampled = rejection_sampler(data, weights, rng, M=M)
    rejection_out =  bootstrapping_with_ace(data_resampled, is_linear=True)

    data_out = {'RCT': rct_out, 'Gentzel':gentzel_out, 'Rejection': rejection_out}
    return data_out

def plot_bootstrap(data_out, rct_ace, title):
    """
    Plot the 95% CIs and their means 
    """ 
    method_order = ['RCT', 'Rejection', 'Gentzel']
    labels = ['RCT', 'RCT rejection sampling', 'Gentzel et al']

    # Unpack the data 
    means, lower_cis, upper_cis = [], [], []
    for method in method_order:
        means.append(data_out[method]['ci_mean'])
        lower_cis.append(data_out[method]['ci_low_95'])
        upper_cis.append(data_out[method]['ci_high_95'])

    # Calculate the error values (distance from the mean)
    errors = [(mean - lower, upper - mean) for mean, lower, upper in zip(means, lower_cis, upper_cis)]
    errors = list(zip(*errors))  # Transpose the list

    # Plot the data with error bars
    fig, ax = plt.subplots()
    ax.errorbar(range(len(means)), means, yerr=errors, fmt='o', capsize=5, color='black')

    # Styling and labels
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('ACE')
    ax.set_title(title)

    #Put in the true value 
    ax.axhline(y=rct_ace, color='red', linewidth=2, label="True (RCT) ACE", linestyle='--')

    ax.legend()

    plt.tight_layout()
    return fig, ax


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

def one_seed(out, seed, data, rct_ace, confound_func_params, is_linear=True, 
               has_bootstrap=False, num_bootstrap_samples=1000): 
    """
    One random seed, both methods 
    """
    rng = np.random.default_rng(seed)

    # OSRCT
    data_resampled = osrct_algorithm(data, rng, confound_func_params=confound_func_params)
    if is_linear:
        if check_invalid_sample(data_resampled): return out 
        sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C", "T*C"])
    else:
        sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C4", "T*C1*C2", "C2*C3", "C5"])
    abs_error = np.abs(sample_ace - rct_ace)
    out['all_abs_error_obrct'].append(abs_error)

    # OSRCT Bootstrapping 
    if has_bootstrap: 
        boot_out = bootstrapping_with_ace(data_resampled, is_rct=False, is_linear=is_linear, 
                                            num_bootstrap_samples=num_bootstrap_samples)
        ci_cov = cacluate_ci_coverage(boot_out, rct_ace)
        out['osrct_ci_coverage'].append(ci_cov)

    # Rejection
    weights, p_TC, pT = weights_for_rejection_sampler(data, confound_func_params=confound_func_params)
    M = np.max(p_TC) / np.min(pT)
    data_resampled = rejection_sampler(data, weights, rng, M=M)
    if is_linear:
        if check_invalid_sample(data_resampled): return out 
        sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C", "T*C"])
    else:
        sample_ace = parametric_backdoor(data_resampled, "Y", "T", ["C4", "T*C1*C2", "C2*C3", "C5"])
    abs_error = np.abs(sample_ace - rct_ace)
    out['all_abs_error_rejection'].append(abs_error)
    out['all_data_resampled_rejection'].append(data_resampled)

    # Rejection Bootstrapping 
    if has_bootstrap: 
        boot_out = bootstrapping_with_ace(data_resampled, is_rct=False, is_linear=is_linear, 
                                            num_bootstrap_samples=num_bootstrap_samples)
        ci_cov = cacluate_ci_coverage(boot_out, rct_ace)
        out['rejection_ci_coverage'].append(ci_cov)
    return out 

def many_seeds(data, rct_ace, confound_func_params, is_linear=True, num_seeds=1000, 
               has_bootstrap=False, num_bootstrap_samples=1000, has_print_out=True,
               run_in_parallel=False, num_cores=20):
    
    out = {}
    out['all_abs_error_obrct'] = []
    out['all_abs_error_rejection'] = []
    out['all_data_resampled_rejection'] = []

    out['osrct_ci_coverage'] = []
    out['rejection_ci_coverage'] = []

    if not run_in_parallel: 
        for seed in range(num_seeds):
            out = one_seed(out, seed, data, rct_ace, confound_func_params, is_linear=is_linear, 
                has_bootstrap=has_bootstrap, num_bootstrap_samples=num_bootstrap_samples)
    
    # run in parallel, multiple cores 
    elif run_in_parallel: 
        pass 
        
    # Print results
    if has_print_out: 
        print("OBRCT num invalid samples=", num_seeds - len(out['all_abs_error_obrct']))
        print("Rejection num invalid samples=", num_seeds - len(out['all_abs_error_rejection']))
        print()

        print(f"OBSRCT: MAE (over{num_seeds} random seeds)= ", np.mean(out['all_abs_error_obrct']))
        print(f"\tOBSRCT: std AE (over{num_seeds} random seeds)= ", np.std(out['all_abs_error_obrct']))
        print(f"OBSRCT: Relative MAE (over{num_seeds} random seeds)= ", np.mean(out['all_abs_error_obrct'] / np.abs(rct_ace)))
        print(f"\tOBSRCT: Relative AE std (over{num_seeds} random seeds)= ", np.std(out['all_abs_error_obrct'] / np.abs(rct_ace)))

        print()
        print(f"Rejection: MAE (over {num_seeds} random seeds)= ", np.mean(out['all_abs_error_rejection']))
        print(f"\tRejection: std AE (over {num_seeds} random seeds)= ", np.std(out['all_abs_error_rejection']))
        print(
            f"Rejection: Relative MAE (over{num_seeds} random seeds)= ", np.mean(out['all_abs_error_rejection'] / np.abs(rct_ace))
        )
        print(
            f"\tRejection: Relative AE std (over{num_seeds} random seeds)= ",
            np.std(out['all_abs_error_rejection'] / np.abs(rct_ace)),
        )

        print()
        print("=== CI Coverage === ")
        print(f"Num bootstrap samples={num_bootstrap_samples}")
        print("OSRCT CI Coverage: ", np.mean(out['osrct_ci_coverage']))
        print("Rejection CI Coverage:", np.mean(out['rejection_ci_coverage']))
    return out

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