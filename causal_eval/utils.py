"""
This file contains helper functions for results and visualizations
"""
from collections import defaultdict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def print_predictive_one_experiment(
    expname, expname_for_print, desired_pred_metric, expname2results, num_seeds_one_setting
):
    # formatting for latex
    traininf2color = {"train": "yellow", "inf": "green"}

    est, train_pred, inf_pred = expname2results[expname]
    assert len(est) == len(train_pred) == len(inf_pred) == num_seeds_one_setting

    # gather the desired_pred_metric into a clean dict of dict

    traininf2model2metric = {}

    for traininf, traininf_results in zip(["train", "inf"], (train_pred, inf_pred)):
        model2metric = defaultdict(list)
        for seed in traininf_results:
            for key, value in seed.items():
                ss = key.split("---")
                model = ss[0]
                metric = ss[1]
                if metric == desired_pred_metric:
                    # print(metric, value)
                    model2metric[model].append(value)
        traininf2model2metric[traininf] = model2metric

    # now print out mean and std in order

    print("PREDICTIVE MODELS")
    print("METRIC =", desired_pred_metric)

    header1 = ""
    header2 = ""
    str_out = expname_for_print
    #### THESE TWO ORDERING ARE IMPORTANT AND SHOULD BE THE SAME AS THE LATEX TABLE
    for model in ["treatment_model", "y_model_Tlearner_T=0", "y_model_Tlearner_T=1", "y_model_both"]:
        header1 += f" & {model} & "
        for traininf in ["train", "inf"]:
            ########################
            this_loop = traininf2model2metric[traininf][model]
            ave = np.round(np.mean(this_loop), 2)
            std = np.round(np.std(this_loop), 2)

            color = traininf2color[traininf]
            color_value = np.round(100 * ave, 0)
            color_value = min(100, color_value)  # package maxes out at 100
            str_out += f"& {ave} ({std}) \\cellcolor {{{color}!{color_value}}}"
            header2 += f" & {traininf}"

    print(header1)
    print(header2)
    print()
    print(str_out)

    return header1, header2, str_out


def print_estimation_one_experiment(expname, expname_for_print, expname2results, num_seeds_one_setting, rct_ace):
    # formatting for latex
    color = "red"

    est, train_pred, inf_pred = expname2results[expname]
    assert len(est) == len(train_pred) == len(inf_pred) == num_seeds_one_setting

    # Calcualte relative absolute error
    print("CAUSAL ESTIMATION: RELATIVE ABSOLUTE ERROR")

    model2rel_error = defaultdict(list)
    for seed in est:
        for model in [
            "naive_ate",
            "backdoor_c_ate",
            "outcome_regression_ate",
            "IPTW_ate",
            "AIPTW_ate",
            "double_ml_ate",
        ]:
            ate = seed[model]

            # relative absolute error between estimate on observational sample and RCT ATE
            rel_abs_error = np.abs(rct_ace - ate) / rct_ace

            model_name = model.rstrip("ate").rstrip("_")
            model2rel_error[model_name].append(rel_abs_error)

    # Printout for latex
    header1 = ""
    str_out = expname_for_print
    color = "red"

    for model, error_list in model2rel_error.items():
        ave = np.round(np.mean(error_list), 2)
        std = np.round(np.std(error_list), 2)
        str_out += f"& {ave} ({std})"
        # don't add color for naive
        if model not in ["naive", "backdoor_c"]:
            color_value = np.round(100 * ave, 0)
            color_value = min(100, color_value)  # package maxes out at 100
            str_out += f"\\cellcolor {{{color}!{color_value}}}"
        header1 += f"& {model}"

    print(header1)
    print()
    print(str_out)
    return header1, str_out


def resampled_data_cleanup(data_resampled_all_seeds, vec):
    # Clean up data by vectorizing the resampled X's to have the same BOW CountVectorizer as the RCT

    # change each resampled data to be a dictionary, (X's are now count vectors)
    data_resampled_dict_all_seeds = []
    for resampled in data_resampled_all_seeds:
        data_dict = {}
        for key in ["Y", "T", "C"]:
            data_dict[key] = resampled[key].to_numpy().astype(int)

        # create X
        texts = resampled["X"].to_numpy()
        X = vec.transform(texts)
        data_dict["X"] = X

        data_resampled_dict_all_seeds.append(data_dict)
    return data_resampled_dict_all_seeds


def model_reports_to_tidy_data(model_report_list, num_obs_sampl_seeds, train_or_test=None):
    """
    Input: model report list

    Output:
        pd.DataFrame
            tidy data format, each entry is a dictionary

    """
    assert len(model_report_list) == num_obs_sampl_seeds

    data = []
    str_sep = "---"

    for seed, one_seed_data in enumerate(model_report_list):
        for k, v in one_seed_data.items():
            ss = k.split(str_sep)
            assert len(ss) == 2

            out = {}
            out["obs_sampl_seed"] = seed
            out["train_or_test"] = train_or_test
            out["pred_model_type"] = ss[0]  # e.g. "y_model_Tlearner_T=1"
            out["metric"] = ss[1]  # e.g "f1"
            out["metric_value"] = float(v)
            data.append(out)

    return data


def estimations_to_tidy_data(
    all_seeds_results, feasible_obs_results, num_obs_sampl_seeds, true_ate, desired_metric=None
):
    """
    desired_metric = "rel_abs_error" means we want the ATE error relative to the gold ate
    """
    assert len(all_seeds_results) == num_obs_sampl_seeds == len(feasible_obs_results)

    data = []
    for seed, one_seed_data in enumerate(all_seeds_results):
        for k, v in one_seed_data.items():
            ss = k.split("_")
            if ss[-1] == "ci":
                continue  # dont do this for now
            out = {}
            out["obs_sampl_seed"] = seed
            out["est_model"] = str("_".join(ss[0 : len(ss) - 1]))
            out["metric"] = ss[-1]
            out["metric_value"] = v
            data.append(out)

        # add the pearl back door
        out = {}
        out["obs_sampl_seed"] = seed
        out["est_model"] = "pearl_backdoor"
        out["metric"] = "ate"
        out["metric_value"] = feasible_obs_results[seed]["pearl_backdoor"]
        out["zeta"] = feasible_obs_results[seed]["zeta"]
        data.append(out)

    # switch at the very end if necessary
    if desired_metric is None:
        return data
    elif desired_metric == "rel_abs_error":
        data2 = []
        for dat in data:
            dat["metric"] = "rel_abs_error"
            dat["metric_value"] = np.abs(dat["metric_value"] - true_ate) / true_ate
            data2.append(dat)
    return data2


def grid_viz_aggs(
    results1, feasible_obs_results, len_feasible_obs_ids, expname, plot_title, pred_metric="roc_auc", save=False
):
    """
    same as grid_viz but only the aggregates

    columns: (2), left side = prediction error stuff
                    right side = estimation error stuff
    """
    # set-up data
    all_seeds_results = results1["all_seeds_results"]
    obs_results_list = results1["obs_results_list"]
    assert len(all_seeds_results) == len(feasible_obs_results)
    true_ate = results1["data_stats"]["true_ate"]

    model_report_list = results1["train_all_seeds_pred_model_report"]
    df_viz1 = model_reports_to_tidy_data(model_report_list, len_feasible_obs_ids, train_or_test="train")

    model_report_list = results1["inference_all_seeds_pred_model_report"]
    df_viz2 = model_reports_to_tidy_data(model_report_list, len_feasible_obs_ids, train_or_test="inference")

    df_viz = pd.DataFrame(df_viz1 + df_viz2)

    df_viz_est = estimations_to_tidy_data(
        all_seeds_results, feasible_obs_results, len_feasible_obs_ids, true_ate, desired_metric="rel_abs_error"
    )
    df_viz_est = pd.DataFrame(df_viz_est)

    # set-up plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # overall
    # plot 1: prediction overall
    pred_model_order = ["treatment_model", "y_model_Tlearner_T=0", "y_model_Tlearner_T=1", "y_model_both"]
    this_axis = axes[0]
    here_metric = pred_metric
    here_df = df_viz.loc[df_viz["metric"] == here_metric]
    box_plot = sns.boxplot(
        data=here_df, x="pred_model_type", y="metric_value", hue="train_or_test", ax=this_axis, order=pred_model_order
    )
    ##add the medians as annotations, https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    medians = here_df.groupby(["pred_model_type", "train_or_test"])["metric_value"].median()
    medians_in_order = []
    for model in pred_model_order:
        for ttype in ["train", "inference"]:
            medians_in_order.append(medians[(model, ttype)])
    # vertical_offset = here_df['pred_model_type'].median() * 0.05 # offset from median for display
    i = 0
    for xtick in box_plot.get_xticks():
        for hue in [-0.2, 0.2]:
            box_plot.text(
                xtick + hue,
                medians_in_order[i] * 1.03,
                np.round(medians_in_order[i], 2),
                horizontalalignment="center",
                size="small",
                color="black",
                weight="semibold",
            )
            i += 1
    # more clean-up
    xticks = [x.get_text() for x in this_axis.get_xticklabels()]
    this_axis.set_xticklabels(xticks, rotation=45)
    this_axis.set_title("Ave. all seeds: Predictive Model Reports")
    this_axis.set_ylabel(here_metric)
    this_axis.axhline(y=0.5, linestyle="dashed")

    # plot 2: estimation overall
    model_order = ["naive", "pearl_backdoor", "outcome_regression", "IPTW", "AIPTW", "double_ml"]
    this_axis = axes[1]
    here_metric = "rel_abs_error"
    here_df = df_viz_est.loc[df_viz_est["metric"] == here_metric]
    box_plot = sns.boxplot(data=here_df, x="est_model", y="metric_value", order=model_order, ax=this_axis)
    # add the medians as annotations, https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    medians = here_df.groupby(["est_model"])["metric_value"].median()
    medians_in_order = [medians[model_name] for model_name in model_order]
    vertical_offset = here_df["metric_value"].median() * 0.1  # offset from median for display
    for xtick in box_plot.get_xticks():
        box_plot.text(
            xtick,
            medians_in_order[xtick] + vertical_offset,
            np.round(medians_in_order[xtick], 2),
            horizontalalignment="center",
            size="small",
            color="black",
            weight="semibold",
        )
    # more clean-up
    xticks = [x.get_text() for x in this_axis.get_xticklabels()]
    this_axis.set_xticklabels(xticks, rotation=45)
    this_axis.set_title("Ave. all seeds: Causal Estimation (ATE) Error")
    this_axis.set_ylabel(here_metric)
    this_axis.annotate("Gold ATE={0:.3f}".format(true_ate), xy=(0, 1), color="r")
    this_axis.set_ylim(0, 1.0)

    fig.suptitle(plot_title, fontweight="bold", y=1.0005)
    plt.tight_layout()

    if save:
        fout = f"results/{expname}.png"
        plt.savefig(fout)
        print("saved to ->", fout)

    return fig, axes, plt


# TODO: clean up this function
def grid_viz(
    results1, feasible_obs_results, len_feasible_obs_ids, expname, plot_title, pred_metric="roc_auc", save=True
):
    """
    rows: different observational samples
    columns: (2), left side = prediction error stuff
                    right side = estimation error stuff
    """
    # set-up data
    all_seeds_results = results1["all_seeds_results"]
    obs_results_list = results1["obs_results_list"]
    assert len(all_seeds_results) == len(feasible_obs_results)
    true_ate = results1["data_stats"]["true_ate"]

    model_report_list = results1["train_all_seeds_pred_model_report"]
    df_viz1 = model_reports_to_tidy_data(model_report_list, len_feasible_obs_ids, train_or_test="train")

    model_report_list = results1["inference_all_seeds_pred_model_report"]
    df_viz2 = model_reports_to_tidy_data(model_report_list, len_feasible_obs_ids, train_or_test="inference")

    df_viz = pd.DataFrame(df_viz1 + df_viz2)

    df_viz_est = estimations_to_tidy_data(
        all_seeds_results, feasible_obs_results, len_feasible_obs_ids, true_ate, desired_metric="rel_abs_error"
    )
    df_viz_est = pd.DataFrame(df_viz_est)

    # set-up plot
    fig, axes = plt.subplots(len_feasible_obs_ids + 1, 2, figsize=(15, int(15 * len_feasible_obs_ids / 3)))

    # overall
    # plot 1: prediction overall
    pred_model_order = ["treatment_model", "y_model_Tlearner_T=0", "y_model_Tlearner_T=1", "y_model_both"]
    this_axis = axes[0][0]
    here_metric = pred_metric
    here_df = df_viz.loc[df_viz["metric"] == here_metric]
    box_plot = sns.boxplot(
        data=here_df, x="pred_model_type", y="metric_value", hue="train_or_test", ax=this_axis, order=pred_model_order
    )
    ##add the medians as annotations, https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    medians = here_df.groupby(["pred_model_type", "train_or_test"])["metric_value"].median()
    medians_in_order = []
    for model in pred_model_order:
        for ttype in ["train", "inference"]:
            medians_in_order.append(medians[(model, ttype)])
    # vertical_offset = here_df['pred_model_type'].median() * 0.05 # offset from median for display
    i = 0
    for xtick in box_plot.get_xticks():
        for hue in [-0.2, 0.2]:
            box_plot.text(
                xtick + hue,
                medians_in_order[i] * 1.03,
                np.round(medians_in_order[i], 2),
                horizontalalignment="center",
                size="small",
                color="black",
                weight="semibold",
            )
            i += 1
    # more clean-up
    xticks = [x.get_text() for x in this_axis.get_xticklabels()]
    this_axis.set_xticklabels(xticks, rotation=45)
    this_axis.set_title("Ave. all seeds: Predictive Model Reports")
    this_axis.set_ylabel(here_metric)
    this_axis.axhline(y=0.5, linestyle="dashed")

    # plot 2: estimation overall
    model_order = ["naive", "pearl_backdoor", "outcome_regression", "IPTW", "AIPTW", "double_ml"]
    this_axis = axes[0][1]
    here_metric = "rel_abs_error"
    here_df = df_viz_est.loc[df_viz_est["metric"] == here_metric]
    box_plot = sns.boxplot(data=here_df, x="est_model", y="metric_value", order=model_order, ax=this_axis)
    # add the medians as annotations, https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
    medians = here_df.groupby(["est_model"])["metric_value"].median()
    medians_in_order = [medians[model_name] for model_name in model_order]
    vertical_offset = here_df["metric_value"].median() * 0.1  # offset from median for display
    for xtick in box_plot.get_xticks():
        box_plot.text(
            xtick,
            medians_in_order[xtick] + vertical_offset,
            np.round(medians_in_order[xtick], 2),
            horizontalalignment="center",
            size="small",
            color="black",
            weight="semibold",
        )
    # more clean-up
    xticks = [x.get_text() for x in this_axis.get_xticklabels()]
    this_axis.set_xticklabels(xticks, rotation=45)
    this_axis.set_title("Ave. all seeds: Causal Estimation (ATE) Error")
    this_axis.set_ylabel(here_metric)
    this_axis.annotate("Gold ATE={0:.3f}".format(true_ate), xy=(0, 1), color="r")
    this_axis.set_ylim(0, 2.0)

    # breakdown
    for row in range(len_feasible_obs_ids):
        here_obs_sampl_seed = row

        # plot 1
        this_axis = axes[row + 1][0]
        here_metric = pred_metric
        here_df = df_viz.loc[(df_viz["obs_sampl_seed"] == here_obs_sampl_seed) & (df_viz["metric"] == here_metric)]
        sns.barplot(data=here_df, x="pred_model_type", y="metric_value", hue="train_or_test", ax=this_axis)
        xticks = [x.get_text() for x in this_axis.get_xticklabels()]
        this_axis.set_xticklabels(xticks, rotation=45)

        real_seed_number = str(feasible_obs_results[row]["ii"])
        this_zeta = str(feasible_obs_results[row]["zeta"])

        this_axis.set_title(f"Obs sampl seed={real_seed_number} (zeta={this_zeta}) : Predictive Model Reports")
        this_axis.set_ylabel(here_metric)
        this_axis.axhline(y=0.5, linestyle="dashed")

        # plot 2
        this_axis = axes[row + 1][1]
        here_metric = "rel_abs_error"
        here_df = df_viz_est.loc[
            (df_viz_est["obs_sampl_seed"] == here_obs_sampl_seed) & (df_viz_est["metric"] == here_metric)
        ]

        sns.barplot(
            data=here_df,
            x="est_model",
            y="metric_value",
            order=["naive", "pearl_backdoor", "outcome_regression", "IPTW", "AIPTW", "double_ml"],
            ax=this_axis,
        )
        xticks = [x.get_text() for x in this_axis.get_xticklabels()]
        this_axis.set_xticklabels(xticks, rotation=45)
        this_axis.set_title(f"Obs sampl seed={real_seed_number} (zeta={this_zeta}) : Causal Estimation (ATE) Error")
        this_axis.set_ylabel(here_metric)
        # this_axis.annotate('Gold ATE={0:.3f}'.format(true_ate), xy=(0, 1), color='r')
        this_axis.set_ylim(0, 2.0)

    fig.suptitle(plot_title, fontweight="bold", y=1.0005)
    plt.tight_layout()

    if save:
        fout = f"results/{expname}.png"
        plt.savefig(fout)
        print("saved to ->", fout)

    return fig, axes, plt


def x_y_estimation_estimation_plots(
    expname,
    expname2results,
    feasible_obs_results,
    len_feasible_obs_ids,
    plot_title,
    desired_est_metric="rel_abs_error",
    desired_x_model="pearl_backdoor",
    desired_y_model="outcome_regression",
):
    """
    Like x_y_subset_analysis_plots except both models are estimation ones
    """
    # set-up data
    results1 = expname2results[expname]
    all_seeds_results = results1["all_seeds_results"]
    true_ate = results1["data_stats"]["true_ate"]

    # causal estimation data frame 1
    df_viz_est = estimations_to_tidy_data(
        all_seeds_results, feasible_obs_results, len_feasible_obs_ids, true_ate, desired_metric=desired_est_metric
    )
    df_viz_est = pd.DataFrame(df_viz_est)

    # TODO: there is probably a pandas function that does this better
    desired_x_label = f"{desired_x_model}: Causal Rel. Est. Error"
    desired_y_label = f"{desired_y_model}: Causal Rel. Est. Error"  # estimation model that uses predictive model

    seed2x = {}
    seed2y = {}

    # estimation x-axis
    for i, row in df_viz_est.iterrows():
        seed = row["obs_sampl_seed"]
        if row["est_model"] == desired_x_model:
            seed2x[seed] = row["metric_value"]

    # estimation y-axis
    for i, row in df_viz_est.iterrows():
        seed = row["obs_sampl_seed"]
        if row["est_model"] == desired_y_model:
            seed2y[seed] = row["metric_value"]

    assert len(seed2x) == len(seed2y)
    xvals = []
    yvals = []
    for seed in seed2x.keys():
        xvals.append(seed2x[seed])
        yvals.append(seed2y[seed])

    print("Pearson (r, p-value)=", pearsonr(xvals, yvals))

    sns.scatterplot(x=xvals, y=yvals)
    plt.xlabel(desired_x_label)
    plt.ylabel(desired_y_label)
    plt.title(plot_title)
    plt.show()
    return plt


def pairwise_seed_plots(
    expname2results,
    obs_dataset_config,
    train_or_inference="train",
    pred_model="y_model_both",
    pred_metric="acc",
    est_desired="double_ml_ate",
):
    # Katie: I think we actually only care about pairwise improvements in prediction and estimation error
    # Otherwise there's too many moving parts with the confounding bias and the sample bias
    # x-axis: Prediction difference: inference, catboost acc y_both - logreg acc y_both
    pred_desired = f"{pred_model}---{pred_metric}"
    x1 = [x[pred_desired] for x in expname2results["no_catboost"][f"{train_or_inference}_all_seeds_pred_model_report"]]
    x2 = [x[pred_desired] for x in expname2results["catboost"][f"{train_or_inference}_all_seeds_pred_model_report"]]
    assert len(x1) == len(x2) == obs_dataset_config["n_zeta_repeats"]
    x_vals = np.array(x2) - np.array(x1)

    # y-axis: Estimation difference: double_ml + catboost rel abs error - double_ml+catboost rel abs error
    true_ate = expname2results["no_catboost"]["data_stats"]["true_ate"]
    assert (
        expname2results["no_catboost"]["data_stats"]["true_ate"]
        == expname2results["catboost"]["data_stats"]["true_ate"]
    )
    y1 = [np.abs(x[est_desired] - true_ate) / true_ate for x in expname2results["no_catboost"]["all_seeds_results"]]
    y2 = [np.abs(x[est_desired] - true_ate) / true_ate for x in expname2results["catboost"]["all_seeds_results"]]
    assert len(y1) == len(y2) == obs_dataset_config["n_zeta_repeats"]
    y_vals = np.array(y2) - np.array(y1)

    print("Pearson (r, p-value)=", pearsonr(x_vals, y_vals))
    plt.scatter(x_vals, y_vals)
    plt.xlabel(f"Prediction metrics: (catboost)-(logreg), {train_or_inference} {pred_model} {pred_metric}")
    plt.ylabel(f"Estimation metrics: (catboost)-(logreg), rel. abs. error, {est_desired}")
    plt.show()


def x_y_subset_analysis_plots(
    expname,
    expname2results,
    feasible_obs_results,
    len_feasible_obs_ids,
    plot_title,
    desired_est_metric="rel_abs_error",
    desired_pred_metric="calibration_rmse",
    train_or_test="inference",
    desired_x_model="treatment_model",
    desired_y_model="IPTW",
):
    """
    Relationship between two variables for the overall results
    """
    # set-up data
    results1 = expname2results[expname]
    all_seeds_results = results1["all_seeds_results"]
    true_ate = results1["data_stats"]["true_ate"]

    # prediction data frame
    model_report_list = results1[f"{train_or_test}_all_seeds_pred_model_report"]
    df_viz_pred_inf = pd.DataFrame(
        model_reports_to_tidy_data(model_report_list, len_feasible_obs_ids, train_or_test=train_or_test)
    )

    # causal estimation data frame
    df_viz_est = estimations_to_tidy_data(
        all_seeds_results, feasible_obs_results, len_feasible_obs_ids, true_ate, desired_metric=desired_est_metric
    )
    df_viz_est = pd.DataFrame(df_viz_est)

    # TODO: there is probably a pandas function that does this better
    desired_x_label = (
        f"Predictive Models: {desired_x_model} {desired_pred_metric} ({train_or_test} folds)"  # predictive model
    )
    desired_x_metric = desired_pred_metric

    desired_y_label = (
        f"Causal Relative Estimation Error: {desired_y_model}"  # estimation model that uses predictive model
    )

    seed2x = {}
    seed2y = {}

    # prediction
    for i, row in df_viz_pred_inf.iterrows():
        seed = row["obs_sampl_seed"]
        if row["pred_model_type"] == desired_x_model and row["metric"] == desired_x_metric:
            seed2x[seed] = row["metric_value"]

    # estimation
    for i, row in df_viz_est.iterrows():
        seed = row["obs_sampl_seed"]
        if row["est_model"] == desired_y_model:
            seed2y[seed] = row["metric_value"]

    assert len(seed2x) == len(seed2y)
    xvals = []
    yvals = []
    for seed in seed2x.keys():
        xvals.append(seed2x[seed])
        yvals.append(seed2y[seed])

    print("Pearson (r, p-value)=", pearsonr(xvals, yvals))

    sns.scatterplot(x=xvals, y=yvals)
    plt.xlabel(desired_x_label)
    plt.ylabel(desired_y_label)
    plt.title(plot_title)
    plt.show()
    return plt
