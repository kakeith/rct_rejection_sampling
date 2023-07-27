from copy import deepcopy
from pprint import pprint
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from causal_eval.ml_utils import (
    get_default_scoring,
    get_cv,
    mean_predictions,
    mean_truth,
    calibration_rmse,
    CLASSIFIER,
    CLASSIFIER_GRID,
    REGRESSOR,
    REGRESSOR_GRID,
)
from causal_eval.sampling import parametric_backdoor
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from joblib import Parallel, delayed
import pandas as pd
import json


def pearl_backdoor_discrete_ate(X, y, t, oracle_confounder_indices):
    """
    Caculate the ATE via Pearl's backdoor criterion for confounders that are *discrete*.
    Given that confounders are discrete, we can just do the explicit sum.

    Caveats:
    - As dimensionality of X gets larger, X will not satisfy overlap

    Pearl's backdoor:
        P(Y| do(T=t)) = \sum_x E(Y|T=t, X=x)P(X=x)

    Thus the ATE becomes:
        ATE =  E(Y| do(T=1)) -  E(Y| do(T=0))
            = \sum_x ( E(Y|T=1, X=x)- E(Y|T=0, X=x))P(X=x)

    Args:
        X (np.ndarray): all covariates, shape=(n_observations, dim_covariates) these must be discrete values
        y (np.ndarray)
        t (np.ndarray)
        oracle_confounder_indices (np.ndarray): indices for columns of X which are the true confounders
    """
    assert X.shape[0] == y.shape[0] == t.shape[0]
    assert len(oracle_confounder_indices) <= X.shape[1]

    t0_idx = np.where(t == 0)[0]
    t1_idx = np.where(t == 1)[0]

    X_conf = X[:, oracle_confounder_indices]
    # collapse multi-dimensional confounders (multiple columns) into string (single column)
    X_conf_collapase = np.array([str(arr) for arr in X_conf[:,]])
    conf_set = set(X_conf_collapase)

    # do the sum_x, at the same time check for overlap
    all_sum = 0
    for this_x in conf_set:
        this_x_idx = np.where(X_conf_collapase == this_x)[0]
        this_x_t0_idx = list(set(t0_idx) & set(this_x_idx))
        this_x_t1_idx = list(set(t1_idx) & set(this_x_idx))

        # check overlap, for each unique x value, we need at least one T=0 and T=1 instance
        if len(this_x_t0_idx) == 0 or len(this_x_t1_idx) == 0:
            raise Exception(f"Overlap not satisfied for x={this_x}")

        # do the sum
        pr_this_x = len(this_x_idx) / X.shape[0]
        all_sum += (np.mean(y[this_x_t1_idx]) - np.mean(y[this_x_t0_idx])) * pr_this_x

    return all_sum


class BagOfCausalInferences:
    """
    Wrapper for various causal inference algorithms
    all of which are done in a T-Learner approach where we first fit 3 models:
    (1) Pr(T = 1 | X)
    (2) E[Y | T = 0, X] (continuous outcome) or Pr(Y = 1 | T = 0, X) (discrete outcome)
    (3) E[Y | T = 1, X] or Pr(Y = 1 | T = 1, X)
    And then combine them in 4 different ways:
    (1) Outcome regression: E[Y_i | T_i = 1, X_i] - E[Y_i | T_i = 0, X_i]
    (2) IPTW, Eq (38.28) of Kevin Murphy's Probabilistic Machine Learning: Advanced Topics (2022)
        We use propensity score clipping
    (3) AIPTW, Eq (3) of https://www.law.berkeley.edu/files/AIPW(1).pdf
        Code inspired by https://github.com/pzivich/zEpid
    (4) Double ML.
        Code inspired by https://github.com/battani/EconML/blob/master/prototypes/orthogonal_forests/residualizer.py
    Args:
        y_model (sklearn model): model to use for predicting y
            If binary, must support predict_proba
        y_model_grid (dict): grid search parameters for y_model
        t_model (sklearn model): model to use for predicting t
            Only support binary for now -> must support predict_proba
        t_model_grid (dict): grid search parameters for t_model
        n_crossfit (int): number of folds for cross-fitting
        n_innercv (int): number of folds for inner cross-validation
        n_repeats (int): number of repeats for the entire procedure
        repeats_combine_function (np function): function to combine the results of the repeats
            np.median is default. np.mean is a sensible alternative
        propensity_score_cutoff (float): propensity scores can't be too close to 0
            So, (1 - propensity_score) and propensity_score will be clipped to this value
        ci_alpha (float): alpha for confidence intervals. defaults to 0.05
        n_jobs (int): number of jobs to use for cross-validation
        random_state (int): random state for cross-validation
    """

    def __init__(
        self,
        y_model=None,
        y_model_grid=None,
        t_model=None,
        t_model_grid=None,
        n_crossfit=4,
        n_innercv=4,
        n_repeats=10,
        repeats_combine_function=np.median,
        propensity_score_cutoff=0.01,  # can turn this off by setting it to 0.00
        ci_alpha=0.05,
        n_jobs=16,
        random_state=0,
    ):
        self.y_model = deepcopy(y_model)
        self.y_model_grid = y_model_grid
        self.t_model = deepcopy(t_model)
        self.t_model_grid = t_model_grid
        self.n_crossfit = n_crossfit
        self.n_innercv = n_innercv
        self.n_repeats = n_repeats
        self.repeats_combine_function = repeats_combine_function
        self.propensity_score_cutoff = np.minimum(propensity_score_cutoff, 1 - propensity_score_cutoff)
        self.ci_alpha = ci_alpha
        self.zalpha = norm.ppf(1 - self.ci_alpha / 2, loc=0, scale=1)
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.X = None
        self.y = None
        self.t = None
        self.crossfit_splits = []
        self.treatment_effects_outcome_regression = None
        self.treatment_effects_singly_robust = None
        self.treatment_effects_doubly_robust = None
        self.treatment_effects_double_ml = None
        self.y_predictions_t_0 = None
        self.y_predictions_t_1 = None
        self.y_predictions = None
        self.t_predictions = None
        self.fitted = False
        self.y_task = None

        # training predictions for scoring (keeping separate from inference predictions above)
        self.training = {}
        self.training["t_predictions"] = None
        self.training["y_predictions_t_0"] = None
        self.training["y_predictions_t_1"] = None
        self.training["y_predictions"] = None
        # also store the training true values since it will be hard to recreate these with the crossfitting split
        self.training["t_true"] = None
        self.training["y_true"] = None

    def fit(self, X, y, t):
        """
        Fit the underlying ML models with cross-fitting and repeats
        Args:
            X (np.array): Features/covariates. n_samples x n_features, can be sparse
            y (np.array): Outcome measurement, n_samples x 1
            t (np.array): Treatment measurement, n_samples x 1
                Only supporting binary for now
        Returns: self
        """
        # what is y prediction regression or binary classification?
        num_unique_y = len(np.unique(y))
        if y.dtype == float:
            self.y_task = "regression"
        elif num_unique_y == 2:
            self.y_task = "binary"
            y = y.astype(int)
        else:
            raise ValueError("`y` must be binary or a float. Multiclass `y` is not supported.")

        y = y.flatten()
        t = t.astype(int).flatten()

        self.X = X
        self.y = y
        self.t = t

        # what metrics to optimize for inner-cv
        scoring_y = get_default_scoring(self.y_task)
        scoring_t = get_default_scoring("binary")

        # specifiying models if not provided
        if self.y_model is None:
            if self.y_task == "regression":
                y_model = deepcopy(REGRESSOR)
                y_model_grid = REGRESSOR_GRID
            else:
                y_model = deepcopy(CLASSIFIER)
                y_model_grid = CLASSIFIER_GRID
        else:
            y_model = deepcopy(self.y_model)
            y_model_grid = self.y_model_grid

        if self.t_model is None:
            t_model = deepcopy(CLASSIFIER)
            t_model_grid = CLASSIFIER_GRID
        else:
            t_model = deepcopy(self.t_model)
            t_model_grid = self.t_model_grid

        # will need later
        y_model_params = y_model.get_params()
        t_model_params = t_model.get_params()

        # store all the treatment effects
        self.y_predictions_t_0 = np.zeros((len(y), self.n_repeats))
        self.y_predictions_t_1 = np.zeros((len(y), self.n_repeats))
        self.y_predictions = np.zeros((len(y), self.n_repeats))
        self.t_predictions = np.zeros((len(y), self.n_repeats))

        # store training predictions for the report
        self.training["y_predictions_t_0"] = np.zeros((len(y), self.n_crossfit, self.n_repeats))
        self.training["y_predictions_t_1"] = np.zeros((len(y), self.n_crossfit, self.n_repeats))
        self.training["y_predictions"] = np.zeros((len(y), self.n_crossfit, self.n_repeats))
        self.training["t_predictions"] = np.zeros((len(y), self.n_crossfit, self.n_repeats))
        # we'll store this so we can ignore the inference indices during the model report
        self.training["train_inds_mask"] = np.zeros((len(y), self.n_crossfit, self.n_repeats), dtype=bool)
        self.training["train_inds_t_0"] = np.zeros((len(y), self.n_crossfit, self.n_repeats), dtype=bool)
        self.training["train_inds_t_1"] = np.zeros((len(y), self.n_crossfit, self.n_repeats), dtype=bool)

        for i_repeat in range(self.n_repeats):
            random_state_i = self.random_state + i_repeat
            # split the data up such that there are T = 0 and 1 in both train and test.
            # this is needed for T-learner approach to work.
            # also, if y_task is classification, then we have to split on that too
            if self.y_task == "regression":
                split_var = t
            else:
                split_var = [f"{i}_{j}" for i, j in zip(t, y)]
            crossfit_split = list(
                StratifiedKFold(n_splits=self.n_crossfit, shuffle=True, random_state=random_state_i).split(X, split_var)
            )
            self.crossfit_splits.append(crossfit_split)  # we'll need this later
            for crossfit_number, (train_inds, test_inds) in enumerate(crossfit_split):
                # y models cross_validation and then prediction
                # the T-learner splits up the data into two groups and learns two models
                # E[y | X, t=0] and E[y | X, t=1]
                # train_inds is a set of indices, and we need to find only those indices that are also t==0
                train_inds_mask = np.zeros(len(y), dtype=bool)
                train_inds_mask[train_inds] = True
                train_inds_t_0 = train_inds_mask & (t == 0)
                train_inds_t_1 = train_inds_mask & (t == 1)

                self.training["train_inds_mask"][:, crossfit_number, i_repeat] = train_inds_mask
                self.training["train_inds_t_0"][:, crossfit_number, i_repeat] = train_inds_t_0
                self.training["train_inds_t_1"][:, crossfit_number, i_repeat] = train_inds_t_1

                # y model for T == 0
                if y_model_grid is not None and self.n_innercv is not None:
                    y_model_gridsearch = GridSearchCV(
                        estimator=y_model,
                        param_grid=y_model_grid,
                        cv=get_cv(self.y_task, n_splits=self.n_innercv, random_state=random_state_i),
                        scoring=scoring_y,
                        n_jobs=self.n_jobs,
                        refit=True,
                    )
                else:
                    y_model_gridsearch = y_model
                    if "n_jobs" in y_model_params:
                        y_model_gridsearch.set_params(n_jobs=self.n_jobs)
                    elif "model__n_jobs" in y_model_params:
                        y_model_gridsearch.set_params(model__n_jobs=self.n_jobs)

                # ipdb.set_trace()

                y_model_gridsearch.fit(X[train_inds_t_0, :], y[train_inds_t_0])
                # make predictions on both the train and inference sets
                if self.y_task == "regression":
                    self.y_predictions_t_0[test_inds, i_repeat] = y_model_gridsearch.predict(X[test_inds])
                    self.training["y_predictions_t_0"][
                        train_inds_t_0, crossfit_number, i_repeat
                    ] = y_model_gridsearch.predict(X[train_inds_t_0, :])
                else:
                    self.y_predictions_t_0[test_inds, i_repeat] = y_model_gridsearch.predict_proba(X[test_inds])[:, 1]
                    self.training["y_predictions_t_0"][
                        train_inds_t_0, crossfit_number, i_repeat
                    ] = y_model_gridsearch.predict_proba(X[train_inds_t_0, :])[:, 1]

                # y model for T == 1
                if y_model_grid is not None and self.n_innercv is not None:
                    y_model_gridsearch = GridSearchCV(
                        estimator=y_model,
                        param_grid=y_model_grid,
                        cv=get_cv(self.y_task, n_splits=self.n_innercv, random_state=random_state_i),
                        scoring=scoring_y,
                        n_jobs=self.n_jobs,
                        refit=True,
                    )
                else:
                    y_model_gridsearch = y_model
                    if "n_jobs" in y_model_params:
                        y_model_gridsearch.set_params(n_jobs=self.n_jobs)
                    elif "model__n_jobs" in y_model_params:
                        y_model_gridsearch.set_params(model__n_jobs=self.n_jobs)

                y_model_gridsearch.fit(X[train_inds_t_1], y[train_inds_t_1])
                # make predictions on both the train and inference sets
                if self.y_task == "regression":
                    self.y_predictions_t_1[test_inds, i_repeat] = y_model_gridsearch.predict(X[test_inds])
                    self.training["y_predictions_t_1"][
                        train_inds_t_1, crossfit_number, i_repeat
                    ] = y_model_gridsearch.predict(X[train_inds_t_1, :])
                else:
                    self.y_predictions_t_1[test_inds, i_repeat] = y_model_gridsearch.predict_proba(X[test_inds])[:, 1]
                    self.training["y_predictions_t_1"][
                        train_inds_t_1, crossfit_number, i_repeat
                    ] = y_model_gridsearch.predict_proba(X[train_inds_t_1, :])[:, 1]

                # y model overall that doesn't know about T
                if y_model_grid is not None and self.n_innercv is not None:
                    y_model_gridsearch = GridSearchCV(
                        estimator=y_model,
                        param_grid=y_model_grid,
                        cv=get_cv(self.y_task, n_splits=self.n_innercv, random_state=random_state_i),
                        scoring=scoring_y,
                        n_jobs=self.n_jobs,
                        refit=True,
                    )
                else:
                    y_model_gridsearch = y_model
                    if "n_jobs" in y_model_params:
                        y_model_gridsearch.set_params(n_jobs=self.n_jobs)
                    elif "model__n_jobs" in y_model_params:
                        y_model_gridsearch.set_params(model__n_jobs=self.n_jobs)

                y_model_gridsearch.fit(X[train_inds], y[train_inds])
                # make predictions on both the train and inference sets
                if self.y_task == "regression":
                    self.y_predictions[test_inds, i_repeat] = y_model_gridsearch.predict(X[test_inds])
                    self.training["y_predictions"][train_inds, crossfit_number, i_repeat] = y_model_gridsearch.predict(
                        X[train_inds, :]
                    )
                else:
                    self.y_predictions[test_inds, i_repeat] = y_model_gridsearch.predict_proba(X[test_inds])[:, 1]
                    self.training["y_predictions"][
                        train_inds, crossfit_number, i_repeat
                    ] = y_model_gridsearch.predict_proba(X[train_inds, :])[:, 1]

                # t model cross_validation and then prediction
                if t_model_grid is not None and self.n_innercv is not None:
                    t_model_gridsearch = GridSearchCV(
                        estimator=t_model,
                        param_grid=t_model_grid,
                        cv=get_cv("binary", n_splits=self.n_innercv, random_state=random_state_i),
                        scoring=scoring_t,
                        n_jobs=self.n_jobs,
                        refit=True,
                    )
                else:
                    t_model_gridsearch = t_model
                    if "n_jobs" in t_model_params:
                        t_model_gridsearch.set_params(n_jobs=self.n_jobs)
                    elif "model__n_jobs" in t_model_params:
                        t_model_gridsearch.set_params(model__n_jobs=self.n_jobs)

                t_model_gridsearch.fit(X[train_inds], t[train_inds])
                # make predictions on both the train and inference sets
                self.t_predictions[test_inds, i_repeat] = t_model_gridsearch.predict_proba(X[test_inds])[:, 1]
                self.training["t_predictions"][
                    train_inds, crossfit_number, i_repeat
                ] = t_model_gridsearch.predict_proba(X[train_inds])[:, 1]

        self.fitted = True
        return self

    def outcome_regression_ate(self):
        """Get average treatment effect via outcome regression"""
        self.treatment_effects_outcome_regression = np.zeros((len(self.y), self.n_repeats))
        ates_outcome_regression = np.zeros(self.n_repeats)
        ate_vars_outcome_regression = np.zeros(self.n_repeats)
        for i_repeat in range(self.n_repeats):
            self.treatment_effects_outcome_regression[:, i_repeat] = (
                self.y_predictions_t_1[:, i_repeat] - self.y_predictions_t_0[:, i_repeat]
            )

            ate_i, ate_var_i = self.ate_combine(
                self.treatment_effects_outcome_regression[:, i_repeat], self.crossfit_splits[i_repeat]
            )
            ates_outcome_regression[i_repeat] = ate_i
            ate_vars_outcome_regression[i_repeat] = ate_var_i

        ate_outcome_regression, ate_ci_outcome_regression = self.ate_combine_overall(
            ates_outcome_regression, ate_vars_outcome_regression
        )

        return ate_outcome_regression, ate_ci_outcome_regression

    def IPTW_ate(self):
        """Get average treatment effect via IPTW"""
        self.treatment_effects_singly_robust = np.zeros((len(self.y), self.n_repeats))
        ates_singly_robust = np.zeros(self.n_repeats)
        ate_vars_singly_robust = np.zeros(self.n_repeats)
        for i_repeat in range(self.n_repeats):
            # propensity and 1-propensity shouldn't be too small or the variance will be huge
            propensity = np.clip(
                self.t_predictions[:, i_repeat], self.propensity_score_cutoff, 1 - self.propensity_score_cutoff
            )
            one_minus_propensity = 1 - propensity

            ######## IPTW EQUATION ######
            self.treatment_effects_singly_robust[:, i_repeat] = (
                self.t * self.y / propensity - (1 - self.t) * self.y / one_minus_propensity
            )
            #############################

            ate_i, ate_var_i = self.ate_combine(
                self.treatment_effects_singly_robust[:, i_repeat], self.crossfit_splits[i_repeat]
            )
            ates_singly_robust[i_repeat] = ate_i
            ate_vars_singly_robust[i_repeat] = ate_var_i

        ate_singly_robust, ate_ci_singly_robust = self.ate_combine_overall(ates_singly_robust, ate_vars_singly_robust)

        return ate_singly_robust, ate_ci_singly_robust

    def AIPTW_ate(self):
        """Get average treatment effect via AIPTW"""
        self.treatment_effects_doubly_robust = np.zeros((len(self.y), self.n_repeats))
        ates_doubly_robust = np.zeros(self.n_repeats)
        ate_vars_doubly_robust = np.zeros(self.n_repeats)
        for i_repeat in range(self.n_repeats):
            # propensity and 1-propensity shouldn't be too small or the variance will be huge
            propensity = np.clip(
                self.t_predictions[:, i_repeat], self.propensity_score_cutoff, 1 - self.propensity_score_cutoff
            )
            one_minus_propensity = 1 - propensity

            ####### AIPTW Equation #######
            self.treatment_effects_doubly_robust[:, i_repeat] = (
                self.y_predictions_t_1[:, i_repeat]
                - self.y_predictions_t_0[:, i_repeat]
                + self.t * (self.y - self.y_predictions_t_1[:, i_repeat]) / propensity
                - (1 - self.t) * (self.y - self.y_predictions_t_0[:, i_repeat]) / (one_minus_propensity)
            )
            ##############################

            ate_i, ate_var_i = self.ate_combine(
                self.treatment_effects_doubly_robust[:, i_repeat], self.crossfit_splits[i_repeat]
            )
            ates_doubly_robust[i_repeat] = ate_i
            ate_vars_doubly_robust[i_repeat] = ate_var_i

        ate_doubly_robust, ate_ci_doubly_robust = self.ate_combine_overall(ates_doubly_robust, ate_vars_doubly_robust)

        return ate_doubly_robust, ate_ci_doubly_robust

    def double_ml_ate(self):
        """
        Get average treatment effect via Double ML

        Referece: http://aeturrell.com/2018/02/10/econometrics-in-python-partI-ML/
            See the code part of that reference
        """
        ates_double_ml = np.zeros(self.n_repeats)
        ate_vars_double_ml = np.zeros(self.n_repeats)
        for i_repeat in range(self.n_repeats):
            # residuals
            propensity = np.clip(
                self.t_predictions[:, i_repeat], self.propensity_score_cutoff, 1 - self.propensity_score_cutoff
            )
            residual_t = self.t - self.t_predictions[:, i_repeat]
            residual_y = self.y - self.y_predictions[:, i_repeat]

            # OLS on the residuals to get the treatment effect
            # statsmodels computes everything for us
            # note that self.treatment_effect_double_ml will stay None
            # as we are just going the estimate of the overall ATE
            model = sm.OLS(residual_y, residual_t).fit()

            # the default for OLS does not estimate an intercept
            # so index 0 is the coefficient
            ates_double_ml[i_repeat] = model.params[0]
            ate_vars_double_ml[i_repeat] = model.cov_params()[0][0]

        ate_double_ml, ate_ci_double_ml = self.ate_combine_overall(ates_double_ml, ate_vars_double_ml)

        return ate_double_ml, ate_ci_double_ml

    def ate_combine_overall(self, ates, ate_vars):
        """Overall effect is the mean or median of effects across repeats
        This suggests median by default:
        https://github.com/pzivich/zEpid/blob/master/zepid/causal/doublyrobust/crossfit.py#L1602
        Args:
            ates (np.array): average treatment effects
            ate_vars (np.array): average treatment effect variances
        Returns:
            ate (float): average treatment effect
            ate_var (float): average treatment effect variance
        """
        ate = self.repeats_combine_function(ates)
        ate_var = self.repeats_combine_function(np.array(ate_vars) / len(self.y) + (ates - ate) ** 2)
        ate_se = np.sqrt(ate_var)
        ate_ci = (ate - self.zalpha * ate_se, ate + self.zalpha * ate_se)
        return ate, ate_ci

    def pred_model_report(self, is_training=False, verbose=True, str_sep="---"):
        """
        Reports the prediction model scores
        Args:
            is_training (bool, optional): store as True if you want to print out the training scores instead of inference
        """
        pred_model_report_out = {}

        # these classification metrics need the "hard" predictions, e.g. y=[0, 0, 1, ...]
        class_hard_name2metric_func = {
            "f1": f1_score,
            "acc": accuracy_score,
            "mean_hard_pred": mean_predictions,
            "mean_true": mean_truth,  # should be same for hard or soft
        }

        # these classification metrics need the "score" predictions, e.g. y=[0.6, 0.77, 0.2, ...]
        class_scores_name2metric_func = {
            "roc_auc": roc_auc_score,
            "ave_prec": average_precision_score,
            "calibration_rmse": calibration_rmse,
            "mean_soft_pred": mean_predictions,
            "mean_true": mean_truth,  # should be same for hard or soft
        }

        regression_name2metric_func = {"MSE": mean_squared_error, "R2": r2_score}

        # make hard predictions for report
        if is_training:
            t_pred = self.training["t_predictions"][self.training["train_inds_mask"]]
        else:
            t_pred = self.t_predictions
        t_pred_hard = (t_pred > 0.5).astype(int)

        # repeat the true value across multiple repeats
        # this should mirror self.t_predictions which takes shape self.t_predictions = np.zeros((len(y), self.n_repeats))
        if is_training:
            t_true = np.repeat(self.t, self.n_crossfit * self.n_repeats).reshape(
                len(self.t), self.n_crossfit, self.n_repeats
            )
            # sanity check that we're repeating correctly
            np.testing.assert_array_equal(t_true[:, 0, 0], self.t)
            # training, only take the training masks
            t_true = t_true[self.training["train_inds_mask"]]
        else:
            t_true = np.repeat(self.t, self.n_repeats).reshape(len(self.t), self.n_repeats)
            # sanity check that we're repeating correctly
            np.testing.assert_array_equal(t_true[:, 0], self.t)

        # now unravel once we know they're the same shape
        assert t_pred_hard.shape == t_true.shape == t_pred.shape
        t_pred = t_pred.reshape(-1)
        t_pred_hard = t_pred_hard.reshape(-1)
        t_true = t_true.reshape(-1)

        # T, propensity score model
        for metric_str, metric_func in class_hard_name2metric_func.items():
            pred_model_report_out["treatment_model" + str_sep + metric_str] = metric_func(t_true, t_pred_hard)
        for metric_str, metric_func in class_scores_name2metric_func.items():
            pred_model_report_out["treatment_model" + str_sep + metric_str] = metric_func(t_true, t_pred)

        # Y, outcome model set-up data
        if is_training:
            y_pred_t0 = self.training["y_predictions_t_0"][self.training["train_inds_t_0"]]
            y_pred_t1 = self.training["y_predictions_t_1"][self.training["train_inds_t_1"]]
            y_pred = self.training["y_predictions"][self.training["train_inds_mask"]]
            # training for T-learner t0, t1 we need to look at the t=1, t=0 training masks
            y_true_t0 = np.repeat(self.y, self.n_crossfit * self.n_repeats).reshape(
                len(self.y), self.n_crossfit, self.n_repeats
            )[self.training["train_inds_t_0"]]
            y_true_t1 = np.repeat(self.y, self.n_crossfit * self.n_repeats).reshape(
                len(self.y), self.n_crossfit, self.n_repeats
            )[self.training["train_inds_t_1"]]
            y_true = np.repeat(self.y, self.n_crossfit * self.n_repeats).reshape(
                len(self.y), self.n_crossfit, self.n_repeats
            )[self.training["train_inds_mask"]]
            assert y_pred_t0.shape == y_true_t0.shape
            assert y_pred_t1.shape == y_true_t1.shape
            # don't need to unravel b/c the training indices already do this
        else:
            y_pred_t0 = self.y_predictions_t_0
            y_pred_t1 = self.y_predictions_t_1
            y_pred = self.y_predictions
            # inference we don't need to look at the t=1, t=0 training masks\
            y_true = np.repeat(self.y, self.n_repeats).reshape(len(self.y), self.n_repeats)
            assert y_pred_t0.shape == y_pred_t1.shape == y_pred.shape == y_true.shape
            # now unravel
            y_pred_t0 = y_pred_t0.reshape(-1)[t_true == 0]
            y_pred_t1 = y_pred_t1.reshape(-1)[t_true == 1]
            y_pred = y_pred.reshape(-1)
            y_true = y_true.reshape(-1)
            y_true_t0 = y_true[t_true == 0]
            y_true_t1 = y_true[t_true == 1]

        if self.y_task == "regression":
            for metric_str, metric_func in regression_name2metric_func.items():
                pred_model_report_out["y_model_Tlearner_T=0" + str_sep + metric_str] = metric_func(y_true_t0, y_pred_t0)

            for metric_str, metric_func in regression_name2metric_func.items():
                pred_model_report_out["y_model_Tlearner_T=1" + str_sep + metric_str] = metric_func(y_true_t1, y_pred_t1)

            for metric_str, metric_func in regression_name2metric_func.items():
                pred_model_report_out["y_model_both" + str_sep + metric_str] = metric_func(y_true, y_pred)
        # classification
        else:
            # classification change to hard predictions
            y_pred_t0_hard = (y_pred_t0 > 0.5).astype(int)
            y_pred_t1_hard = (y_pred_t1 > 0.5).astype(int)
            y_pred_hard = (y_pred > 0.5).astype(int)

            for metric_str, metric_func in class_hard_name2metric_func.items():
                pred_model_report_out["y_model_Tlearner_T=0" + str_sep + metric_str] = metric_func(
                    y_true_t0, y_pred_t0_hard
                )
            for metric_str, metric_func in class_scores_name2metric_func.items():
                pred_model_report_out["y_model_Tlearner_T=0" + str_sep + metric_str] = metric_func(y_true_t0, y_pred_t0)

            for metric_str, metric_func in class_hard_name2metric_func.items():
                pred_model_report_out["y_model_Tlearner_T=1" + str_sep + metric_str] = metric_func(
                    y_true_t1, y_pred_t1_hard
                )
            for metric_str, metric_func in class_scores_name2metric_func.items():
                pred_model_report_out["y_model_Tlearner_T=1" + str_sep + metric_str] = metric_func(y_true_t1, y_pred_t1)

            for metric_str, metric_func in class_hard_name2metric_func.items():
                pred_model_report_out["y_model_both" + str_sep + metric_str] = metric_func(y_true, y_pred_hard)
            for metric_str, metric_func in class_scores_name2metric_func.items():
                pred_model_report_out["y_model_both" + str_sep + metric_str] = metric_func(y_true, y_pred)

        if verbose:
            print_predictive_report(pred_model_report_out, is_training=is_training)
        return pred_model_report_out

    @staticmethod
    def ate_combine(x, crossfit_split):
        """Combine treatment effects across splits
        Source: https://github.com/pzivich/zEpid/blob/master/zepid/causal/utils.py#L746
        Args:
            x (np.array): treatment effeccts
            crossfit_split ([train_inds, test_inds]): _description_
        Returns:
            ate (float): average treatment effect
            ate_var_i (float): variance of treatment effect
        """
        ate_i = np.nanmean(x)
        ate_vars_local = []
        for _, test_inds in crossfit_split:
            ate_vars_local.append(np.nanvar(x[test_inds] - ate_i, ddof=1))
        ate_var_i = np.mean(ate_vars_local)
        return ate_i, ate_var_i


def print_predictive_report(pred_model_report_out, is_training=False):
    if is_training:
        str_name = "TRAINING"
    else:
        str_name = "INFERENCE"
    print("----" * 10)
    print(str_name + " predictive models report")
    print("----" * 10)
    pprint(pred_model_report_out)
    print("----" * 10 + "\n")


def baseline_estimation_one_seed(data_dict, est_methods_config, n_jobs=1):
    """ """
    ci_model = BagOfCausalInferences(
        n_crossfit=est_methods_config["n_crossfit"],
        n_innercv=est_methods_config["n_innercv"],
        n_repeats=est_methods_config["n_repeats"],
        n_jobs=n_jobs,
        y_model=est_methods_config["y_model"],
        y_model_grid=est_methods_config["y_model_grid"],
        t_model=est_methods_config["t_model"],
        t_model_grid=est_methods_config["t_model_grid"],
        propensity_score_cutoff=est_methods_config["propensity_score_cutoff"],
    ).fit(data_dict["X"], data_dict["Y"], data_dict["T"])

    assert est_methods_config["y_learning_task"] == ci_model.y_task

    train_pred_model_report_out = ci_model.pred_model_report(verbose=False, is_training=True)
    inf_pred_model_report_out = ci_model.pred_model_report(verbose=False, is_training=False)

    results = {}
    results["outcome_regression_ate"], results["outcome_regression_ate_ci"] = ci_model.outcome_regression_ate()
    results["IPTW_ate"], results["IPTW_ate_ci"] = ci_model.IPTW_ate()
    results["AIPTW_ate"], results["AIPTW_ate_ci"] = ci_model.AIPTW_ate()
    results["double_ml_ate"], results["double_ml_ate_ci"] = ci_model.double_ml_ate()
    results["naive_ate"] = calc_naive_ate(data_dict["Y"], data_dict["T"])
    results["backdoor_c_ate"] = parametric_backdoor(pd.DataFrame(data_dict), "Y", "T", ["C", "T*C"])

    return {
        "results": results,
        "train_pred_model_report_out": train_pred_model_report_out,
        "inf_pred_model_report_out": inf_pred_model_report_out,
    }


def calculate_ate(data):
    """
    E[Y|T=1] - E[Y|T=0]

    This works for both the naive and the RCT estimates
    """
    y1 = np.mean([unit["y"] for unit in data if unit["t"] == 1])
    y0 = np.mean([unit["y"] for unit in data if unit["t"] == 0])
    return y1 - y0


def calc_naive_ate(y, t):
    return np.mean(y[t == 1]) - np.mean(y[t == 0])


def calc_backdoor_c(data):
    """
    Backdoor adjustment with the low-dimensional covariates (C)
        (As opposed to the estimation methods that use high-dimensional X)
    """
    data_copy = deepcopy(data)
    del data_copy["X"]
    print(data_copy["C"].shape, data_copy["T"].shape)
    backdoor_adjusted_ate = parametric_backdoor(data_copy, "Y", "T", ["C", "T*C"])
    return backdoor_adjusted_ate


def baseline_estimation_all_seeds(data_resampled_dict_all_seeds, est_methods_config):
    data_stats = {}

    # apply the estimation methods to the observational sampled datasets in parallel
    print("fitting estimation models")
    all_seeds_out = Parallel(
        n_jobs=est_methods_config["n_jobs_obs_samp"], prefer="threads", verbose=5, require="sharedmem"
    )(
        delayed(baseline_estimation_one_seed)(data_dict, est_methods_config, n_jobs=1)
        for data_dict in data_resampled_dict_all_seeds
    )

    all_seeds_results = [x["results"] for x in all_seeds_out]
    train_all_seeds_pred_model_report = [x["train_pred_model_report_out"] for x in all_seeds_out]
    inference_all_seeds_pred_model_report = [x["inf_pred_model_report_out"] for x in all_seeds_out]

    return all_seeds_results, train_all_seeds_pred_model_report, inference_all_seeds_pred_model_report


def combine_pred_model_report(all_seeds_pred_model_report):
    """
    Takes the mean across all observational samples (all seeds)
    """
    assert len(all_seeds_pred_model_report) > 0
    keys = all_seeds_pred_model_report[0].keys()
    key2ave = {}
    for k in keys:
        data = [x[k] for x in all_seeds_pred_model_report]
        key2ave[k] = np.mean(np.array(data))
    return key2ave


def num_t1(dataset):
    # returns the number for which T=1
    return len([x for x in dataset if x["t"] == 1])


def odds_ratio(Y: np.ndarray, C: np.ndarray):
    """
    Given Y is binary and C is binary returns the odds ratio

    odds ratio

    (count Y==1 and C==1)/(count Y==0 and C==1) /
    (count Y==1 and C==0)/(count Y==0 and C==0)

    Tests (can use with doctest):

    >>> import numpy as np
    >>> Y = np.array([0, 0, 0, 1, 1, 1])
    >>> C = np.array([0, 1, 0, 1, 0, 1])
    >>> odds_ratio(Y, C)
    4.0

    # test where one of the denominators is 0
    >>> Y = np.array([1, 1, 1, 1, 0])
    >>> C = np.array([0, 1, 0, 1, 1])
    >>> odds_ratio(Y, C)
    nan
    """
    assert C.shape == (len(Y),)
    assert set(Y) == set([0, 1])
    assert set(C) == set([0, 1])

    Y1C1 = np.count_nonzero(Y[C == 1] == 1)
    Y0C1 = np.count_nonzero(Y[C == 1] == 0)
    Y1C0 = np.count_nonzero(Y[C == 0] == 1)
    Y0C0 = np.count_nonzero(Y[C == 0] == 0)

    # check that the denominators are not equal to 0
    if Y0C1 == 0 or Y0C0 == 0 or (Y1C0 / Y0C0) == 0:
        return np.nan

    return (Y1C1 / Y0C1) / (Y1C0 / Y0C0)


def dataset_checks(data):
    X = data["X"].to_numpy()
    Y = data["Y"].to_numpy()
    C = data["C"].to_numpy()
    T = data["T"].to_numpy()

    # Checks
    print("==== DATASET CHECKS ====")
    print("Num observations=", len(X))
    print(
        "Num duplicate papers = ",
        len(X) - len(set(X)),
        "({}%)".format(np.round((len(X) - len(set(X))) / len(X) * 100, 2)),
    )
    print("C=1, C=0 balance=", np.mean(C))
    oratio = odds_ratio(Y, C)
    print("Odds ratio (Y, C)=", oratio)
    ate = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    print("RCT_ATE = ", ate)
    print("Mean Y = ", np.mean(Y))
    print("T=1% ", np.mean(T))
    return {"rct_ate": ate, "odds_ratio": oratio, "num_observations": len(X)}


if __name__ == "__main__":
    """
    An example usage 
    """
    import dowhy.datasets

    # beta is true ATE
    ground_truth_ate = 10
    print("CAUSAL ESTIMATION ON TOY DATASET")
    data = dowhy.datasets.linear_dataset(
        beta=ground_truth_ate,
        num_common_causes=5,
        num_effect_modifiers=0,
        num_instruments=0,
        num_samples=1000,
        treatment_is_binary=True,
    )
    df = data["df"]

    t = df["v0"].values
    y = df["y"].values
    X = df[["W0", "W1", "W2", "W3", "W4"]].values
    print("Proportion T=1:", t.mean())

    ci_model = BagOfCausalInferences(
        y_model=REGRESSOR, y_model_grid=None, t_model=CLASSIFIER, t_model_grid=None, n_crossfit=4, n_repeats=10
    )
    ci_model.fit(X, y, t)

    outcome_regression_ate, outcome_regression_ate_ci = ci_model.outcome_regression_ate()
    IPTW_ate, IPTW_ate_ci = ci_model.IPTW_ate()
    AIPTW_ate, AIPTW_ate_ci = ci_model.AIPTW_ate()
    double_ml_ate, double_ml_ate_ci = ci_model.double_ml_ate()

    print(f"ATEs (CI) for the simulated dowhy dataset where the ground truth is {ground_truth_ate}:")
    print(
        "Outcome regression ATE:",
        outcome_regression_ate,
        "| CI:",
        outcome_regression_ate_ci,
    )
    print("IPTW ATE:", IPTW_ate, "| CI:", IPTW_ate_ci)
    print("AIPTW ATE:", AIPTW_ate, "| CI:", AIPTW_ate_ci)
    print("Double ML ATE:", double_ml_ate, "| CI:", double_ml_ate_ci)

    # report for inference time
    report = ci_model.pred_model_report()
