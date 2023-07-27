import numpy as np
from scipy.sparse import find
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, QuantileTransformer, MaxAbsScaler
import gensim
from gensim.models.keyedvectors import FastTextKeyedVectors
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.calibration import calibration_curve
from catboost import CatBoostRegressor, CatBoostClassifier
from copy import deepcopy
import re


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # get rid of numbers
    return text


# define sklearn pipelines that are good for large-dimension sparse data
# TODO: shouldw we put in a kselector here?
REGRESSOR = Pipeline(
    [
        # (
        #     "normalizer",
        #     Normalizer(norm="l1"),
        # ),  # each row is summed to 1 -> avoids issues with diff document lengths for BOW
        (
            "model",
            SVR(
                kernel="linear", tol=1e-3
            ),  # SVR does not allow one to set a random state parameter (https://github.com/scikit-learn/scikit-learn/issues/17391)
        ),
    ]
)
REGRESSOR_GRID = {
    "model__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
}


CLASSIFIER = Pipeline(
    [
        # (
        #     "normalizer",
        #     Normalizer(norm="l1"),
        # ),  # each row is summed to 1 -> avoids issues with diff document lengths for BOW
        (
            "model",
            LogisticRegression(
                l1_ratio=0.1,
                solver="saga",
                max_iter=20000,
                tol=1e-3,  # hopefully this is OK. default is 1e-4
                penalty="elasticnet",
                dual=False,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]
)

CLASSIFIER_GRID = {
    "model__C": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
}


COUNTVECTORIZER = CountVectorizer(
    max_features=2000,
    lowercase=True,
    strip_accents="unicode",
    stop_words="english",
    max_df=0.9,
    min_df=5,
    binary=True,
    preprocessor=preprocess_text,  # remove numbers
)


def set_np_random_seed(seed=0):
    return np.random.default_rng(seed)


def get_cv(
    y_learning_task,
    n_splits=4,
    shuffle_split=False,
    shuffle_split_train_size=0.75,
    shuffle_split_test_size=0.25,
    random_state=0,
):
    """
    Get a cross-validation object for the given learning task

    Args:
        y_learning_task (str): "binary", "multiclass", or "regression"
        n_splits (int): number of folds, defaults to 4
        shuffle_split (bool): whether to use a shuffle split
        shuffle_split_train_size (float): fraction of data to use for training
        shuffle_split_test_size (float): fraction of data to use for testing
        random_state (int): random state

    Returns:
        cv (sklearn.model_selection object): cross-validation object
    """

    # there is some wonkiness when you have exactly one split - that's a scikit learn issue
    n_splits = int(n_splits)
    assert n_splits >= 1
    if shuffle_split or n_splits == 1:
        if y_learning_task in {"binary", "multiclass"}:
            cv = StratifiedShuffleSplit(
                n_splits=n_splits,
                train_size=shuffle_split_train_size,
                test_size=shuffle_split_test_size,
                random_state=random_state,
            )
        else:
            cv = ShuffleSplit(
                n_splits=n_splits,
                train_size=shuffle_split_train_size,
                test_size=shuffle_split_test_size,
                random_state=random_state,
            )
    else:
        if y_learning_task in {"binary", "multiclass"}:
            cv = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits, shuffle=True, random_state=random_state)
    return cv


def get_default_scoring(y_learning_task):
    # this function defines the "default" scoring
    # that will be used to evaluate ML models
    if y_learning_task == "binary":
        return "roc_auc"
    elif y_learning_task == "multiclass":
        return "roc_auc_ovr_weighted"
    else:
        return "r2"  #'r2_score' is not a valid scoring value. Use sorted(sklearn.metrics.SCORERS.keys()) to get valid options.


class SIF(BaseEstimator, TransformerMixin):
    """SIF model
    A Simple but Tough-to-Beat Baseline for Sentence Embeddings by Arora et al

    SIF reference paper: https://openreview.net/pdf?id=SyK00v5xx
    SIF reference code: https://github.com/PrincetonML/SIF/blob/master/src/SIF_embedding.py

    Args:
        model_path (str): where the word vector file is - either a .bin file or a word2vec .vec format file
            Should be able to be loaded by gensim
        a (float): smoothness parameter for SIF
        vectorizer (sklearn vectorizer): vectorizer to use for the text.
            If not provided, a default term frequency one is used one is used

    We use word vector from fasttext: https://fasttext.cc/docs/en/english-vectors.html
    """

    def __init__(self, model_path, a=1e-4, vectorizer=None):
        self.model_path = model_path
        if self.model_path.endswith(".bin"):
            self.word_vectors = gensim.models.fasttext.load_facebook_vectors(self.model_path)
        else:
            self.word_vectors = gensim.models.KeyedVectors.load_word2vec_format(self.model_path)
        self.a = a
        if vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                strip_accents="unicode",
                lowercase=True,
                stop_words="english",
                max_df=0.9,
                min_df=5,
                norm=None,
                sublinear_tf=False,
                binary=False,
                use_idf=False,
                smooth_idf=False,
            )
        else:
            self.vectorizer = vectorizer

        self.pc = None
        self.training_word_weights = None
        self.inverse_vocab = None

    def fit_transform(self, text):
        """
        SIF is described by the authors as:
        "Use word embeddings computed using one of the popular methods on unlabeled corpus like Wikipedia,
        represent the sentence by a weighted average of the word vectors, and then modify them a bit using PCA/SVD."

        There is only one parameter: `a` which is the smoothness in the smooth inverse frequency (SIF) weighting.:
        a / (a + p(w)), where p(w) is the probability of seeing a word w in the training data.
        """
        # edge case where there are fewer texts than min_df in the vectorizer
        if self.vectorizer.min_df >= len(text):
            self.vectorizer.min_df = 0
            self.vectorizer.max_idf = 1.0

        vectorized_text = self.vectorizer.fit_transform(text)
        self.inverse_vocab = {v: k for k, v in self.vectorizer.vocabulary_.items()}

        training_word_counts = np.sum(vectorized_text, axis=0).A[0]
        training_word_weights = self.a / (self.a + training_word_counts / np.sum(training_word_counts))
        self.training_word_weights = {self.inverse_vocab[i]: val for i, val in enumerate(training_word_weights)}

        X_emb = self.get_emb(vectorized_text)
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0).fit(X_emb)

        self.pc = svd.components_

        X_emb -= X_emb.dot(self.pc.transpose()) * self.pc

        return X_emb

    def fit(self, text):
        self.fit_transform(text)
        return self

    def transform(self, text):
        vectorized_text = self.vectorizer.transform(text)
        X_emb = self.get_emb(vectorized_text)
        X_emb -= X_emb.dot(self.pc.transpose()) * self.pc

        return X_emb

    def get_emb(self, vectorized_text):
        """Function that converts vectorized text into embeddings

        Args:
            vectorized_text (sparse np matrix): sklearn vectorized text

        Returns:
            emb (np array): embeddings
        """
        n_samples = vectorized_text.shape[0]
        emb = np.zeros((n_samples, self.word_vectors.vector_size))
        for i in range(n_samples):
            _, word_inds, word_tfs = find(vectorized_text[i, :])
            word_vecs = self.get_word_vecs(
                word_inds, word_tfs, self.word_vectors, self.inverse_vocab, self.training_word_weights
            )
            if len(word_vecs) > 0:
                emb[i, :] = np.mean(word_vecs, axis=0)
        return emb

    @staticmethod
    def get_word_vecs(word_inds, word_tfs, word_vectors, inverse_vocab, training_word_weights):
        """Get word vectors for a single text given its indices and term frequencies

        Args:
            word_inds (list[int]): list of word indices
            word_tfs (list[int]): list of term frequencies
            word_vectors (FastTextKeyedVectors): gensim vectors
            inverse_vocab (dict[int, str]): inverse vocabulary mapping
            training_word_weights (dict[str, float]): word probabilities

        Returns:
            word_vecs (list[np.array]): list of word vectors for the input indices
        """
        if type(word_vectors) == FastTextKeyedVectors:
            # can do OOV
            word_vecs = [
                tf * word_vectors[inverse_vocab[j]] * training_word_weights[inverse_vocab[j]]
                for j, tf in zip(word_inds, word_tfs)
            ]
        else:
            word_vecs = [
                tf * word_vectors[inverse_vocab[j]] * training_word_weights[inverse_vocab[j]]
                for j, tf in zip(word_inds, word_tfs)
                if inverse_vocab[j] in word_vectors
            ]
        return word_vecs


def predict_y_from_X_crossfit(
    X,
    y,
    y_learning_task="binary",
    n_crossfit=4,
    n_innercv=4,
    n_jobs=1,
    random_state=0,
):
    """
    WITH CROSS-FITTING: Assess how much predictive power there is in X or y.
    Simple classification/regression methods using sklearn framework.
    Using default C unless cv is specified as some k > 1.

    Arguments:
        X, y -- sklearn formatted data for ML
        y_learning_task -- 'binary', 'multiclass' or 'regression'.
        n_crossfit (int): number of folds for outer cross-fitting
        n_innercv (int): number of folds for inner cross-validation
        n_jobs -- number of cores to use for cross-validation
        random_state -- random seed for cross-validation


    Returns:
        AUC on X_test, y_test if classification or
        R^2 on X_test, y_test if regression
    """
    y_predictions = np.zeros(len(y))
    scoring_y = get_default_scoring(y_learning_task)
    crossfit_split = list(StratifiedKFold(n_splits=n_crossfit, shuffle=True, random_state=random_state).split(X, y))
    for train_inds, test_inds in crossfit_split:
        X_train = X[train_inds]
        X_test = X[test_inds]
        y_train = y[train_inds]

        if y_learning_task in {"binary", "multiclass"}:
            pipeline = CLASSIFIER
            param_grid = CLASSIFIER_GRID
        else:
            pipeline = REGRESSOR
            param_grid = REGRESSOR_GRID

        if n_innercv is None:
            model = pipeline
            if "n_jobs" in model.get_params():
                model.set_params(n_jobs=n_jobs)
            elif "model__n_jobs" in model.get_params():
                model.set_params(model__n_jobs=n_jobs)
        else:
            model = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=get_cv(y_learning_task, n_splits=n_innercv, random_state=random_state),
                scoring=scoring_y,
                n_jobs=n_jobs,
                refit=True,
            )
        model.fit(X_train, y_train)
        y_predictions[test_inds] = model.predict(X_test)
        if y_learning_task in {"binary", "multiclass"}:
            y_predictions[test_inds] = model.predict_proba(X_test)[:, 1]
        else:
            y_predictions[test_inds] = model.predict(X_test)

    # calculate the score across all predictions across all folds
    scores = {}
    # scorer = get_scorer(scoring_y)._score_func
    # scores[scoring_y] = scorer(y, y_predictions)

    # for classification print out some more scores on the inference data
    if y_learning_task == "binary":
        scores = scores_from_fit_classifier(y, y_predictions)
    return scores, y_predictions


def scores_from_fit_classifier(y, y_predictions):
    scores = {}
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
    y_pred_soft = deepcopy(y_predictions)
    y_pred_hard = (y_pred_soft > 0.5).astype(int)
    assert y_pred_soft.shape == y_pred_hard.shape == y.shape
    for metric_str, metric_func in class_hard_name2metric_func.items():
        scores[metric_str] = metric_func(y, y_pred_hard)
    for metric_str, metric_func in class_scores_name2metric_func.items():
        scores[metric_str] = metric_func(y, y_pred_soft)
    return scores


def mean_predictions(dummy, y_pred):
    """
    Helpful for error diagnosing. Returns the mean of the predcted values

    Args:
        - dummy : we need this arg so that this function looks the same as
        sklearn error metrics that take inputs y_true, y_pred
        - y_pred : np.array
    """
    return np.mean(y_pred)


def calibration_rmse(y_true, y_pred):
    """
    Calculates calibration root mean squared error (RMSE).

    Calibration is the extent to which a model's probabilistic predictions match their
    corresponding empirical frequencies.

    See Nguyen and O'Connor 2015's introduction and definitions
    https://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP182.pdf
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy="uniform")
    rms = mean_squared_error(prob_true, prob_pred, squared=False)  # False returns RMSE vs MSE
    return rms


def mean_truth(y_true, dummy):
    """
    Helpful for error diagnosing. Returns the mean of the true values

    Args:
        - y_true : np.array
        - dummy : we need this arg so that this function looks the same as
        sklearn error metrics that take inputs y_true, y_pred
    """
    return np.mean(y_true)
