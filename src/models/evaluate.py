import pandas as pd
from sklearn.metrics import (
    check_scoring,
    confusion_matrix,
    make_scorer,
    precision_score,
)
from sklearn.model_selection import cross_validate

SCORING = {
    "accuracy": "accuracy",
    # Precision divides by the number of "yes" predictions, which is 0 for a model
    # that never predicts "yes", like the baseline. zero_division=0 scores that
    # case as 0 without a warning.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
}


def cv_summary(model, X, y, cv):
    """Return the mean cross-validation score for each metric.

    The metrics are the ones in SCORING.

    Parameters
    ----------
    model : estimator
        Model or pipeline to cross-validate. A new copy is trained on each fold.
    X : pandas.DataFrame
        Features.
    y : pandas.Series
        Target, 1 for "yes" and 0 for "no".
    cv : cross-validation splitter
        Splitter that defines the folds, such as StratifiedKFold.

    Returns
    -------
    pandas.Series
        Mean score over the folds, indexed by metric name.
    """
    scores = cross_validate(model, X, y, cv=cv, scoring=SCORING)
    return pd.Series({metric: scores[f"test_{metric}"].mean() for metric in SCORING})


def score_models(models, X, y):
    """Score fitted models on the same data with every metric.

    The metrics are the ones in SCORING.

    Parameters
    ----------
    models : dict
        Maps each name to a fitted model or pipeline.
    X : pandas.DataFrame
        Features to score on.
    y : pandas.Series
        Target, 1 for "yes" and 0 for "no".

    Returns
    -------
    pandas.DataFrame
        One row per model and one column per metric.
    """
    scorer = check_scoring(scoring=SCORING)
    scores = {name: scorer(model, X, y) for name, model in models.items()}
    return pd.DataFrame.from_dict(scores, orient="index")


def call_outcomes(y_true, y_pred):
    """Count the calls a set of predictions makes and the subscribers it finds.

    Parameters
    ----------
    y_true : array-like
        True target values, 1 for "yes" and 0 for "no".
    y_pred : array-like
        Predicted values, where 1 or True means the customer is called.

    Returns
    -------
    dict
        Counts and percentages of calls made, subscribers found and missed, and
        unnecessary calls made and avoided.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    customers = tn + fp + fn + tp
    subscribers = tp + fn
    non_subscribers = tn + fp
    calls = tp + fp
    return {
        "calls made": calls,
        "calls made (% of customers)": 100 * calls / customers,
        "subscribers found": tp,
        "subscribers found (% of subscribers)": 100 * tp / subscribers,
        "subscribers missed": fn,
        "subscribers missed (% of subscribers)": 100 * fn / subscribers,
        "unnecessary calls": fp,
        "unnecessary calls (% of non-subscribers)": 100 * fp / non_subscribers,
        "unnecessary calls avoided": tn,
        "unnecessary calls avoided (% of non-subscribers)": 100 * tn / non_subscribers,
        "calls reaching a subscriber (% of calls made)": (
            100 * tp / calls if calls else 0.0
        ),
    }


def threshold_table(y_true, y_score, thresholds):
    """Return the call outcomes at each decision threshold.

    Parameters
    ----------
    y_true : array-like
        True target values, 1 for "yes" and 0 for "no".
    y_score : numpy.ndarray
        Predicted "yes" score for each customer.
    thresholds : list of float
        Decision thresholds to compare. A customer is called when their score is
        at or above the threshold.

    Returns
    -------
    pandas.DataFrame
        One row per threshold, with the columns returned by call_outcomes.
    """
    rows = {}
    for threshold in thresholds:
        rows[threshold] = call_outcomes(y_true, y_score >= threshold)
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("threshold")


def gains_table(y_true, y_score, shares=(10, 20, 30, 50)):
    """Return the subscribers found by calling the top-scored share of customers.

    Parameters
    ----------
    y_true : array-like
        True target values, 1 for "yes" and 0 for "no".
    y_score : numpy.ndarray
        Predicted "yes" score for each customer, in the same order as y_true.
    shares : tuple of int, default=(10, 20, 30, 50)
        Percentages of customers to call, starting from the highest score.

    Returns
    -------
    pandas.DataFrame
        One row per share, with the calls made, the subscribers found, and the
        subscribers found as a percentage of all subscribers.
    """
    ranked = pd.DataFrame({"subscribed": y_true, "score": y_score})
    ranked = ranked.sort_values("score", ascending=False)
    total = ranked["subscribed"].sum()
    rows = {}
    for share in shares:
        calls = round(len(ranked) * share / 100)
        found = ranked["subscribed"].head(calls).sum()
        rows[share] = {
            "calls made": calls,
            "subscribers found": found,
            "subscribers found (%)": 100 * found / total,
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.rename_axis("customers called (%)")
