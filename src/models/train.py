import json
import platform
from importlib.metadata import version

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, TunedThresholdClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.utils.config import (
    FINAL_MODEL_PATH,
    RANDOM_STATE,
    THRESHOLD_BETA,
    TUNING_ITERATIONS,
    TUNING_METRIC,
)

# Ranges searched by tune_model. The keys match the names in build_models, and
# "model__" sends each setting to the pipeline's "model" step.
PARAM_DISTRIBUTIONS = {
    "Logistic regression": {"model__C": loguniform(1e-3, 100)},
    "Random forest": {
        "model__max_depth": [None, 10, 20],
        "model__min_samples_leaf": [1, 2, 5, 10, 20, 50],
        "model__max_features": ["sqrt", 0.3, 0.5, 0.7],
    },
    "Hist gradient boosting": {
        "model__learning_rate": loguniform(0.01, 0.3),
        "model__max_iter": [100, 200, 400],
        "model__max_leaf_nodes": [15, 31, 63, 127],
        "model__min_samples_leaf": [5, 10, 20, 50, 100],
        "model__l2_regularization": loguniform(1e-3, 100),
    },
    "XGBoost": {
        "model__n_estimators": [200, 400, 800, 1600],
        "model__learning_rate": loguniform(0.005, 0.3),
        "model__max_depth": [3, 4, 6, 8, 10],
        "model__min_child_weight": [1, 5, 10, 20, 50],
        "model__subsample": [0.5, 0.7, 0.9, 1.0],
        "model__colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    },
}


def build_preprocessor(categorical_cols, numeric_cols):
    """Build the preprocessing step that every model uses.

    Parameters
    ----------
    categorical_cols : list of str
        Columns to one-hot encode.
    numeric_cols : list of str
        Columns to scale to a mean of 0 and a standard deviation of 1.

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Unfitted preprocessor.
    """
    return ColumnTransformer(
        [
            (
                "categorical",
                # HistGradientBoostingClassifier only accepts dense input.
                OneHotEncoder(
                    drop="if_binary", handle_unknown="ignore", sparse_output=False
                ),
                categorical_cols,
            ),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )


def build_models(y_train, class_weighted=True):
    """Return the four candidate models, with or without class weights.

    Parameters
    ----------
    y_train : pandas.Series
        Training target, used to set XGBoost's scale_pos_weight.
    class_weighted : bool, default=True
        If True, the "yes" rows count more during training. If False, every row
        counts the same.

    Returns
    -------
    dict
        Maps each model name to an unfitted model.
    """
    if class_weighted:
        weight = "balanced"
        # XGBoost has no class_weight option. scale_pos_weight multiplies the
        # weight of every "yes" row, and the ratio of "no" to "yes" rows gives the
        # same balance as "balanced".
        positive_weight = (y_train == 0).sum() / (y_train == 1).sum()
    else:
        weight = None
        positive_weight = 1.0

    return {
        "Logistic regression": LogisticRegression(
            class_weight=weight, random_state=RANDOM_STATE
        ),
        "Random forest": RandomForestClassifier(
            class_weight=weight, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Hist gradient boosting": HistGradientBoostingClassifier(
            class_weight=weight, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=positive_weight, random_state=RANDOM_STATE
        ),
    }


def build_pipeline(model, categorical_cols, numeric_cols, smote=False):
    """Put the preprocessing in front of a model, with optional SMOTE between them.

    Parameters
    ----------
    model : estimator
        Unfitted model, such as one from build_models.
    categorical_cols : list of str
        Columns to one-hot encode.
    numeric_cols : list of str
        Columns to scale.
    smote : bool, default=False
        If True, SMOTE runs between the preprocessing and the model.

    Returns
    -------
    sklearn.pipeline.Pipeline or imblearn.pipeline.Pipeline
        Unfitted pipeline with the steps "preprocess", "smote" (only if used), and
        "model".
    """
    preprocessor = build_preprocessor(categorical_cols, numeric_cols)
    if smote:
        # scikit-learn's Pipeline cannot hold a step that adds rows, so SMOTE needs
        # imbalanced-learn's Pipeline. It runs SMOTE only during fit, so the
        # validation part of each fold is scored on real rows.
        return ImbPipeline(
            [
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", model),
            ]
        )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def tune_model(name, pipeline, X, y, cv):
    """Find the model settings with the best score using a randomized search.

    The search tries TUNING_ITERATIONS combinations from PARAM_DISTRIBUTIONS[name]
    and scores each one with TUNING_METRIC.

    Parameters
    ----------
    name : str
        Model name, used to look up its ranges in PARAM_DISTRIBUTIONS.
    pipeline : sklearn.pipeline.Pipeline
        Unfitted pipeline from build_pipeline.
    X : pandas.DataFrame
        Training features.
    y : pandas.Series
        Training target.
    cv : cross-validation splitter
        Splitter that defines the folds.

    Returns
    -------
    sklearn.model_selection.RandomizedSearchCV
        Fitted search. It is refit on all of X and y with the best settings, so
        best_estimator_ is ready to use.
    """
    search = RandomizedSearchCV(
        pipeline,
        PARAM_DISTRIBUTIONS[name],
        n_iter=TUNING_ITERATIONS,
        scoring=TUNING_METRIC,
        cv=cv,
        random_state=RANDOM_STATE,
    )
    return search.fit(X, y)


def tune_threshold(model, X, y, cv):
    """Choose the decision threshold with the highest F-beta score.

    Beta is THRESHOLD_BETA.

    Parameters
    ----------
    model : estimator
        Model or pipeline whose threshold is chosen.
    X : pandas.DataFrame
        Training features.
    y : pandas.Series
        Training target.
    cv : cross-validation splitter
        Splitter that defines the folds.

    Returns
    -------
    sklearn.model_selection.TunedThresholdClassifierCV
        Fitted model with the chosen threshold in best_threshold_. It is refit on
        all of X and y, and it predicts "yes" for scores at or above the threshold.
    """
    scorer = make_scorer(fbeta_score, beta=THRESHOLD_BETA)
    tuner = TunedThresholdClassifierCV(model, scoring=scorer, cv=cv)
    return tuner.fit(X, y)


def save_model(model, info, path=FINAL_MODEL_PATH):
    """Save a fitted model, plus a JSON file of its details and library versions.

    Parameters
    ----------
    model : estimator
        Fitted model to save with joblib.
    info : dict
        Details to save with the model, such as its settings and scores.
    path : pathlib.Path, default=FINAL_MODEL_PATH
        Where to save the model. The JSON file goes next to it, with the same name
        and a .json extension.
    """
    path.parent.mkdir(exist_ok=True)
    joblib.dump(model, path)
    versions = {
        "python": platform.python_version(),
        "numpy": version("numpy"),
        "scikit-learn": version("scikit-learn"),
        "xgboost": version("xgboost"),
    }
    record = {**info, "versions": versions}
    # default=float writes NumPy numbers, such as the float32 threshold, as floats.
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, default=float))
