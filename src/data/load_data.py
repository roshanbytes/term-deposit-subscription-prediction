import pandas as pd

from src.utils.config import FEATURE_COLS, RAW_DATA_PATH, TARGET_COL


def load_raw_data(path=RAW_DATA_PATH):
    """Load the raw term deposit data and check it before use.

    Parameters
    ----------
    path : str or pathlib.Path, default=RAW_DATA_PATH
        Location of the CSV file.

    Returns
    -------
    pandas.DataFrame
        The raw data, unchanged.

    Raises
    ------
    ValueError
        If a column in FEATURE_COLS or TARGET_COL is missing, if any value is
        missing, or if the target has values other than "yes" and "no".
    """
    df = pd.read_csv(path)

    expected_cols = FEATURE_COLS + [TARGET_COL]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isna().to_numpy().any():
        raise ValueError("Raw data contains missing values.")

    unexpected = set(df[TARGET_COL].unique()) - {"yes", "no"}
    if unexpected:
        raise ValueError(f"Unexpected target values: {unexpected}")

    return df


def split_features_target(df):
    """Split the data into features and target.

    Parameters
    ----------
    df : pandas.DataFrame
        Data with the columns in FEATURE_COLS and TARGET_COL.

    Returns
    -------
    X : pandas.DataFrame
        The columns in FEATURE_COLS.
    y : pandas.Series
        The target, 1 for "yes" and 0 for "no".
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].map({"yes": 1, "no": 0})
    return X, y
