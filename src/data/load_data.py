import pandas as pd

from src.utils.config import FEATURE_COLS, RAW_DATA_PATH, TARGET_COL


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw survey CSV and validate it before use.
    
    Parameters
    ----------
    path : str or Path, optional
        Location of the CSV file. Defaults to RAW_DATA_PATH from config.
    
    Returns
    -------
    pandas.DataFrame
        The raw survey data, unmodified, with columns Y, X1-X6.
    
    Raises
    ------
    ValueError
        If any of the expected columns (Y, X1-X6) are missing, or if
        the data contains any null values..  
    """
    df = pd.read_csv(path)

    expected_cols = FEATURE_COLS + [TARGET_COL]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isnull().values.any():
        raise ValueError("Raw data contains missing values.")

    unexpected = set(df[TARGET_COL].unique()) - {"yes", "no"}
    if unexpected:
        raise ValueError(f"Unexpected target values: {unexpected}")
    
    return df


def split_features_target(df: pd.DataFrame):
    """
    Split a dataframe into the feature matrix and target vector.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing at least the columns listed in
        FEATURE_COLS and TARGET_COL (from config).

    Returns
    -------
    X : pandas.DataFrame
        The six feature columns, X1-X6.
    y : pandas.Series
        The target column, Y (0 = unhappy, 1 = happy).
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].map({"yes": 1, "no": 0})
    return X, y
