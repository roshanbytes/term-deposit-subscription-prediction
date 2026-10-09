from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "term-deposit-marketing-2020.csv"

TARGET_COL = "y"
FEATURE_COLS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
]

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET_ACCURACY = 0.81
