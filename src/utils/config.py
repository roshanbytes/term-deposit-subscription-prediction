from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "term-deposit-marketing-2020.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"

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
VALIDATION_SIZE = 0.2
CV_FOLDS = 5

TUNING_METRIC = "average_precision"
TUNING_ITERATIONS = 20
THRESHOLD_BETA = 2

TARGET_ACCURACY = 0.81
