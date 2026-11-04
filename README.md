mbjImOZHMbq46vsX

# Term Deposit Subscription Prediction

Predicts whether a bank customer will subscribe to a term deposit, using data from a European bank's phone marketing campaign. Only 7.2% of the customers in the data subscribed, so the model is built to find as many of them as possible while keeping the number of calls reasonable.

## Data

The data has 40,000 customers and 13 features: customer details (age, job, marital status, and education), account details (default, balance, housing loan, and personal loan), and call details (contact type, date and duration of the last call, and number of calls). The target, `y`, is whether the customer subscribed.

## Approach

1. The data was split into 32,000 training customers and 8,000 test customers, stratified on `y` so both sets keep the 7.2% subscription rate. The test set was used once, at the end.
2. A LazyPredict screen of 22 classifiers on one validation split put logistic regression, random forest, and XGBoost in its top five.
3. These three, plus hist gradient boosting, were compared with 5-fold cross-validation, each with no imbalance fix, with class weights, and with SMOTE. Class weights gave the highest recall at the default 0.5 threshold.
4. Each class-weighted model was tuned with a randomized search of 20 combinations of settings, scored on PR-AUC. XGBoost scored highest (0.553).
5. The decision threshold was chosen to maximize the F2 score, which counts recall twice as much as precision. A missed subscriber is a lost sale, and an unnecessary call costs a few minutes of an agent's time. The chosen threshold is 0.493.

## Results

The final model is XGBoost with class weights and a decision threshold of 0.493.

| Metric | Always "no" (cross-validation) | Final model (cross-validation) | Final model (test set) |
|---|---|---|---|
| Accuracy | 0.928 | 0.888 | 0.890 |
| Recall | 0.000 | 0.882 | 0.883 |
| Precision | 0.000 | 0.383 | 0.386 |
| PR-AUC | 0.072 | 0.553 | 0.578 |
| ROC-AUC | 0.500 | 0.949 | 0.952 |

Cross-validation scores are averages over 5 folds of the training set, with the threshold chosen inside each fold.

The model meets the 81% accuracy target, with 88.8% in 5-fold cross-validation and 89.0% on the test set. A model that always predicts "no" scores higher, 92.8%, and finds no subscribers, so the models were compared on PR-AUC and recall.

On the test set, the model calls 1,324 of the 8,000 customers (16.6%) and finds 511 of the 579 subscribers (88.3%). Of its calls, 38.6% reach a subscriber, compared with 7.2% if every customer were called.

## Setup

From the repository root, create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate term-deposit-subscription-prediction
```

The dataset is not included in this repository. If you have it, save it as `data/raw/term-deposit-marketing-2020.csv` before running the notebooks.

## Usage

Run the notebooks in order, with the `term-deposit-subscription-prediction` environment as the kernel:

1. `notebooks/exploration/01_eda.ipynb` checks the data and shows how each feature relates to subscribing.
2. `notebooks/modeling/01_model_development.ipynb` builds and tests the model and saves it to `models/`. It takes about 15 to 20 minutes.

The saved file holds the whole pipeline, including the preprocessing and the decision threshold. To load it from the repository root:

```python
import joblib

from src.utils.config import FINAL_MODEL_PATH

model = joblib.load(FINAL_MODEL_PATH)
```

`model.predict(X)` takes a DataFrame with the 13 feature columns and returns 1 for customers predicted to subscribe and 0 for the rest. `model.predict_proba(X)[:, 1]` returns each customer's score. Load the model with the library versions listed in `models/final_model.json`, since scikit-learn does not support loading a model saved with other versions. Only load the file from a trusted source, because joblib can run code when it loads a file.

## Project structure

```
├── data/
│   └── raw/                        # the CSV goes here (not included)
├── models/                         # final_model.joblib and final_model.json
├── notebooks/
│   ├── exploration/
│   │   └── 01_eda.ipynb
│   └── modeling/
│       └── 01_model_development.ipynb
├── reports/
│   └── figures/                    # charts saved by the notebooks
├── src/
│   ├── data/
│   │   └── load_data.py            # loads and checks the raw data
│   ├── models/
│   │   ├── evaluate.py             # metrics, call outcomes, and gains table
│   │   └── train.py                # pipelines, tuning, threshold, and saving
│   └── utils/
│       └── config.py               # paths and settings
├── environment.yml
└── README.md
```
