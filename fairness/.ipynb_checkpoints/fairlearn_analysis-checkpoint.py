"""
fairlearn_analysis.py
----------------------
Deliverable 3: Fairness testing with Fairlearn.

The deliverable text specifies the sensitive attribute as "age" (note:
the pipeline-overview section of the same problem statement mentions
"gender" instead -- this script evaluates BOTH and clearly labels which
one satisfies the literal deliverable text, so either reading of the
brief is covered).

Age is binned into clinically-conventional groups since Fairlearn's
group-fairness metrics expect a small number of discrete groups, not a
continuous variable.

Outputs:
  - fairness/fairness_report.json  (MetricFrame breakdown + disparity
    metrics: demographic parity difference/ratio, equalized odds
    difference, per-group accuracy/selection rate)

Run after training (needs data/data.csv + model/model.joblib, both
git-ignored):

    python -m fairness.fairlearn_analysis \
        --model model/model.joblib --data-path data/data.csv
"""

import argparse
import json
import os

import joblib
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    selection_rate,
)
from sklearn.metrics import accuracy_score, f1_score

FEATURE_COLUMNS = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def bin_age(age_series: pd.Series) -> pd.Series:
    bins = [0, 40, 50, 60, 200]
    labels = ["<40", "40-49", "50-59", "60+"]
    return pd.cut(age_series, bins=bins, labels=labels, right=False)


def evaluate(y_true, y_pred, sensitive_features, label: str) -> dict:
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection_rate": selection_rate,
                 "f1": f1_score},
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features,
    )
    return {
        "sensitive_attribute": label,
        "by_group": mf.by_group.to_dict(),
        "overall": mf.overall.to_dict(),
        "demographic_parity_difference": demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features),
        "demographic_parity_ratio": demographic_parity_ratio(
            y_true, y_pred, sensitive_features=sensitive_features),
        "equalized_odds_difference": equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/model.joblib")
    parser.add_argument("--data-path", default="data/data.csv")
    parser.add_argument("--out", default="fairness/fairness_report.json")
    args = parser.parse_args()

    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.data_path).dropna(subset=["target"])
    df["target"] = df["target"].map({"yes": 1, "no": 0}).astype(int)

    X = df[FEATURE_COLUMNS].copy()
    y_true = df["target"].values
    y_pred = pipeline.predict(X)

    age_groups = bin_age(df["age"])
    age_report = evaluate(y_true, y_pred, age_groups, "age (binned)")

    gender_report = evaluate(y_true, y_pred, df["gender"], "gender")

    combined = {
        "note": (
            "Deliverable text specifies 'age' as the sensitive attribute; "
            "the pipeline overview mentions 'gender'. Both are reported "
            "below for completeness."
        ),
        "age_analysis": age_report,
        "gender_analysis": gender_report,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(json.dumps(combined, indent=2, default=str))


if __name__ == "__main__":
    main()