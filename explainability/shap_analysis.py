"""
shap_analysis.py
-----------------
Deliverable 2: Model explainability.

Loads the trained pipeline, computes SHAP values on a held-out sample,
and writes:
  - explainability/shap_summary.png   (global feature importance)
  - explainability/shap_report.json   (mean |SHAP| per feature, ranked)

Run locally (needs a local data.csv and a trained model.joblib -- both
of which are git-ignored build/data artifacts, so run this after
`python training/train.py` has produced model/model.joblib):

    python explainability/shap_analysis.py \
        --model model/model.joblib \
        --data-path data/data.csv \
        --out-dir explainability

The printed/saved ranking is what the plain-English writeup in the
README's "Explainability findings" section is based on: the features
with the SMALLEST mean |SHAP| value are the ones with the least impact
on the prediction.
"""

import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap

FEATURE_COLUMNS = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/model.joblib")
    parser.add_argument("--data-path", default="data/data.csv")
    parser.add_argument("--out-dir", default="explainability")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pipeline = joblib.load(args.model)
    df = pd.read_csv(args.data_path).dropna(subset=["target"])
    X = df[FEATURE_COLUMNS].copy()
    if len(X) > args.sample_size:
        X = X.sample(args.sample_size, random_state=42)

    # Transform through every step except the final classifier, then
    # explain the linear classifier on the transformed (numeric) space.
    pre = pipeline[:-1]
    clf = pipeline.named_steps["clf"]
    X_transformed = pre.transform(X)

    explainer = shap.LinearExplainer(clf, X_transformed)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = FEATURE_COLUMNS

    mean_abs_shap = pd.Series(
        abs(shap_values.values).mean(axis=0), index=FEATURE_COLUMNS
    ).sort_values(ascending=False)

    report = {
        "ranked_feature_importance_mean_abs_shap": mean_abs_shap.to_dict(),
        "least_impactful_features": mean_abs_shap.tail(3).index.tolist(),
        "most_impactful_features": mean_abs_shap.head(3).index.tolist(),
    }
    with open(os.path.join(args.out_dir, "shap_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    plt.figure()
    shap.summary_plot(shap_values, X_transformed, feature_names=FEATURE_COLUMNS,
                       plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "shap_summary.png"), dpi=150)

    print(json.dumps(report, indent=2))
    print(
        "\nPlain-English summary: the features with the smallest mean |SHAP| "
        f"value — {report['least_impactful_features']} — contribute the "
        "least to the model's predictions. Their SHAP values cluster near "
        "zero across patients, meaning changing these values has little "
        "effect on the predicted probability of heart disease relative to "
        f"the top drivers, {report['most_impactful_features']}."
    )


if __name__ == "__main__":
    main()
