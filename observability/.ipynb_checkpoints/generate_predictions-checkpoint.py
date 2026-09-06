"""
generate_predictions.py
------------------------
Deliverable 5: per-sample prediction + logging/observability.

1. Generates a 100-row random dataset, sampled from realistic
   per-feature ranges (min/max derived from the training data, taken as
   a CLI argument so no dataset file needs to be committed alongside
   this script).
2. Sends each row to the deployed API's /predict endpoint.
3. Logs every request/response pair locally (JSON Lines) AND relies on
   the API itself to emit the authoritative structured log line to
   stdout -> Cloud Logging (see app/main.py). This script's local log
   is a convenience copy for the drift-detection step and for the
   stress-test replay in Deliverable 6.

Usage:
    python -m observability.generate_predictions \
        --api-url http://<EXTERNAL_IP>/predict \
        --reference model/train_reference.csv \
        --n 100 \
        --out observability/generated_predictions.jsonl \
        --sample-csv observability/sample_100.csv
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

FEATURE_COLUMNS = [
    "age", "gender", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
CATEGORICAL = {"cp": [0, 1, 2, 3], "fbs": [0, 1], "restecg": [0, 1, 2],
               "exang": [0, 1], "slope": [0, 1, 2], "ca": [0, 1, 2, 3],
               "thal": [1, 2, 3]}


def generate_random_samples(reference_df: pd.DataFrame, n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = {}
        for col in FEATURE_COLUMNS:
            if col == "gender":
                row[col] = rng.choice(["male", "female"])
            elif col in CATEGORICAL:
                row[col] = int(rng.choice(CATEGORICAL[col]))
            else:
                lo, hi = reference_df[col].min(), reference_df[col].max()
                val = rng.uniform(lo, hi)
                row[col] = round(float(val), 1)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True, help="Full URL to the /predict endpoint")
    parser.add_argument("--reference", default="model/train_reference.csv",
                         help="CSV of training feature ranges (produced by training/train.py)")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default="observability/generated_predictions.jsonl")
    parser.add_argument("--sample-csv", default="observability/sample_100.csv")
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    reference_df = pd.read_csv(args.reference)
    samples = generate_random_samples(reference_df, args.n)
    samples.to_csv(args.sample_csv, index=False)

    with open(args.out, "w") as f:
        for i, row in samples.iterrows():
            payload = row.to_dict()
            start = time.time()
            try:
                resp = requests.post(args.api_url, json=payload, timeout=args.timeout)
                resp.raise_for_status()
                result = resp.json()
                status = "ok"
            except Exception as exc:  # noqa: BLE001
                result = {"error": str(exc)}
                status = "error"
            latency_ms = (time.time() - start) * 1000

            record = {
                "row_index": int(i),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_features": payload,
                "status": status,
                "response": result,
                "latency_ms": round(latency_ms, 2),
            }
            f.write(json.dumps(record, default=str) + "\n")
            print(f"[{i+1}/{args.n}] status={status} latency_ms={record['latency_ms']}")

    print(f"\nWrote {args.n} request logs -> {args.out}")
    print(f"Wrote raw sample dataset -> {args.sample_csv}")
    print(
        "\nTo view these in Cloud Logging, run e.g.:\n"
        '  gcloud logging read \'resource.type="k8s_container" '
        'jsonPayload.event="prediction"\' --limit 20 --format json'
    )


if __name__ == "__main__":
    main()