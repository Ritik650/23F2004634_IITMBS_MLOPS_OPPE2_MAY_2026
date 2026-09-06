# 23F2004634_IITMBS_MLOPS_OPPE2_MAY_2026

Production deployment of a heart-disease prediction model on GKE, built for
OPPE-2 (MLOps, May 2026).

## Repo layout & what each file is for

| Path | Purpose |
|---|---|
| `training/train.py` | Loads data, imputes/encodes/scales, tunes a `LogisticRegression` via `RandomizedSearchCV`, saves a single bundled `Pipeline` artifact + a reference-distribution CSV used later for drift detection. |
| `app/main.py` | FastAPI service (`/predict`, `/health`, `/ready`). Logs every prediction as one structured JSON line to stdout for Cloud Logging (Deliverable 5). |
| `Dockerfile` | Multi-stage build: trains the model inside the image (so no `.joblib`/data files are ever committed), then copies just the trained artifact + app code into a slim runtime image. |
| `requirements.txt` | Python deps for both training and serving. |
| `explainability/shap_analysis.py` | Deliverable 2 — SHAP global feature importance + plain-English ranking of least/most impactful features. |
| `fairness/fairlearn_analysis.py` | Deliverable 3 — Fairlearn `MetricFrame` fairness audit with `age` (as the deliverable specifies) and `gender` (as the pipeline-overview section separately mentions) as sensitive attributes. |
| `k8s/deployment.yaml` | GKE Deployment: resource requests/limits, readiness/liveness probes. |
| `k8s/service.yaml` | `LoadBalancer` Service exposing the API on an external IP. |
| `k8s/hpa.yaml` | `HorizontalPodAutoscaler`, capped at **3 pods max** (Deliverable 4 requirement), CPU-utilization target 60%. |
| `.github/workflows/cicd.yaml` | GitHub Actions: build image → push to Artifact Registry → `kubectl apply` to GKE on every push to `main`. |
| `observability/generate_predictions.py` | Deliverable 5 — generates the 100-row random dataset and drives it through the live `/predict` endpoint, logging each request/response locally as well. |
| `observability/drift_detection.py` | Deliverable 7 — KS-test (numeric) / chi-square (categorical) drift comparison between training reference and the 100-row generated set. |
| `stress_test/post.lua` + `stress_test/stress_test.sh` | Deliverable 6 — `wrk` load test driver, configured for >2,000 concurrent connections against `/predict`. |
| `.gitignore` / `.dockerignore` | Excludes model binaries, datasets, and local run artifacts from git, per the assignment's repo-content rules. |

**Not included in this repo (by design, per the assignment rules):** `data.csv` /
any training split, the trained `model.joblib`, and any other binary
execution output. The Dockerfile trains the model itself at build time from
source, so the image is fully reproducible without committing those files.

## One-time setup

1. **Point training at the real dataset.** Edit `DEFAULT_DATA_URL` in
   `training/train.py` (or pass `--build-arg DATA_URL=...` to `docker build`)
   to the actual raw-file path for `data.csv` inside
   `https://github.com/IITMBSMLOps/MLOPS_MAY_2026_OPPE2` (main branch).
   Verify the exact path/filename in that repo before running CI — it's
   left as a one-line constant rather than hardcoded elsewhere.
2. **GCP resources** (one-time, from Cloud Shell / your `oppe2-instance`):
   ```bash
   gcloud services enable artifactregistry.googleapis.com container.googleapis.com

   gcloud artifacts repositories create heart-disease-repo \
     --repository-format=docker --location=us-central1

   gcloud container clusters create-auto heart-disease-cluster \
     --region=us-central1
   ```
3. **GitHub Actions secrets** (repo Settings → Secrets and variables →
   Actions):
   - `GCP_PROJECT_ID`
   - `GCP_REGION` (e.g. `us-central1`)
   - `GKE_CLUSTER_NAME` (e.g. `heart-disease-cluster`)
   - `GKE_ZONE` (region or zone matching the cluster)
   - `GCP_SA_KEY` — JSON key for a service account with
     `roles/artifactregistry.writer` and `roles/container.developer`.
4. Add collaborator access for `IITMBSMLOps` / `da5014_1@study.iitm.ac.in`
   and confirm the invite shows as **accepted** (not pending) before the
   session ends.

## Running each deliverable

```bash
# Local venv for anything you run outside Docker (explainability, fairness,
# drift, generating the 100-row set)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Deliverable 4: build & run locally to sanity check before pushing
docker build -t heart-disease-api:local .
docker run -p 8080:8080 heart-disease-api:local
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d '{
  "age": 57, "gender": "male", "cp": 2, "trestbps": 140, "chol": 240,
  "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.2,
  "slope": 1, "ca": 0, "thal": 2
}'

# Push to main -> GitHub Actions builds, pushes, and deploys to GKE
git push origin main
kubectl get svc heart-disease-api-svc   # grab EXTERNAL-IP once assigned

# Deliverable 2: explainability (needs a local data.csv + a model you
# trained locally, since neither is committed to git)
python -m training.train --data-path data/data.csv
python -m explainability.shap_analysis --data-path data/data.csv

# Deliverable 3: fairness
python -m fairness.fairlearn_analysis --data-path data/data.csv

# Deliverable 5: 100-row generation + per-sample logging
python -m observability.generate_predictions \
  --api-url http://<EXTERNAL_IP>/predict \
  --reference model/train_reference.csv
# then inspect Cloud Logging:
gcloud logging read 'resource.type="k8s_container" jsonPayload.event="prediction"' \
  --limit 20 --format json

# Deliverable 6: stress test (run from a VM/Cloud Shell that can reach the LB)
sudo apt-get install -y wrk
./stress_test/stress_test.sh http://<EXTERNAL_IP> 30s 12 2500

# Deliverable 7: drift detection
python -m observability.drift_detection \
  --reference model/train_reference.csv \
  --current observability/sample_100.csv
```

## Explainability findings (Deliverable 2)

Fill in after running `shap_analysis.py` against the real dataset —
the script prints and saves (`explainability/shap_report.json`) the
mean absolute SHAP value per feature, ranked. Report the bottom 2-3
features here in plain English (e.g. "`fbs` and `restecg` have the
smallest effect on predicted risk — changing them barely moves the
model's output compared to the top drivers like `cp`, `thal`, `oldpeak`").

## Fairness findings (Deliverable 3)

Fill in after running `fairlearn_analysis.py` — report the demographic
parity difference/ratio and equalized-odds difference for the `age`
groups (and `gender`, included for completeness given the discrepancy
between the deliverable text and the pipeline-overview section of the
brief), plus per-group accuracy from `fairness/fairness_report.json`.

## Stress test results (Deliverable 6)

Fill in after running `stress_test.sh` — paste throughput (Req/Sec),
p50/p90/p99 latency, and any socket errors/timeouts from the `wrk`
output, plus whether the HPA scaled pods 1→3 under load
(`kubectl get hpa --watch` output).

## Drift detection results (Deliverable 7)

Fill in after running `drift_detection.py` — list which features were
flagged as drifted and note the caveat already baked into the script's
output: the 100-row set is generated by uniform sampling over the
training min/max range, so some drift is expected by construction.