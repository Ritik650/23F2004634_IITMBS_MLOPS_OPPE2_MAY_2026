# OPPE AI Usage Documentation 

## AI Tools Utilized and Conversation History
> List all GenAI / LLM tools used during the exam  
> Provide **public share links** to AI chats or attach conversation files if links are not available  

- Claude (Anthropic)
    - Purpose : To generate the ML Pipeline
    - Shared Chat Link : https://claude.ai/share/e318dc9d-7239-43d7-aea6-40eafdbc407a
    - Notes (optional) :


---

## Prompts and Responses Used
> Include **all prompts** that contributed to solving the exam tasks

> Include **all responses** in case of public share links are not available to share

### Tool Name #1: ___________________
- Prompt 1: this is the oppe 2 mlops problem statement :-


OPPE-2: Problem Statement
Build a production-ready, explainable, observable, scalable, and maintainable deployment for a heart disease prediction model on Google Cloud Platform (GCP).

Role & Objective
As an MLOps engineer at a healthcare firm, you need to transition a heart disease prediction model from a notebook to a production-ready environment. The model classifies whether a patient is likely to have heart disease based on 14 clinical attributes.
Your task is to build a dockerized, API-served deployment on Google Kubernetes Engine (GKE) with end-to-end CI/CD. You must ensure the system is explainable, fair, observable, scalable, and resilient to data drift and adversarial attacks.
Pipeline Overview
This sequence outlines the high-level flow of the deployment pipeline. Refer to the Deliverables section below for specific implementation instructions, requirements, and marking criteria before you begin building.

1. Data & Scripts: Heart disease dataset and scripts from the `MLOPS_MAY_2026_OPPE2` repository (`main` branch).
2. Responsible AI: Explainability and fairness analysis using SHAP and Fairlearn (bias detection on gender).
3. Deployment: Dockerized API deployed on GKE with Kubernetes autoscaling (maximum of 3 pods) and CI/CD via GitHub Actions.
4. Observability: Logging and monitoring, including per-sample prediction logging using a 100-row random dataset.
5. Stress Testing: Performance monitoring under load using `wrk` with over 2,000 concurrent connections to analyze throughput, latency, and timeouts.
6. Drift Detection: Input drift analysis comparing the training data distribution against the generated prediction data.

Deliverables
Deliverable 1 [Mandatory]: Setup Private Git Repository

* Set up a private Git repository.
* Use the following repository name format: `<IITM_BS_ROLL_NUMBER>_IITMBS_MLOPS_OPPE2_MAY_2026` (e.g., `21F10005000_IITMBS_MLOPS_OPPE2_MAY_2026`).
* Important: Add collaborator access for [IITMBSMLOps](https://github.com/IITMBSMLOps) or da5014_1@study.iitm.ac.in.

Before leaving the exam, verify that the collaborator invitation has been accepted by checking your GitHub repository's collaborators list. If the invitation is still pending, notify the course team before your session ends.
Deliverable 2 [10 Marks]: Model Explainability
Using explainability tools, describe in plain English the factors that have the least impact on predicting whether a patient has heart disease.
Deliverable 3 [10 Marks]: Fairness Testing with Fairlearn
Test the model for fairness using Fairlearn, with "age" as the sensitive attribute.
Deliverable 4 [40 Marks]: Dockerized API Deployment on GKE
Convert the provided notebook into a dockerized, API-deployed model running on GCP.

* Use Kubernetes with autoscaling (configured for a maximum of 3 pods) as the deployment layer.
* CI/CD workflows must be triggered using GitHub Actions.

Deliverable 5 [20 Marks]: Per-Sample Prediction with Logging & Observability
Generate a 100-row random dataset and run per-sample predictions through your deployed API.

* Log each prediction request individually with its input features, predicted output, and timestamp.
* Demonstrate observability using GCP Cloud Logging.

Deliverable 6 [10 Marks]: Performance Monitoring & Stress Testing with wrk
Stress test your deployed API using `wrk` with a high-concurrency workload (greater than 2,000 concurrent connections).

* Analyze throughput, latency distribution, and request timeout behavior under load.
* You may reuse the 100-row random sample data generated in Deliverable 5.

Deliverable 7 [10 Marks]: Input Drift Detection
Detect whether the distribution of incoming data has shifted from the training data.

* Compute input drift by comparing the training data distribution against the 100-row generated dataset used for prediction in Deliverable 5.

Total: 100 marks (10 + 10 + 40 + 20 + 10 + 10)



Select the files included as part of your private Git Repo? *
A valid Git Repo should contain
- code/scripts utilized to reach the objective (*.py, *.ipynb, *.sh, etc.)
- output files(if any) demonstrating successful completion of the objective
- README.md (explaining utility of each of the included files)
should not contain
- any binary files which are output artifacts of executions (pickle files, models, etc.)
- any standard dataset splits used for training
- video screencast
- Please feel free to share any other type of files included as part of zip file 


https://github.com/IITMBSMLOps/MLOPS_MAY_2026_OPPE2


i have created the github repo :- https://github.com/Ritik650/23F2004634_IITMBS_MLOPS_OPPE2_MAY_2026


and prepared the instance on cloud
now first of all provide me with all the script/files needed for this .


-   Response Log:  
    
    This is a big deliverable — let me build out the full project scaffold (training, API, Docker, K8s, CI/CD, explainability, fairness, observability, drift, stress test) as real files you can drop into your repo.Now the K8s manifests, wrk stress test, and CI/CD workflow.Now the README that ties everything together and gives you the exact commands to run end-to-end.I've built out the full scaffold as a downloadable repo. Here's what's in it and how it maps to the deliverables:

**Structure:**
- `training/train.py` — cleans/imputes/encodes data, tunes `LogisticRegression`, bundles preprocessing + model into one `Pipeline` artifact (fixes the notebook's `sno`-as-feature bug and dropped-NaN approach)
- `app/main.py` — FastAPI service (`/predict`, `/health`, `/ready`) with structured JSON logging to stdout for Cloud Logging
- `Dockerfile` — multi-stage build that **trains the model inside the image**, so neither `data.csv` nor `model.joblib` ever needs to be committed to git
- `explainability/shap_analysis.py` — Deliverable 2, SHAP feature ranking
- `fairness/fairlearn_analysis.py` — Deliverable 3, evaluates both `age` (as the deliverable text says) and `gender` (as the pipeline-overview section separately says) — worth flagging to your course team since the brief contradicts itself
- `k8s/deployment.yaml`, `service.yaml`, `hpa.yaml` — GKE deployment, LoadBalancer, HPA capped at **3 pods**
- `.github/workflows/cicd.yaml` — builds, pushes to Artifact Registry, deploys to GKE on push to `main`
- `observability/generate_predictions.py` + `drift_detection.py` — Deliverables 5 & 7
- `stress_test/post.lua` + `stress_test.sh` — Deliverable 6, `wrk` at >2,000 connections
- `README.md` — full setup steps, required GitHub secrets, and the exact commands for each deliverable

**Things you need to do before this runs for real:**
1. `training/train.py`'s `DEFAULT_DATA_URL` is a guess at the course repo's raw file path — verify the actual path in `MLOPS_MAY_2026_OPPE2` and fix it.
2. Set the 5 GitHub Actions secrets listed in the README (`GCP_PROJECT_ID`, `GCP_REGION`, `GKE_CLUSTER_NAME`, `GKE_ZONE`, `GCP_SA_KEY`).
3. Create the Artifact Registry repo and GKE cluster (commands in README).
4. Confirm the `IITMBSMLOps` collaborator invite is accepted before your session ends.

I included your `data.csv` inside the zip under `data/` for local testing convenience only — it's git-ignored so it won't get committed, per the assignment's rules.

Want me to walk through the GCP setup commands next, or dig into any one script (e.g. tightening the fairness analysis or the wrk test)?
