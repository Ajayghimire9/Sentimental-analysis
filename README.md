# Sentiment Intelligence Platform

**End-to-end NLP / MLOps platform for social-media sentiment classification, experiment tracking, model governance, API serving, monitoring, drift detection, containerization, and automated retraining.**

[![CI](https://github.com/Ajayghimire9/Sentimental-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajayghimire9/Sentimental-analysis/actions/workflows/ci.yml)

## Executive summary

This repository was rebuilt from a university-style sentiment-analysis script into an engineering portfolio project. It demonstrates the complete ML lifecycle rather than only model training: reproducible data acquisition, validation, NLP preprocessing, model evaluation, experiment tracking, model registry integration, REST serving, observability, drift analysis, Docker/Kubernetes deployment, testing, and automated retraining.

## System architecture

```text
                 ┌──────────────────────┐
                 │ TweetEval benchmark  │
                 └──────────┬───────────┘
                            ↓
                 Data validation layer
                            ↓
                 Text normalization
                            ↓
              ┌─────────────┴─────────────┐
              │                           │
       TF-IDF + LR                 Transformer-ready
       production baseline             extension point
              │
              ↓
       Evaluation / reports
              ↓
          MLflow Tracking
              ↓
        Model Registry
              ↓
       FastAPI inference
              ↓
      ┌───────┴────────┐
      ↓                ↓
 Prometheus         Drift monitor
      ↓                ↓
 Grafana          investigation
      │
      └──── Docker / Kubernetes
               ↓
        GitHub Actions CI
               ↓
        Scheduled retraining
```

## What this project demonstrates

### 1. Data engineering for ML
- Reproducible public dataset acquisition through Hugging Face Datasets
- Explicit dataset schema validation
- No hard-coded Windows/local filesystem paths
- Large datasets kept outside Git
- Deterministic training inputs

### 2. NLP / machine learning
- Social-media text normalization
- URL and user-mention normalization
- TF-IDF unigram/bigram features
- Class-balanced Logistic Regression
- Multiclass sentiment classification
- Accuracy, Macro-F1, precision and recall
- Reproducible random state

### 3. MLOps
- MLflow experiment tracking
- Registered model: `sentiment-classifier`
- Training metadata and metrics
- Model artifact logging
- Candidate/champion model workflow ready for registry aliases
- Scheduled retraining
- Metrics published as CI artifacts

### 4. Model monitoring
- Prometheus request counter
- Prediction counters by sentiment
- Inference latency histogram
- Population Stability Index utility for detecting distribution changes
- Health endpoint for service availability

### 5. Production API

FastAPI exposes:

| Endpoint | Purpose |
|---|---|
| `GET /health` | service health |
| `POST /predict` | sentiment inference |
| `GET /metrics` | Prometheus metrics |

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The product is excellent and I love it!"}'
```

### 6. DevOps / platform engineering
- Dockerized inference service
- Kubernetes Deployment
- Two API replicas
- CPU/memory requests and limits
- Readiness and liveness probes
- Kubernetes Service
- Prometheus scrape configuration
- GitHub Actions CI
- Weekly retraining workflow

## Project structure

```text
.
├── src/
│   ├── api.py                 # FastAPI inference service
│   ├── data.py                # data loading + schema validation
│   ├── drift.py               # PSI-based drift utility
│   ├── model.py               # TF-IDF + Logistic Regression
│   ├── text.py                # shared preprocessing
│   └── train.py               # training + MLflow logging
├── tests/
│   ├── test_drift.py
│   └── test_text.py
├── monitoring/
│   └── prometheus.yml
├── k8s/
│   └── deployment.yaml
├── .github/workflows/
│   ├── ci.yml
│   └── retrain.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run tests:

```bash
make test
```

Run linting:

```bash
make lint
```

Train a development model:

```bash
make train
```

Run the API:

```bash
make api
```

## MLflow

Set the tracking server before training when using a remote MLflow deployment:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.train --limit 10000
```

The training process records model configuration and evaluation metrics and registers the resulting model as `sentiment-classifier`.

## Kubernetes

Build the image and apply the deployment in a local Kubernetes environment such as Docker Desktop or Minikube:

```bash
docker build -t sentiment-intelligence:latest .
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get service sentiment-api
```

The deployment is intentionally infrastructure-neutral: cloud-specific registries, credentials, and clusters are supplied by the deployment environment rather than committed to Git.

## CI/CD

Pull requests and pushes execute:

1. Python environment setup
2. Dependency installation
3. Ruff linting
4. pytest

A weekly GitHub Actions workflow can execute the training pipeline. A remote MLflow tracking URI can be supplied through the repository's Actions secret configuration.

## Model governance

The serving layer prefers the MLflow `champion` alias. This is a deliberate separation between **training** and **serving**: an API should not need to know which numeric model version is currently deployed.

For a real production deployment, promotion should be gated by an evaluation threshold and human/automated approval rather than blindly promoting every scheduled run.

## Why TF-IDF instead of pretending this is a transformer project?

TF-IDF + Logistic Regression is an excellent interpretable and inexpensive production baseline. The architecture deliberately keeps the model behind a pipeline boundary so a transformer such as DistilBERT can be evaluated against the baseline later using the same metrics and MLflow workflow.

This is stronger portfolio engineering than adding a transformer dependency without demonstrating model comparison or operational value.

## Technology stack

**Python · Pandas · NumPy · scikit-learn · Hugging Face Datasets · MLflow · FastAPI · Pydantic · Prometheus · Docker · Kubernetes · GitHub Actions · pytest · Ruff · Git**

Every technology listed here is represented by repository implementation or deployment configuration.

## Portfolio story

This project demonstrates progression from **data → NLP → ML → MLOps → API → observability → deployment → automation**.

It is intended to complement the time-series forecasting project rather than duplicate it.

## License

MIT
