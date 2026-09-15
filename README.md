# SentinelNLP — Sentiment Intelligence Platform

**Production-style NLP platform for social-media sentiment intelligence, combining classical NLP, transformer benchmarking, MLflow model governance, FastAPI serving, observability, drift detection, Docker/Kubernetes deployment, reproducibility, and automated retraining.**

[![CI](https://github.com/Ajayghimire9/Sentimental-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajayghimire9/Sentimental-analysis/actions/workflows/ci.yml)

## Why SentinelNLP?

SentinelNLP was rebuilt from a university sentiment-analysis script into an engineering portfolio project that demonstrates the full ML lifecycle. The repository is intentionally structured like a small production platform: reproducible data acquisition, validation, NLP preprocessing, model training, evaluation, experiment tracking, registry integration, inference serving, monitoring, drift analysis, containerization, Kubernetes deployment, testing, and scheduled retraining.

## Architecture

```text
                         TweetEval
                            │
                            ▼
                  Data acquisition layer
                            │
                            ▼
                    Schema validation
                            │
                            ▼
                  Deterministic NLP text
                       normalization
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        TF-IDF + Logistic        DistilBERT benchmark
          Regression baseline       (optional)
                │                       │
                └───────────┬───────────┘
                            ▼
                    Evaluation / metrics
                            │
                            ▼
                     MLflow Tracking
                            │
                            ▼
                      Model Registry
                            │
                            ▼
                       FastAPI API
                       /     |      \
                      /      |       \
               Prometheus  Drift   Structured logs
                    │        │
                    ▼        ▼
                 Grafana  Investigation
                    │
                    └──────────┐
                               ▼
                         Docker / K8s
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
                 CI checks        Scheduled retraining
                    │
                    ▼
                 GitHub Actions
```

## Engineering capabilities

### NLP / ML
- Multiclass sentiment classification: negative, neutral, positive
- TweetEval benchmark loaded reproducibly through Hugging Face Datasets
- Social-media text normalization
- TF-IDF unigram/bigram representation
- Class-balanced Logistic Regression baseline
- Optional DistilBERT extension for transformer benchmarking
- Accuracy, Macro-F1, precision and recall
- Deterministic preprocessing and random seeds

### Data engineering / reproducibility
- Runtime dataset acquisition instead of machine-specific paths
- Explicit validation for required columns, nulls, empty datasets, and label domains
- DVC pipeline definition for repeatable validation and training
- Metrics persisted as machine-readable JSON
- Large datasets kept outside the Git history

### MLOps
- MLflow experiment tracking
- Registered model: `sentiment-classifier`
- Artifact and metric logging
- Candidate/champion serving architecture
- Scheduled retraining through GitHub Actions
- Separation between training, evaluation, registration, and serving

### Production serving
- FastAPI REST service
- Pydantic input validation
- Model loading through MLflow
- Champion alias preferred, with development fallback to the newest registered version
- Health and metrics endpoints
- Explicit request/latency/prediction instrumentation

### Observability
- Prometheus counters and latency histograms
- Sentiment prediction distribution monitoring
- PSI-based distribution-drift utility
- Grafana-compatible scrape configuration
- Kubernetes health probes

### Platform / DevOps
- Multi-stage-ready container architecture using Docker
- Kubernetes Deployment with two API replicas
- Resource requests and limits
- Readiness and liveness probes
- Kubernetes Service
- GitHub Actions CI with Ruff + pytest
- Weekly training workflow
- Infrastructure-neutral cloud deployment boundary

## Repository structure

```text
.
├── src/
│   ├── api.py                 # FastAPI inference service
│   ├── data.py                # TweetEval loading + validation
│   ├── drift.py               # PSI drift calculation
│   ├── model.py               # TF-IDF + Logistic Regression
│   ├── text.py                # deterministic text normalization
│   ├── train.py               # baseline training + MLflow
│   └── transformer.py         # optional DistilBERT extension
├── tests/
│   ├── test_drift.py
│   └── test_text.py
├── monitoring/
│   ├── prometheus.yml
│   ├── alerts.yml
│   └── grafana/
├── k8s/
│   ├── deployment.yaml
│   └── hpa.yaml
├── orchestration/
├── dvc.yaml
├── .github/workflows/
│   ├── ci.yml
│   └── retrain.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run quality checks:

```bash
make lint
make test
```

Train the lightweight production baseline:

```bash
make train
```

Run the API:

```bash
make api
```

Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The product is excellent and I love it!"}'
```

Expected response shape:

```json
{
  "label": "positive",
  "class_id": 2,
  "text_length": 36
}
```

## MLflow

For a remote tracking server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.train --limit 10000
```

Training records configuration, evaluation metrics and the serialized model in MLflow and registers it as `sentiment-classifier`.

The serving layer is designed around a registry alias rather than hard-coding a numeric model version. Promotion should be gated by evaluation quality and approval in a real deployment.

## Transformer benchmark

The transformer dependency is optional so the lightweight baseline remains fast to install and CI remains inexpensive.

```bash
pip install -e '.[transformer]'
```

The transformer module provides a DistilBERT model factory for extending the same benchmark with a 3-class classifier. Transformer training is deliberately kept out of the default CI path because model training is substantially more expensive than unit tests and classical inference.

## Drift monitoring

`src/drift.py` provides a Population Stability Index implementation for numeric distribution comparison. In a production deployment, text-length distributions, prediction distributions, and other monitored signals can be compared against a reference window and exported to the monitoring stack.

## Kubernetes

Build and deploy locally:

```bash
docker build -t sentiment-intelligence:latest .
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl get service sentiment-api
```

Cloud-specific registries, credentials, and clusters are supplied by the deployment environment and are not committed to Git.

## CI / automation

Pull requests and pushes run:

1. Python environment setup
2. Dependency installation
3. Ruff linting
4. pytest

The scheduled workflow can run the training pipeline and publish the resulting metrics as a GitHub Actions artifact. A remote MLflow URI is injected through GitHub Actions secrets rather than stored in source control.

## Design decisions

**Why TF-IDF first?** It is fast, interpretable, inexpensive, and establishes a strong baseline. A transformer should only replace it when its accuracy/latency/cost trade-off is demonstrated.

**Why MLflow?** The model artifact, parameters, metrics, and registry lifecycle need to be separated from application code.

**Why Kubernetes?** The API is stateless and can scale horizontally; probes and resource policies make the deployment behavior explicit.

**Why DVC?** Reproducibility matters when datasets or generated training artifacts grow beyond what should live directly in Git.

## Portfolio positioning

SentinelNLP demonstrates a progression from **data engineering → NLP → machine learning → MLOps → model serving → observability → containerization → orchestration → automation**.

It complements the agricultural forecasting project by demonstrating a different ML domain and a different production architecture.

## License

MIT
