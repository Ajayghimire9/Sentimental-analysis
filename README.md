# Sentiment Intelligence Platform

**Production-style NLP system for social-media sentiment classification, reproducible model training, experiment tracking, and monitored API inference.**

[![CI](https://github.com/Ajayghimire9/Sentimental-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Ajayghimire9/Sentimental-analysis/actions/workflows/ci.yml)

## Why this project

This project was rebuilt from a notebook-style sentiment-analysis exercise into an engineering-focused NLP system. The original approach used TF-IDF and classical classifiers; the new implementation separates data, preprocessing, training, evaluation, serving, monitoring, and automation.

The project uses the public **TweetEval sentiment benchmark** rather than committing a large dataset to Git. Data is downloaded reproducibly at runtime through Hugging Face Datasets.

## Architecture

```text
TweetEval
   ↓
Dataset validation
   ↓
Deterministic text normalization
   ↓
TF-IDF (1–2 grams)
   ↓
Logistic Regression baseline
   ↓
Accuracy / Macro-F1 / Precision / Recall
   ↓
MLflow experiment tracking + model registry
   ↓
Champion model alias
   ↓
FastAPI inference service
   ↓
Prometheus metrics
   ↓
GitHub Actions CI + scheduled retraining
```

## Engineering capabilities

### NLP / ML
- Social-media text normalization
- URL and mention normalization
- TF-IDF unigram/bigram representation
- Logistic Regression classifier
- Class-balanced training
- Macro-averaged evaluation for multiclass sentiment
- Reproducible random seeds

### MLOps
- MLflow experiment tracking
- Registered model: `sentiment-classifier`
- Stable `champion` model alias for serving
- Automated scheduled retraining
- Training metrics stored as CI artifacts
- Dataset validation before training

### Serving
- FastAPI REST API
- Request validation with Pydantic
- `/health` health endpoint
- `/predict` inference endpoint
- `/metrics` Prometheus endpoint
- Inference latency and prediction counters

### DevOps
- Python package structure
- Docker image for inference
- GitHub Actions CI
- Ruff linting
- pytest automated tests
- Prometheus monitoring configuration

## API

After a champion model has been registered in MLflow:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"I absolutely love this product!"}'
```

Example response:

```json
{
  "label": "positive",
  "class_id": 2,
  "text_length": 31
}
```

## Training

Install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

Run a small development training job:

```bash
python -m src.train --limit 10000
```

Run the full benchmark:

```bash
python -m src.train
```

MLflow can be configured with:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

## Reproducibility

The dataset is not stored in the repository. The training pipeline retrieves the named TweetEval sentiment configuration at runtime, validates its schema, applies the same preprocessing used by the serving pipeline, and records model parameters and metrics in MLflow.

This avoids a multi-hundred-megabyte Git repository while keeping the experiment reproducible.

## Monitoring

The API exposes Prometheus-compatible metrics:

```text
sentiment_requests_total
sentiment_predictions_total
sentiment_request_latency_seconds
```

Prometheus is configured to scrape the API every 15 seconds.

## CI/CD workflow

Every push and pull request runs:

1. Dependency installation
2. Ruff linting
3. pytest

A scheduled GitHub Actions workflow retrains the model weekly and stores the resulting metrics as a workflow artifact. The MLflow tracking URI is supplied through a GitHub Actions secret when a remote tracking server is configured.

## Repository structure

```text
.
├── src/
│   ├── api.py              # FastAPI inference service
│   ├── data.py             # TweetEval loading + validation
│   ├── model.py            # TF-IDF + Logistic Regression
│   ├── text.py             # Shared text normalization
│   └── train.py            # Reproducible training + MLflow
├── tests/
│   └── test_text.py
├── monitoring/
│   └── prometheus.yml
├── .github/workflows/
│   ├── ci.yml
│   └── retrain.yml
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Technology stack

**Python · Pandas · NumPy · scikit-learn · Hugging Face Datasets · MLflow · FastAPI · Docker · Prometheus · GitHub Actions · pytest · Ruff · Git**

The technologies listed above correspond to implementation in this repository rather than keyword-only CV claims.

## Next production extensions

The architecture is intentionally ready for additional model families such as DistilBERT/BERT, model drift detection, a feature/model registry workflow, Kubernetes deployment, and cloud object storage. Those components should be added when they are backed by executable code and infrastructure rather than documentation-only claims.

## License

MIT
