# Sentiment Intelligence

Three-class sentiment training and registry serving.

This project classifies short messages as negative, neutral or positive. A TF-IDF baseline keeps the training and serving path inspectable, with a separate transformer experiment interface.

## Run locally

Use Python 3.11 or newer in a virtual environment.

```bash
pip install -e ".[dev]"
python -m src.train --limit 1000
# Set MLFLOW_TRACKING_URI and explicitly assign the champion alias before serving.
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

## Design decisions

Training uses the TweetEval train and test partitions and records macro-averaged metrics in MLflow.

Serving loads only sentiment-classifier@champion. A registry outage or missing alias produces an unavailable response; it cannot silently promote the newest model.

Readiness checks model availability separately from process health. Tests simulate registry failures without needing a live tracking server.

## Technology

Python, scikit-learn, Hugging Face Datasets, MLflow, FastAPI, Prometheus, Docker.

## Validation

Run `python -m pytest tests -q` from the repository root. CI runs the maintained test suite and lint checks. Tests use local fixtures or mocks and do not deploy cloud resources.

## Scope and limitations

Training downloads TweetEval and requires network access. Registering a model does not assign the champion alias automatically. Transformer construction is an experiment interface, not a fine-tuned deployed model. Language and domain shifts require new evaluation.
