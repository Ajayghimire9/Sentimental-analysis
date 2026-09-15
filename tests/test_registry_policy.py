import pytest

from src import api


def test_registry_failure_never_falls_back_to_latest(monkeypatch):
    calls = []

    def fail(uri):
        calls.append(uri)
        raise RuntimeError("registry offline")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://example.invalid")
    monkeypatch.setattr(api, "_model", None)
    monkeypatch.setattr(api.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(api.mlflow.pyfunc, "load_model", fail)
    with pytest.raises(RuntimeError):
        api.get_model()
    assert calls == ["models:/sentiment-classifier@champion"]
