
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import shikshasetu.api as api_module


client = TestClient(api_module.app)


def get_valid_features():
    """Build a valid request from the project's processed dataset."""
    model = api_module.model

    assert model is not None, (
        "Model is unavailable. Check the API startup logs."
    )

    expected_features = list(model.feature_names_in_)

    data_path = (
        Path(api_module.PROJECT_ROOT)
        / "data"
        / "processed"
        / "student_performance_processed.csv"
    )

    assert data_path.exists(), (
        "Processed dataset not found. Run the ETL pipeline first."
    )

    data = pd.read_csv(data_path)

    # Remove target and grade columns, if present.
    X = data.drop(
        columns=["G1", "G2", "G3", "at_risk"],
        errors="ignore",
    )

    missing_columns = set(expected_features) - set(X.columns)
    assert not missing_columns, (
        f"Dataset is missing model features: {sorted(missing_columns)}"
    )

    row = X.iloc[0][expected_features].to_dict()

    # Convert NumPy scalar values into JSON-compatible Python values.
    return {
        name: value.item() if hasattr(value, "item") else value
        for name, value in row.items()
    }


def test_predict_returns_valid_response():
    response = client.post(
        "/predict",
        json={"features": get_valid_features()},
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["interpretation"] in [
        "Flagged for supportive review",
        "Not flagged by this model",
    ]
    assert result["model_type"] == "Logistic Regression pipeline"
    assert "disclaimer" in result

    assert "positive_class_probability" in result
    probability = result["positive_class_probability"]
    assert 0.0 <= probability <= 1.0


def test_predict_rejects_missing_features():
    features = get_valid_features()
    features.pop(next(iter(features)))

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_predict_rejects_unexpected_features():
    features = get_valid_features()
    features["unexpected_feature"] = 123

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422
    assert "unexpected_feature" in (
        response.json()["detail"]["unexpected_features"]
    )


def test_predict_rejects_missing_features_and_unexpected_features():
    features = get_valid_features()
    features.pop(next(iter(features)))
    features["unexpected_feature"] = 123

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["missing_features"]
    assert "unexpected_feature" in detail["unexpected_features"]


def test_predict_returns_503_when_model_is_unavailable(monkeypatch):
    monkeypatch.setattr(api_module, "model", None)

    response = client.post(
        "/predict",
        json={"features": {}},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Model is unavailable. Check the API startup logs."
    )
