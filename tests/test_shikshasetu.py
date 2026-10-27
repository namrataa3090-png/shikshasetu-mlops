import joblib
from pathlib import Path

import pandas as pd
import pytest

from shikshasetu.etl.transform_data import transform_student_data
from shikshasetu.etl.validate_data import validate_student_data


def sample_data():
    return pd.DataFrame(
        {
            "school": ["GP", "MS", "GP"],
            "age": [15, 16, 17],
            "G1": [8, 12, 9],
            "G2": [9, 13, 8],
            "G3": [9, 13, 8],
        }
    )


def test_transform_creates_correct_risk_labels():
    result = transform_student_data(sample_data())

    assert result["at_risk"].tolist() == [1, 0, 1]
    assert "G3" in result.columns


def test_validation_accepts_valid_data():
    validate_student_data(sample_data())


def test_validation_rejects_missing_values():
    data = sample_data()
    data.loc[0, "age"] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        validate_student_data(data)


def test_validation_rejects_invalid_grade():
    data = sample_data()
    data.loc[0, "G3"] = 25

    with pytest.raises(ValueError, match="outside the range"):
        validate_student_data(data)

def test_trained_model_artifact_can_predict():
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "artifacts"
        / "models"
        / "student_risk_model.joblib"
    )

    data_path = (
        project_root
        / "data"
        / "processed"
        / "student_performance_processed.csv"
    )

    assert model_path.exists(), (
        "Trained model not found. Run train.py first."
    )
    assert data_path.exists(), (
        "Processed data not found. Run the ETL pipeline first."
    )

    model = joblib.load(model_path)
    data = pd.read_csv(data_path)

    # Use the same input features used during training.
    excluded_columns = ["G1", "G2", "G3", "at_risk"]
    X = data.drop(columns=excluded_columns).iloc[[0]]

    prediction = model.predict(X)
    probabilities = model.predict_proba(X)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1]
    assert probabilities.shape == (1, 2)
    assert abs(probabilities[0].sum() - 1.0) < 1e-6
