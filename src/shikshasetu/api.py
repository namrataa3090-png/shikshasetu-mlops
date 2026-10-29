
from pathlib import Path
from typing import Any

import json
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "student_risk_model.joblib"
)
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"


# Load the trained pipeline once when the API starts.
model = None
model_load_error = None

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model_load_error = str(exc)


app = FastAPI(
    title="ShikshaSetu API",
    description=(
        "Student support prototype API using a trained "
        "scikit-learn pipeline."
    ),
    version="1.0.0",
)

# Allow the local React development server to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:5175",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(
        ...,
        description="Feature names and values expected by the trained model.",
    )


@app.get("/")
def root():
    return {
        "application": "ShikshaSetu",
        "message": "Student Success Platform API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
    }


@app.get("/metrics")
def get_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Evaluation metrics file was not found.",
        )

    try:
        return json.loads(METRICS_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read evaluation metrics.",
        ) from exc


@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable. Check the API startup logs.",
        )

    # The fitted pipeline remembers the original input column names.
    expected_features = list(
        getattr(model, "feature_names_in_", [])
    )

    if not expected_features:
        raise HTTPException(
            status_code=500,
            detail="Model feature names are unavailable.",
        )

    supplied_features = request.features
    expected_set = set(expected_features)
    supplied_set = set(supplied_features)

    missing = sorted(expected_set - supplied_set)
    unexpected = sorted(supplied_set - expected_set)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Input features do not match the trained model.",
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    try:
        # Preserve the exact column order used during training.
        input_df = pd.DataFrame(
            [{name: supplied_features[name] for name in expected_features}]
        )

        prediction = int(model.predict(input_df)[0])

        result = {
            "prediction": prediction,
            "interpretation": (
                "Flagged for supportive review"
                if prediction == 1
                else "Not flagged by this model"
            ),
            "model_type": "Logistic Regression pipeline",
            "disclaimer": (
                "Prototype output only. This prediction is not a diagnosis "
                "or a definitive judgment about a student. Educator review "
                "and broader validation are required."
            ),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                positive_index = classes.index(1)
                result["positive_class_probability"] = float(
                    probabilities[positive_index]
                )

        return result

    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not process the supplied feature values. "
                "Check their types and values."
            ),
        ) from exc
