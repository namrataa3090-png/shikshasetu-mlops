
from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = (
    PROJECT_ROOT / "data" / "processed"
    / "student_performance_processed.csv"
)
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"

TARGET = "at_risk"
EXCLUDED_FEATURES = ["G1", "G2", "G3", TARGET]


def build_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true: pd.Series,
    predictions,
    probabilities,
) -> dict:
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc_average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true, predictions
        ).tolist(),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    if TARGET not in data.columns:
        raise ValueError(f"Target column {TARGET!r} is missing.")

    # Never allow the final-grade target or its derived label into X.
    feature_columns = [
        column for column in data.columns
        if column not in EXCLUDED_FEATURES
    ]

    X = data[feature_columns].copy()
    y = data[TARGET].astype(int)

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()
    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Baseline: always predict the majority class.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_predictions = dummy.predict(X_test)

    model = build_pipeline(numeric_columns, categorical_columns)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "UCI Student Performance",
        "target": "G3 < 10",
        "excluded_features": EXCLUDED_FEATURES,
        "feature_count": len(feature_columns),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "positive_rate_test": float(y_test.mean()),
        "dummy_baseline": {
            "precision": float(
                precision_score(y_test, dummy_predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y_test, dummy_predictions, zero_division=0)
            ),
            "f1": float(
                f1_score(y_test, dummy_predictions, zero_division=0)
            ),
        },
        "logistic_regression": calculate_metrics(
            y_test, predictions, probabilities
        ),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_DIR / "student_risk_model.joblib")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print("Baseline training completed.")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features used: {len(feature_columns)}")
    print("\nDummy baseline:")
    print(metrics["dummy_baseline"])
    print("\nLogistic Regression metrics:")
    print(json.dumps(metrics["logistic_regression"], indent=2))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Model saved to: {MODEL_DIR / 'student_risk_model.joblib'}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
