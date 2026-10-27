
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_performance_raw.csv"

REQUIRED_GRADE_COLUMNS = ["G1", "G2", "G3"]


def validate_student_data(data: pd.DataFrame) -> None:
    """Raise an error when essential dataset quality checks fail."""

    if data.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [
        column
        for column in REQUIRED_GRADE_COLUMNS
        if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data.isna().any().any():
        missing_counts = data.isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(
            f"Dataset contains missing values:\n{missing_counts}"
        )

    if data.duplicated().any():
        raise ValueError(
            f"Dataset contains {data.duplicated().sum()} duplicate rows."
        )

    for column in REQUIRED_GRADE_COLUMNS:
        invalid_grades = ~data[column].between(0, 20)
        if invalid_grades.any():
            raise ValueError(
                f"{column} contains {invalid_grades.sum()} "
                "values outside the range 0–20."
            )

    print("Data validation passed.")
    print(f"Rows validated: {len(data)}")
    print(f"Columns validated: {len(data.columns)}")


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}. "
            "Run ingest_data.py first."
        )

    data = pd.read_csv(RAW_DATA_PATH)
    validate_student_data(data)


if __name__ == "__main__":
    main()
