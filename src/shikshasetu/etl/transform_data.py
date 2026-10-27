
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_performance_raw.csv"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "student_performance_processed.csv"
)


def transform_student_data(data: pd.DataFrame) -> pd.DataFrame:
    """Create a binary academic-risk target from final grades."""

    if "G3" not in data.columns:
        raise ValueError("Required target column G3 is missing.")

    processed = data.copy()

    # Demonstration target: below the passing-grade threshold.
    processed["at_risk"] = (processed["G3"] < 10).astype("int8")

    # Preserve G3 in this processed dataset for auditability.
    # The ML training pipeline must explicitly exclude G3 from X.
    return processed


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found: {RAW_DATA_PATH}. "
            "Run ingestion first."
        )

    raw_data = pd.read_csv(RAW_DATA_PATH)
    processed_data = transform_student_data(raw_data)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"Processed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Rows: {len(processed_data)}")
    print("\nRisk-label distribution:")
    print(
        processed_data["at_risk"]
        .value_counts()
        .sort_index()
        .rename(index={0: "Not at risk", 1: "At risk"})
    )
    print(
        "\nImportant: G3 is retained for auditing, "
        "but must not be used as a model input."
    )


if __name__ == "__main__":
    main()
