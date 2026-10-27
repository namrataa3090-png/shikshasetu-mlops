
from pathlib import Path

from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def ingest_student_data() -> Path:
    """Download the UCI Student Performance dataset and save the raw records."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = fetch_ucirepo(id=320)

    # Preserve the original predictors and all three grade columns.
    features = dataset.data.features.reset_index(drop=True)
    targets = dataset.data.targets.reset_index(drop=True)
    raw_data = features.join(targets)

    output_path = RAW_DATA_DIR / "student_performance_raw.csv"
    raw_data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Rows: {raw_data.shape[0]}")
    print(f"Columns: {raw_data.shape[1]}")
    print(f"Missing values: {raw_data.isna().sum().sum()}")
    print("\nGrade preview:")
    print(raw_data[["G1", "G2", "G3"]].head())

    return output_path


if __name__ == "__main__":
    ingest_student_data()
