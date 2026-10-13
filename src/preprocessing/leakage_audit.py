import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def audit_exact_overlap(context):
    data_dir = DATA_DIR / context

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")

    feature_columns = list(X_train.columns)

    # Remove repeated rows within each split
    train_unique = X_train.drop_duplicates()
    test_unique = X_test.drop_duplicates()

    # Find unique feature vectors shared by train and test
    overlap = train_unique.merge(
        test_unique,
        on=feature_columns,
        how="inner"
    )

    print(f"\n===== {context.upper()} =====")

    print(f"Train rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Unique train feature rows: {len(train_unique)}")
    print(f"Unique test feature rows: {len(test_unique)}")
    print(
        f"Unique feature vectors shared by train/test: "
        f"{len(overlap)}"
    )

    # Repeated rows within each split
    train_duplicate_rows = len(X_train) - len(train_unique)
    test_duplicate_rows = len(X_test) - len(test_unique)

    print(f"Repeated rows within train: {train_duplicate_rows}")
    print(f"Repeated rows within test: {test_duplicate_rows}")


if __name__ == "__main__":
    audit_exact_overlap("novpn")
    audit_exact_overlap("vpn")