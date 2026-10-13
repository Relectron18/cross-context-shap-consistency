import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results" / "leakage_audit"


def audit_context(context):
    folder = DATA_DIR / context

    X_train = pd.read_csv(folder / "X_train.csv")
    X_test = pd.read_csv(folder / "X_test.csv")
    y_train = pd.read_csv(folder / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(folder / "y_test.csv").iloc[:, 0]

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(f"{context}: feature and label row counts differ.")

    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(f"{context}: train/test feature columns differ.")

    features = list(X_train.columns)

    train = X_train.copy()
    train["__train_label"] = y_train.to_numpy()

    test = X_test.copy()
    test["__test_label"] = y_test.to_numpy()

    # Keep distinct feature-vector/label combinations.
    train = train.drop_duplicates(subset=features + ["__train_label"])
    test = test.drop_duplicates(subset=features + ["__test_label"])

    pairs = train.merge(test, on=features, how="inner")
    mismatches = pairs[pairs["__train_label"] != pairs["__test_label"]]

    shared_vectors = X_train[features].drop_duplicates().merge(
        X_test[features].drop_duplicates(),
        on=features,
        how="inner"
    )

    mismatch_vectors = mismatches[features].drop_duplicates()

    summary = {
        "context": context,
        "shared_distinct_feature_vectors": len(shared_vectors),
        "distinct_cross_split_label_pairs": len(pairs),
        "label_mismatch_pairs": len(mismatches),
        "shared_vectors_with_label_mismatch": len(mismatch_vectors),
    }

    combinations = (
        mismatches.groupby(["__train_label", "__test_label"])
        .size()
        .reset_index(name="distinct_vector_label_pairs")
    )
    combinations.insert(0, "context", context)

    print(f"\n{context.upper()}")
    for key, value in summary.items():
        if key != "context":
            print(f"{key}: {value}")

    if combinations.empty:
        print("Mismatching label combinations: None")
    else:
        print("\nMismatching label combinations:")
        print(combinations.to_string(index=False))

    return summary, combinations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    all_combinations = []

    for context in ("novpn", "vpn"):
        summary, combinations = audit_context(context)
        summaries.append(summary)
        all_combinations.append(combinations)

    pd.DataFrame(summaries).to_csv(
        OUTPUT_DIR / "label_conflict_summary.csv", index=False
    )

    pd.concat(all_combinations, ignore_index=True).to_csv(
        OUTPUT_DIR / "label_conflict_combinations.csv", index=False
    )

    print(f"\nAudit results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
