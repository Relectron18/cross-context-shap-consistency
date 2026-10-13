import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONTROLLED_DIR = DATA_DIR / "controlled"

RANDOM_STATE = 42


def load_context(context):
    data_dir = DATA_DIR / context

    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]

    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

    return X_train, y_train, X_test, y_test


def balanced_subset(X, y, n_per_class):
    selected_indices = []

    for label in sorted(y.unique()):
        indices = y[y == label].index

        sampled = indices.to_series().sample(
            n=n_per_class,
            random_state=RANDOM_STATE
        )

        selected_indices.extend(sampled.tolist())

    selected_indices = sorted(selected_indices)

    return (
        X.loc[selected_indices].reset_index(drop=True),
        y.loc[selected_indices].reset_index(drop=True)
    )


def save_context(context, X_train, y_train, X_test, y_test):
    output_dir = CONTROLLED_DIR / context
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(output_dir / "X_train.csv", index=False)
    y_train.to_csv(output_dir / "y_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_test.to_csv(output_dir / "y_test.csv", index=False)


def main():
    contexts = {}

    for context in ["novpn", "vpn"]:
        contexts[context] = load_context(context)

    train_counts = pd.concat(
        [
            contexts["novpn"][1].value_counts(),
            contexts["vpn"][1].value_counts()
        ],
        axis=1
    ).min(axis=1)

    test_counts = pd.concat(
        [
            contexts["novpn"][3].value_counts(),
            contexts["vpn"][3].value_counts()
        ],
        axis=1
    ).min(axis=1)

    train_n = int(train_counts.min())
    test_n = int(test_counts.min())

    print(f"Matched train samples per class: {train_n}")
    print(f"Matched test samples per class: {test_n}")

    for context in ["novpn", "vpn"]:
        X_train, y_train, X_test, y_test = contexts[context]

        X_train_balanced, y_train_balanced = balanced_subset(
            X_train,
            y_train,
            train_n
        )

        X_test_balanced, y_test_balanced = balanced_subset(
            X_test,
            y_test,
            test_n
        )

        save_context(
            context,
            X_train_balanced,
            y_train_balanced,
            X_test_balanced,
            y_test_balanced
        )

        print(
            f"{context.upper()}: "
            f"train={len(X_train_balanced)}, "
            f"test={len(X_test_balanced)}"
        )

    print("\nControlled datasets saved.")
    print("Original datasets were not modified.")


if __name__ == "__main__":
    main()