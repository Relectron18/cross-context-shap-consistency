import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def nearest_distances(X_source, X_reference, exclude_self=False):
    n_neighbors = 2 if exclude_self else 1

    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="euclidean"
    )

    nn.fit(X_reference)

    distances, _ = nn.kneighbors(X_source)

    if exclude_self:
        return distances[:, 1]

    return distances[:, 0]


def audit_context(context):
    data_dir = DATA_DIR / context

    X_train = pd.read_csv(data_dir / "X_train_scaled.csv")
    X_test = pd.read_csv(data_dir / "X_test_scaled.csv")

    # Cross-split nearest-neighbor distances
    train_to_test = nearest_distances(
        X_train,
        X_test
    )

    test_to_train = nearest_distances(
        X_test,
        X_train
    )

    # Within-split nearest-neighbor distances
    train_to_train = nearest_distances(
        X_train,
        X_train,
        exclude_self=True
    )

    test_to_test = nearest_distances(
        X_test,
        X_test,
        exclude_self=True
    )

    print(f"\n===== {context.upper()} =====")

    print("\nTrain -> nearest Test")
    print(f"Minimum distance : {train_to_test.min():.6f}")
    print(f"1st percentile   : {np.percentile(train_to_test, 1):.6f}")
    print(f"5th percentile   : {np.percentile(train_to_test, 5):.6f}")
    print(f"Median           : {np.median(train_to_test):.6f}")

    print("\nTest -> nearest Train")
    print(f"Minimum distance : {test_to_train.min():.6f}")
    print(f"1st percentile   : {np.percentile(test_to_train, 1):.6f}")
    print(f"5th percentile   : {np.percentile(test_to_train, 5):.6f}")
    print(f"Median           : {np.median(test_to_train):.6f}")

    print("\nTrain -> nearest other Train")
    print(f"Minimum distance : {train_to_train.min():.6f}")
    print(f"1st percentile   : {np.percentile(train_to_train, 1):.6f}")
    print(f"5th percentile   : {np.percentile(train_to_train, 5):.6f}")
    print(f"Median           : {np.median(train_to_train):.6f}")

    print("\nTest -> nearest other Test")
    print(f"Minimum distance : {test_to_test.min():.6f}")
    print(f"1st percentile   : {np.percentile(test_to_test, 1):.6f}")
    print(f"5th percentile   : {np.percentile(test_to_test, 5):.6f}")
    print(f"Median           : {np.median(test_to_test):.6f}")


if __name__ == "__main__":
    audit_context("novpn")
    audit_context("vpn")
