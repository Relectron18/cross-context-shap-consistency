import sys
import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

FEATURES = [
    "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond", "min_flowiat",
    "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle"
]

def bootstrap_by_class(X, y, seed=1001):
    rng = np.random.default_rng(seed)
    sampled_indices = []

    for cls in sorted(y.unique()):
        cls_indices = np.flatnonzero(y.to_numpy() == cls)
        sampled = rng.choice(
            cls_indices,
            size=len(cls_indices),
            replace=True
        )
        sampled_indices.extend(sampled.tolist())

    sampled_indices = np.array(sampled_indices)
    rng.shuffle(sampled_indices)

    return (
        X.iloc[sampled_indices].reset_index(drop=True),
        y.iloc[sampled_indices].reset_index(drop=True)
    )

def create_model():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

def main():
    context = "novpn"

    X_train = pd.read_csv(f"data/{context}/X_train.csv")
    X_test = pd.read_csv(f"data/{context}/X_test.csv")
    y_train = pd.read_csv(f"data/{context}/y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(f"data/{context}/y_test.csv").iloc[:, 0]

    print("=== SINGLE-RUN VALIDATION ===")
    print(f"Context: {context}")
    print(f"Original X_train: {X_train.shape}")
    print(f"Original X_test : {X_test.shape}")
    print(f"Feature count   : {len(FEATURES)}")

    assert list(X_train.columns) == FEATURES
    assert list(X_test.columns) == FEATURES
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)

    X_boot, y_boot = bootstrap_by_class(X_train, y_train, seed=1001)

    print(f"Bootstrap X_train: {X_boot.shape}")
    print("Original class counts :", y_train.value_counts().sort_index().to_dict())
    print("Bootstrap class counts:", y_boot.value_counts().sort_index().to_dict())

    assert X_boot.shape == X_train.shape
    assert y_boot.shape == y_train.shape
    assert y_boot.value_counts().sort_index().equals(
        y_train.value_counts().sort_index()
    )

    model = create_model()
    model.fit(X_boot, y_boot)

    predictions = model.predict(X_test)
    test_accuracy = np.mean(predictions == y_test.to_numpy())

    print(f"Test accuracy: {test_accuracy:.4f}")

    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )
    sv = explainer.shap_values(
        X_test,
        check_additivity=False
    )

    sv = np.array(sv)
    n_classes = len(np.unique(y_train))

    if sv.ndim == 3:
        if sv.shape[0] == X_test.shape[0] and sv.shape[-1] == n_classes:
            sv = np.transpose(sv, (2, 0, 1))
    elif sv.ndim == 2:
        sv = sv[None, :, :]

    print(f"SHAP shape: {sv.shape}")

    assert sv.shape[0] == n_classes
    assert sv.shape[1] == len(X_test)
    assert sv.shape[2] == len(FEATURES)

    mean_abs = np.mean(np.abs(sv), axis=(0, 1))
    ranking = pd.Series(
        mean_abs,
        index=FEATURES
    ).sort_values(ascending=False)

    print("\nTop-10 SHAP features:")
    for rank, (feature, value) in enumerate(ranking.head(10).items(), start=1):
        print(f"{rank:2d}. {feature:<25} {value:.6f}")

    print("\nValidation passed.")
    print("Original test set was read only and was not resampled or modified.")

if __name__ == "__main__":
    main()
