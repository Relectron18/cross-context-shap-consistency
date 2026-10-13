"""
Task D - SHAP stability analysis for the ISCX VPN/Non-VPN dataset.

Method:
- Keep the original train/test partitions fixed.
- For each context, create class-stratified bootstrap training samples,
  each with the same number of rows as the original training set.
- Train Random Forest and XGBoost with the exact Task B/C hyperparameters.
- Explain the original untouched test set with TreeSHAP using the same
  tree_path_dependent configuration as Task C.
- Store one 23-feature SHAP ranking per run.
- Compute within-context and cross-context pairwise stability using
  Spearman, Kendall, Top-k overlap, and mean absolute rank displacement.
- Compare observed Top-k overlap with the expected overlap from random rankings.
"""

from pathlib import Path
import json
import sys
import warnings

import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "stability"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond", "min_flowiat",
    "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle",
]

CONTEXTS = ("novpn", "vpn")
MODELS = ("RandomForest", "XGBoost")
TOP_KS = (5, 10, 12, 15, 18)

N_RUNS_DEFAULT = 30
BOOTSTRAP_SEED_BASE = 1001


def load_context(context):
    data_dir = DATA_DIR / context

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

    if list(X_train.columns) != FEATURES:
        raise ValueError(
            f"{context}: X_train feature order does not match Task C."
        )

    if list(X_test.columns) != FEATURES:
        raise ValueError(
            f"{context}: X_test feature order does not match Task C."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            f"{context}: X_train/y_train length mismatch."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"{context}: X_test/y_test length mismatch."
        )

    return X_train, X_test, y_train, y_test


def bootstrap_by_class(X, y, seed):
    rng = np.random.default_rng(seed)
    sampled_indices = []

    for cls in sorted(y.unique()):
        cls_indices = np.flatnonzero(y.to_numpy() == cls)

        sampled = rng.choice(
            cls_indices,
            size=len(cls_indices),
            replace=True,
        )

        sampled_indices.extend(sampled.tolist())

    sampled_indices = np.asarray(sampled_indices, dtype=int)
    rng.shuffle(sampled_indices)

    X_boot = X.iloc[sampled_indices].reset_index(drop=True)
    y_boot = y.iloc[sampled_indices].reset_index(drop=True)

    return X_boot, y_boot


def create_model(model_name):
    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    if model_name == "XGBoost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=7,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def get_shap_ranking(model, X_test, n_classes):
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )

    sv = explainer.shap_values(
        X_test,
        check_additivity=False,
    )

    sv = np.asarray(sv)

    if sv.ndim == 3:
        if sv.shape[0] == len(X_test) and sv.shape[-1] == n_classes:
            sv = np.transpose(sv, (2, 0, 1))

    elif sv.ndim == 2:
        sv = sv[None, :, :]

    expected_shape = (n_classes, len(X_test), len(FEATURES))

    if sv.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape {sv.shape}; "
            f"expected {expected_shape}."
        )

    mean_abs = np.mean(np.abs(sv), axis=(0, 1))

    importance = pd.Series(
        mean_abs,
        index=FEATURES,
        dtype=float,
    )

    return importance.sort_values(ascending=False)


def rank_series(importance):
    return importance.rank(
        ascending=False,
        method="first",
    )


def compare_rankings(importance_a, importance_b):
    ranks_a = rank_series(importance_a)
    ranks_b = rank_series(importance_b)

    rho, _ = spearmanr(
        ranks_a.loc[FEATURES],
        ranks_b.loc[FEATURES],
    )

    tau, _ = kendalltau(
        ranks_a.loc[FEATURES],
        ranks_b.loc[FEATURES],
    )

    result = {
        "spearman_rho": float(rho),
        "kendall_tau": float(tau),
    }

    for k in TOP_KS:
        top_a = set(importance_a.nlargest(k).index)
        top_b = set(importance_b.nlargest(k).index)

        result[f"top{k}_overlap"] = len(top_a & top_b)

    result["mean_abs_rank_displacement"] = float(
        np.mean(
            np.abs(
                ranks_a.loc[FEATURES] -
                ranks_b.loc[FEATURES]
            )
        )
    )

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs",
        type=int,
        default=N_RUNS_DEFAULT,
        help="Number of bootstrap runs per context. Use 1 for validation.",
    )

    args = parser.parse_args()

    if args.runs < 1 or args.runs > N_RUNS_DEFAULT:
        raise ValueError("--runs must be between 1 and 30.")

    data = {}

    for context in CONTEXTS:
        X_train, X_test, y_train, y_test = load_context(context)

        data[context] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    rankings = {}
    run_metadata = []

    print(
        f"=== Task D: {args.runs} run(s) per context/model ==="
    )

    for context in CONTEXTS:
        X_train = data[context]["X_train"]
        X_test = data[context]["X_test"]
        y_train = data[context]["y_train"]
        y_test = data[context]["y_test"]

        n_classes = y_train.nunique()

        print(
            f"\n{context.upper()}: "
            f"train={X_train.shape}, "
            f"test={X_test.shape}, "
            f"classes={n_classes}"
        )

        for model_name in MODELS:
            key = (context, model_name)
            rankings[key] = []

            for run_id in range(1, args.runs + 1):
                seed = BOOTSTRAP_SEED_BASE + run_id

                X_boot, y_boot = bootstrap_by_class(
                    X_train,
                    y_train,
                    seed,
                )

                expected_counts = (
                    y_train
                    .value_counts()
                    .sort_index()
                )

                actual_counts = (
                    y_boot
                    .value_counts()
                    .sort_index()
                )

                if not actual_counts.equals(expected_counts):
                    raise AssertionError(
                        f"{context} run {run_id}: "
                        "bootstrap class counts changed."
                    )

                model = create_model(model_name)

                model.fit(
                    X_boot,
                    y_boot,
                )

                predictions = model.predict(X_test)

                accuracy = float(
                    np.mean(
                        predictions ==
                        y_test.to_numpy()
                    )
                )

                ranking = get_shap_ranking(
                    model,
                    X_test,
                    n_classes,
                )

                rankings[key].append(ranking)

                run_metadata.append({
                    "context": context,
                    "model": model_name,
                    "run": run_id,
                    "bootstrap_seed": seed,
                    "n_train_original": len(X_train),
                    "n_train_bootstrap": len(X_boot),
                    "n_test": len(X_test),
                    "test_accuracy": accuracy,
                })

                print(
                    f"  {model_name:12s} "
                    f"run {run_id:02d}: "
                    f"accuracy={accuracy:.4f}, "
                    f"top1={ranking.index[0]}"
                )

    ranking_rows = []

    for (context, model_name), run_list in rankings.items():
        for run_id, importance in enumerate(
            run_list,
            start=1,
        ):
            for rank, feature in enumerate(
                importance.index,
                start=1,
            ):
                ranking_rows.append({
                    "context": context,
                    "model": model_name,
                    "run": run_id,
                    "feature": feature,
                    "rank": rank,
                    "mean_abs_shap": float(
                        importance.loc[feature]
                    ),
                })

    pd.DataFrame(ranking_rows).to_csv(
        RESULTS_DIR / "run_rankings.csv",
        index=False,
    )

    pd.DataFrame(run_metadata).to_csv(
        RESULTS_DIR / "run_metadata.csv",
        index=False,
    )

    pairwise_rows = []

    for model_name in MODELS:

        # Within-context pairwise comparisons
        for context in CONTEXTS:
            run_list = rankings[
                (context, model_name)
            ]

            for i in range(len(run_list)):
                for j in range(i + 1, len(run_list)):

                    comparison = compare_rankings(
                        run_list[i],
                        run_list[j],
                    )

                    pairwise_rows.append({
                        "model": model_name,
                        "comparison_type": "within_context",
                        "context_a": context,
                        "run_a": i + 1,
                        "context_b": context,
                        "run_b": j + 1,
                        **comparison,
                    })

        # Cross-context pairwise comparisons
        novpn_runs = rankings[
            ("novpn", model_name)
        ]

        vpn_runs = rankings[
            ("vpn", model_name)
        ]

        for i, novpn_ranking in enumerate(
            novpn_runs,
            start=1,
        ):
            for j, vpn_ranking in enumerate(
                vpn_runs,
                start=1,
            ):

                comparison = compare_rankings(
                    novpn_ranking,
                    vpn_ranking,
                )

                pairwise_rows.append({
                    "model": model_name,
                    "comparison_type": "cross_context",
                    "context_a": "novpn",
                    "run_a": i,
                    "context_b": "vpn",
                    "run_b": j,
                    **comparison,
                })

    pairwise_df = pd.DataFrame(pairwise_rows)

    pairwise_df.to_csv(
        RESULTS_DIR / "pairwise_stability.csv",
        index=False,
    )

    summary_rows = []

    numeric_metrics = [
        "spearman_rho",
        "kendall_tau",
        *[
            f"top{k}_overlap"
            for k in TOP_KS
        ],
        "mean_abs_rank_displacement",
    ]

    for model_name in MODELS:

        model_df = pairwise_df[
            pairwise_df["model"] == model_name
        ]

        # Separate within-context summaries
        for context in CONTEXTS:

            current = model_df[
                (model_df["comparison_type"] == "within_context") &
                (model_df["context_a"] == context)
            ]

            row = {
                "model": model_name,
                "comparison_type": "within_context",
                "comparison": context,
                "n_pairs": len(current),
            }

            for metric in numeric_metrics:
                row[f"{metric}_mean"] = float(
                    current[metric].mean()
                )

                row[f"{metric}_std"] = float(
                    current[metric].std(ddof=1)
                )

                row[f"{metric}_median"] = float(
                    current[metric].median()
                )

            summary_rows.append(row)

        # Cross-context summary
        cross = model_df[
            model_df["comparison_type"] == "cross_context"
        ]

        row = {
            "model": model_name,
            "comparison_type": "cross_context",
            "comparison": "novpn_vs_vpn",
            "n_pairs": len(cross),
        }

        for metric in numeric_metrics:
            row[f"{metric}_mean"] = float(
                cross[metric].mean()
            )

            row[f"{metric}_std"] = float(
                cross[metric].std(ddof=1)
            )

            row[f"{metric}_median"] = float(
                cross[metric].median()
            )

        summary_rows.append(row)

        # Pooled within-context vs cross-context
        within = model_df[
            model_df["comparison_type"] == "within_context"
        ]

        row = {
            "model": model_name,
            "comparison_type":
                "pooled_within_context_vs_cross_context",
            "comparison": "pooled_within_vs_cross",
            "n_pairs": len(within),
        }

        for metric in numeric_metrics:
            within_mean = float(
                within[metric].mean()
            )

            cross_mean = float(
                cross[metric].mean()
            )

            row[f"{metric}_within_mean"] = within_mean
            row[f"{metric}_cross_mean"] = cross_mean
            row[f"{metric}_cross_minus_within"] = (
                cross_mean - within_mean
            )

        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(
        RESULTS_DIR / "stability_summary.csv",
        index=False,
    )

    # Random Top-k overlap baseline:
    # E(overlap) = k^2 / F
    n_features = len(FEATURES)
    random_rows = []

    for k in TOP_KS:
        random_rows.append({
            "top_k": k,
            "n_features": n_features,
            "expected_random_overlap":
                (k * k) / n_features,
        })

    pd.DataFrame(random_rows).to_csv(
        RESULTS_DIR / "random_overlap_baseline.csv",
        index=False,
    )

    manifest = {
        "n_runs_per_context_model": args.runs,
        "contexts": list(CONTEXTS),
        "models": list(MODELS),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "software_versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "shap": shap.__version__,
        },
        "bootstrap": {
            "type": "class_stratified_with_replacement",
            "same_size_as_original_training_set": True,
            "seed_base": BOOTSTRAP_SEED_BASE,
        },
        "test_set": {
            "fixed_and_untouched": True,
            "novpn_n": len(data["novpn"]["X_test"]),
            "vpn_n": len(data["vpn"]["X_test"]),
        },
        "shap": {
            "explainer": "TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "check_additivity": False,
            "importance":
                "mean(abs(SHAP)) across classes and test samples",
        },
        "top_k_values": list(TOP_KS),
    }

    with open(
        RESULTS_DIR / "method_manifest.json",
        "w",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        f"\nTask D completed."
        f"\nOutputs written to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
