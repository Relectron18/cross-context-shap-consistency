from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import shap

from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier


RANDOM_STATE = 42
N_CLASSES = 7

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "sentinel"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "duration",
    "total_fiat",
    "total_biat",
    "min_fiat",
    "min_biat",
    "max_fiat",
    "max_biat",
    "mean_fiat",
    "mean_biat",
    "flowPktsPerSecond",
    "flowBytesPerSecond",
    "min_flowiat",
    "max_flowiat",
    "mean_flowiat",
    "std_flowiat",
    "min_active",
    "mean_active",
    "max_active",
    "std_active",
    "min_idle",
    "mean_idle",
    "max_idle",
    "std_idle",
]

SENTINEL_FEATURES = [
    "min_fiat",
    "min_biat",
    "max_fiat",
    "max_biat",
    "min_flowiat",
    "max_flowiat",
    "min_active",
    "max_active",
    "min_idle",
    "max_idle",
]

FEATURE_GROUPS = {
    "Duration": ["duration"],
    "Directional IAT": [
        "total_fiat",
        "total_biat",
        "min_fiat",
        "min_biat",
        "max_fiat",
        "max_biat",
        "mean_fiat",
        "mean_biat",
    ],
    "Flow throughput": [
        "flowPktsPerSecond",
        "flowBytesPerSecond",
    ],
    "Flow IAT": [
        "min_flowiat",
        "max_flowiat",
        "mean_flowiat",
        "std_flowiat",
    ],
    "Active": [
        "min_active",
        "mean_active",
        "max_active",
        "std_active",
    ],
    "Idle": [
        "min_idle",
        "mean_idle",
        "max_idle",
        "std_idle",
    ],
}

CONTEXTS = ("novpn", "vpn")
MODELS = ("RF", "XGB")
CONDITIONS = ("S1", "S2")


def build_sentinel_data(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Create the 33-feature sentinel representation."""
    X = df[FEATURES].copy()

    # Create indicators from the original -1 representation.
    indicators = X[SENTINEL_FEATURES].eq(-1).astype(np.int8)
    indicators.columns = [f"{c}_sentinel" for c in SENTINEL_FEATURES]

    X = pd.concat([X, indicators], axis=1)

    if condition == "S2":
        X[SENTINEL_FEATURES] = X[SENTINEL_FEATURES].replace(-1, np.nan)
    elif condition != "S1":
        raise ValueError(f"Unknown condition: {condition}")

    return X


def make_model(model_name: str):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if model_name == "XGB":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=N_CLASSES,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def shap_mean_abs(values, n_features: int) -> np.ndarray:
    """
    Return one mean-absolute-SHAP value per feature.

    Handles SHAP layouts:
      2D: samples x features
      3D: samples x features x classes
      3D: samples x classes x features
    """
    arr = np.asarray(values)

    if arr.ndim == 2:
        if arr.shape[1] != n_features:
            raise ValueError(
                f"Unexpected 2D SHAP shape {arr.shape}; "
                f"expected feature axis of size {n_features}."
            )
        return np.mean(np.abs(arr), axis=0)

    if arr.ndim == 3:
        feature_axes = [i for i, size in enumerate(arr.shape) if size == n_features]

        if len(feature_axes) != 1:
            raise ValueError(
                f"Could not uniquely identify feature axis in SHAP shape {arr.shape} "
                f"for {n_features} features."
            )

        feature_axis = feature_axes[0]
        other_axes = tuple(i for i in range(arr.ndim) if i != feature_axis)
        return np.mean(np.abs(arr), axis=other_axes)

    raise ValueError(f"Unsupported SHAP output shape: {arr.shape}")


def compute_shap_importance(model, X_test: pd.DataFrame) -> pd.DataFrame:
    """Compute global mean absolute TreeSHAP importance."""
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )
    shap_values = explainer.shap_values(X_test)
    mean_abs = shap_mean_abs(shap_values, X_test.shape[1])

    ranking = pd.DataFrame(
        {
            "feature": X_test.columns,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def evaluate_model(model, X_test, y_test):
    """Return overall metrics and per-class recall."""
    pred = model.predict(X_test)

    overall = {
        "accuracy": accuracy_score(y_test, pred),
        "weighted_precision": precision_score(
            y_test, pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_test, pred, average="weighted", zero_division=0
        ),
        "macro_f1": f1_score(
            y_test, pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_test, pred, average="weighted", zero_division=0
        ),
    }

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0,
    )

    recalls = []
    for cls in range(N_CLASSES):
        cls_key = str(cls)
        recalls.append(
            {
                "class": cls,
                "recall": report.get(cls_key, {}).get("recall", 0.0),
            }
        )

    return overall, recalls


def rank_metrics(rank_a: pd.DataFrame, rank_b: pd.DataFrame):
    """Compare rankings using only the original 23 features."""
    a = rank_a.set_index("feature").loc[FEATURES]["rank"]
    b = rank_b.set_index("feature").loc[FEATURES]["rank"]

    spearman = spearmanr(a, b).statistic
    kendall = kendalltau(a, b).statistic
    rank_displacement = float(np.mean(np.abs(a.values - b.values)))

    top5_a = set(rank_a[rank_a["feature"].isin(FEATURES)].head(5)["feature"])
    top5_b = set(rank_b[rank_b["feature"].isin(FEATURES)].head(5)["feature"])

    top10_a = set(rank_a[rank_a["feature"].isin(FEATURES)].head(10)["feature"])
    top10_b = set(rank_b[rank_b["feature"].isin(FEATURES)].head(10)["feature"])

    return {
        "spearman": float(spearman),
        "kendall": float(kendall),
        "rank_displacement": rank_displacement,
        "top5_overlap": len(top5_a & top5_b),
        "top10_overlap": len(top10_a & top10_b),
    }


def group_contributions(ranking: pd.DataFrame):
    """Calculate group-level contribution among the original 23 features."""
    original = ranking[ranking["feature"].isin(FEATURES)].copy()

    total_original = float(original["mean_abs_shap"].sum())

    rows = []
    for group, members in FEATURE_GROUPS.items():
        value = float(
            original.loc[
                original["feature"].isin(members),
                "mean_abs_shap",
            ].sum()
        )

        rows.append(
            {
                "group": group,
                "group_mean_abs_shap": value,
                "normalized_group_contribution": (
                    value / total_original if total_original > 0 else 0.0
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "group_mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)


def main():
    all_results = []
    per_class_rows = []
    ranking_rows = []
    ranking_objects = {}
    group_rows = []
    indicator_rows = []

    manifest = {
        "task": "G",
        "title": "Sentinel robustness analysis",
        "random_state": RANDOM_STATE,
        "conditions": {
            "S1": "Retain -1 and add sentinel indicators",
            "S2": "Convert -1 to NaN and retain the same sentinel indicators",
        },
        "original_feature_count": len(FEATURES),
        "sentinel_feature_count": len(SENTINEL_FEATURES),
        "total_feature_count": len(FEATURES) + len(SENTINEL_FEATURES),
        "sentinel_features": SENTINEL_FEATURES,
        "models": {
            "RF": {
                "n_estimators": 200,
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
            "XGB": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "objective": "multi:softprob",
                "num_class": N_CLASSES,
                "eval_metric": "mlogloss",
                "random_state": RANDOM_STATE,
                "n_jobs": -1,
            },
        },
        "shap": {
            "explainer": "TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "importance": "mean absolute SHAP",
        },
        "comparison": {
            "main_rankings": "original 23 features only",
            "top_k": [5, 10],
            "group_analysis": "original 23 features only",
            "indicator_analysis": "10 sentinel indicators reported separately",
        },
    }

    start_all = time.perf_counter()

    for context in CONTEXTS:
        print(f"\n===== {context.upper()} =====")

        X_train_raw = pd.read_csv(DATA_DIR / context / "X_train.csv")
        X_test_raw = pd.read_csv(DATA_DIR / context / "X_test.csv")
        y_train = pd.read_csv(DATA_DIR / context / "y_train.csv").iloc[:, 0]
        y_test = pd.read_csv(DATA_DIR / context / "y_test.csv").iloc[:, 0]

        for model_name in MODELS:
            condition_rankings = {}

            for condition in CONDITIONS:
                print(f"\n[{context} | {model_name} | {condition}]")

                X_train = build_sentinel_data(X_train_raw, condition)
                X_test = build_sentinel_data(X_test_raw, condition)

                print(f"Training with {X_train.shape[1]} features...")
                t0 = time.perf_counter()

                model = make_model(model_name)
                model.fit(X_train, y_train)

                train_seconds = time.perf_counter() - t0

                overall, recalls = evaluate_model(
                    model,
                    X_test,
                    y_test,
                )

                print(
                    f"Macro-F1={overall['macro_f1']:.4f}, "
                    f"Accuracy={overall['accuracy']:.4f}, "
                    f"train_time={train_seconds:.2f}s"
                )

                ranking = compute_shap_importance(model, X_test)
                condition_rankings[condition] = ranking
                ranking_objects[(context, model_name, condition)] = ranking

                # Overall results.
                all_results.append(
                    {
                        "context": context,
                        "model": model_name,
                        "condition": condition,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "n_features": X_train.shape[1],
                        "train_seconds": train_seconds,
                        **overall,
                    }
                )

                # Per-class recall.
                for row in recalls:
                    per_class_rows.append(
                        {
                            "context": context,
                            "model": model_name,
                            "condition": condition,
                            **row,
                        }
                    )

                # Full ranking.
                for _, row in ranking.iterrows():
                    ranking_rows.append(
                        {
                            "context": context,
                            "model": model_name,
                            "condition": condition,
                            "feature": row["feature"],
                            "rank": int(row["rank"]),
                            "mean_abs_shap": float(row["mean_abs_shap"]),
                            "feature_type": (
                                "original"
                                if row["feature"] in FEATURES
                                else "sentinel_indicator"
                            ),
                        }
                    )

                # Group contribution among the original 23 features.
                groups = group_contributions(ranking)
                for _, row in groups.iterrows():
                    group_rows.append(
                        {
                            "context": context,
                            "model": model_name,
                            "condition": condition,
                            **row.to_dict(),
                        }
                    )

                # Total contribution of the 10 sentinel indicators.
                indicator_mask = ranking["feature"].isin(
                    [f"{c}_sentinel" for c in SENTINEL_FEATURES]
                )
                original_mask = ranking["feature"].isin(FEATURES)

                total_all = float(ranking["mean_abs_shap"].sum())
                indicator_total = float(
                    ranking.loc[indicator_mask, "mean_abs_shap"].sum()
                )
                original_total = float(
                    ranking.loc[original_mask, "mean_abs_shap"].sum()
                )

                indicator_rows.append(
                    {
                        "context": context,
                        "model": model_name,
                        "condition": condition,
                        "indicator_total_mean_abs_shap": indicator_total,
                        "original_23_total_mean_abs_shap": original_total,
                        "indicator_fraction_of_all_shap": (
                            indicator_total / total_all
                            if total_all > 0
                            else 0.0
                        ),
                    }
                )

            # Compare S1 vs S2 for this context/model.
            metrics = rank_metrics(
                condition_rankings["S1"],
                condition_rankings["S2"],
            )

            ranking_rows_key = {
                "context": context,
                "model": model_name,
                "comparison": "S1_vs_S2",
                **metrics,
            }

            pd.DataFrame([ranking_rows_key]).to_csv(
                RESULTS_DIR
                / f"sentinel_ranking_comparison_{context}_{model_name}.csv",
                index=False,
            )


    # Consolidated ranking comparison.
    comparison_rows = []
    for context in CONTEXTS:
        for model_name in MODELS:
            metrics = rank_metrics(
                ranking_objects[(context, model_name, "S1")],
                ranking_objects[(context, model_name, "S2")],
            )
            comparison_rows.append(
                {
                    "context": context,
                    "model": model_name,
                    "comparison": "S1_vs_S2",
                    **metrics,
                }
            )

    results_df = pd.DataFrame(all_results)
    recall_df = pd.DataFrame(per_class_rows)
    ranking_df = pd.DataFrame(ranking_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    group_df = pd.DataFrame(group_rows)
    indicator_df = pd.DataFrame(indicator_rows)

    results_df.to_csv(
        RESULTS_DIR / "sentinel_results.csv",
        index=False,
    )
    recall_df.to_csv(
        RESULTS_DIR / "sentinel_per_class_recall.csv",
        index=False,
    )
    ranking_df.to_csv(
        RESULTS_DIR / "sentinel_rankings.csv",
        index=False,
    )
    comparison_df.to_csv(
        RESULTS_DIR / "sentinel_ranking_comparison.csv",
        index=False,
    )
    group_df.to_csv(
        RESULTS_DIR / "sentinel_group_contributions.csv",
        index=False,
    )
    indicator_df.to_csv(
        RESULTS_DIR / "sentinel_indicator_contributions.csv",
        index=False,
    )

    with open(RESULTS_DIR / "method_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Human-readable summary.
    summary_lines = [
        "# Task G — Sentinel Analysis",
        "",
        "## Experimental conditions",
        "- S1: retain `-1` and add sentinel indicators.",
        "- S2: convert `-1` to `NaN` and retain the same indicators.",
        "- RF and XGBoost are retrained separately for Non-VPN and VPN.",
        "- Evaluation uses the untouched original test labels and rows.",
        "- SHAP ranking comparisons are restricted to the original 23 features.",
        "",
        "## Performance summary",
        results_df[
            [
                "context",
                "model",
                "condition",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        ].to_string(index=False),
        "",
        "## S1 vs S2 SHAP ranking stability",
        comparison_df.to_string(index=False),
        "",
        "## Sentinel-indicator contribution",
        indicator_df.to_string(index=False),
        "",
        f"Total runtime: {time.perf_counter() - start_all:.2f} seconds",
    ]

    (RESULTS_DIR / "sentinel_summary.md").write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\nTask G completed.")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
