"""
Task F - SHAP Consequence Test for the ISCX VPN/Non-VPN dataset.

Leakage-safe design:
- Original train/test partitions are preserved.
- Feature selection uses ONLY an inner split of the original training data.
- SHAP is computed on the inner-validation set.
- Top-10 features are selected separately for each context/model.
- Final models are retrained on the FULL original training set.
- Original untouched test sets are used only for final evaluation.

For each target context/model, compare:
1. Full 23-feature model
2. Context-specific Top-10 SHAP subset
3. Transferred Top-10 SHAP subset from the other context

Primary metric: Macro-F1
Secondary metric: per-class recall
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import shap

from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results" / "consequence_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


CONTEXTS = ("novpn", "vpn")
MODELS = ("RandomForest", "XGBoost")

TOP_K = 10

INNER_TEST_SIZE = 0.20
INNER_SPLIT_SEED = 20261008


def load_context(context):
    data_dir = DATA_DIR / context

    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(data_dir / "y_test.csv").iloc[:, 0]

    if list(X_train.columns) != FEATURES:
        raise ValueError(
            f"{context}: X_train columns do not match Task C features."
        )

    if list(X_test.columns) != FEATURES:
        raise ValueError(
            f"{context}: X_test columns do not match Task C features."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            f"{context}: training feature/label length mismatch."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"{context}: test feature/label length mismatch."
        )

    return X_train, X_test, y_train, y_test


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


def get_shap_ranking(model, X_inner_valid, n_classes):
    """
    Match Task C exactly:
    mean absolute SHAP averaged over classes and samples.
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
    )

    sv = explainer.shap_values(
        X_inner_valid,
        check_additivity=False,
    )

    sv = np.asarray(sv)

    if sv.ndim == 3:
        if (
            sv.shape[0] == len(X_inner_valid)
            and sv.shape[-1] == n_classes
        ):
            sv = np.transpose(sv, (2, 0, 1))

    elif sv.ndim == 2:
        sv = sv[None, :, :]

    expected_shape = (
        n_classes,
        len(X_inner_valid),
        X_inner_valid.shape[1],
    )

    if sv.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape {sv.shape}; "
            f"expected {expected_shape}."
        )

    mean_abs = np.mean(
        np.abs(sv),
        axis=(0, 1),
    )

    return pd.Series(
        mean_abs,
        index=X_inner_valid.columns,
        dtype=float,
    ).sort_values(ascending=False)


def select_features(context, model_name, X_train, y_train):
    """
    Perform feature selection using only an inner train/validation split.
    The outer/original test set is never touched here.
    """
    X_inner_train, X_inner_valid, y_inner_train, _ = train_test_split(
        X_train,
        y_train,
        test_size=INNER_TEST_SIZE,
        random_state=INNER_SPLIT_SEED,
        stratify=y_train,
    )

    model = create_model(model_name)

    model.fit(
        X_inner_train,
        y_inner_train,
    )

    ranking = get_shap_ranking(
        model,
        X_inner_valid,
        y_train.nunique(),
    )

    selected = list(ranking.head(TOP_K).index)

    ranking_table = pd.DataFrame({
        "context": context,
        "model": model_name,
        "feature": ranking.index,
        "rank": np.arange(1, len(ranking) + 1),
        "mean_abs_shap_inner_validation": ranking.values,
    })

    return selected, ranking_table


def evaluate_final_model(
    target_context,
    model_name,
    feature_set_name,
    selected_features,
    data,
):
    X_train = data[target_context]["X_train"]
    X_test = data[target_context]["X_test"]
    y_train = data[target_context]["y_train"]
    y_test = data[target_context]["y_test"]

    model = create_model(model_name)

    if feature_set_name == "full":
        features_used = FEATURES
    else:
        features_used = selected_features

    model.fit(
        X_train[features_used],
        y_train,
    )

    predictions = model.predict(
        X_test[features_used]
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    labels = sorted(
        y_test.unique()
    )

    _, recalls, _, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        labels=labels,
        zero_division=0,
    )

    per_class = {
        int(label): float(recall)
        for label, recall in zip(labels, recalls)
    }

    return {
        "target_context": target_context,
        "model": model_name,
        "feature_set": feature_set_name,
        "n_features": len(features_used),
        "features": "|".join(features_used),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_recall": per_class,
    }


def main():
    data = {}

    print("=== Task F: SHAP Consequence Test ===")

    for context in CONTEXTS:
        X_train, X_test, y_train, y_test = load_context(
            context
        )

        data[context] = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

        print(
            f"{context.upper()}: "
            f"train={X_train.shape}, "
            f"test={X_test.shape}"
        )

    # ---------------------------------------------------------
    # Step 1: Leakage-safe SHAP feature selection
    # ---------------------------------------------------------
    selected_features = {}
    ranking_tables = []

    for context in CONTEXTS:
        for model_name in MODELS:
            selected, ranking_table = select_features(
                context,
                model_name,
                data[context]["X_train"],
                data[context]["y_train"],
            )

            selected_features[(context, model_name)] = selected
            ranking_tables.append(ranking_table)

            print(
                f"\n{context.upper()} - {model_name}"
            )
            print(
                "Selected Top-10:"
            )

            for rank, feature in enumerate(
                selected,
                start=1,
            ):
                print(
                    f"{rank:2d}. {feature}"
                )

    pd.concat(
        ranking_tables,
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "inner_shap_rankings.csv",
        index=False,
    )

    # Save selected feature sets in a readable table.
    selection_rows = []

    for (context, model_name), features in selected_features.items():
        for rank, feature in enumerate(
            features,
            start=1,
        ):
            selection_rows.append({
                "context": context,
                "model": model_name,
                "rank": rank,
                "feature": feature,
            })

    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_DIR / "selected_top10_features.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Step 2: Final evaluation on untouched original test sets
    # ---------------------------------------------------------
    final_results = []
    per_class_rows = []

    for target_context in CONTEXTS:

        source_context = (
            "vpn"
            if target_context == "novpn"
            else "novpn"
        )

        for model_name in MODELS:

            own_features = selected_features[
                (target_context, model_name)
            ]

            transferred_features = selected_features[
                (source_context, model_name)
            ]

            evaluations = [
                (
                    "full",
                    FEATURES,
                ),
                (
                    "context_specific_top10",
                    own_features,
                ),
                (
                    "transferred_top10",
                    transferred_features,
                ),
            ]

            for feature_set_name, features in evaluations:

                result = evaluate_final_model(
                    target_context=target_context,
                    model_name=model_name,
                    feature_set_name=feature_set_name,
                    selected_features=features,
                    data=data,
                )

                per_class = result.pop(
                    "per_class_recall"
                )

                final_results.append(result)

                for class_id, recall in per_class.items():
                    per_class_rows.append({
                        "target_context": target_context,
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "class": class_id,
                        "recall": recall,
                    })

                print(
                    f"\n{target_context.upper()} - "
                    f"{model_name} - "
                    f"{feature_set_name}: "
                    f"accuracy={result['accuracy']:.4f}, "
                    f"macro_f1={result['macro_f1']:.4f}"
                )

    results_df = pd.DataFrame(
        final_results
    )

    per_class_df = pd.DataFrame(
        per_class_rows
    )

    results_df.to_csv(
        OUTPUT_DIR / "consequence_results.csv",
        index=False,
    )

    per_class_df.to_csv(
        OUTPUT_DIR / "consequence_per_class_recall.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Step 3: Calculate practical consequences
    # ---------------------------------------------------------
    consequence_rows = []

    for target_context in CONTEXTS:
        for model_name in MODELS:

            current = results_df[
                (results_df["target_context"] == target_context)
                & (results_df["model"] == model_name)
            ].set_index("feature_set")

            full_f1 = float(
                current.loc["full", "macro_f1"]
            )

            own_f1 = float(
                current.loc[
                    "context_specific_top10",
                    "macro_f1",
                ]
            )

            transferred_f1 = float(
                current.loc[
                    "transferred_top10",
                    "macro_f1",
                ]
            )

            consequence_rows.append({
                "target_context": target_context,
                "model": model_name,
                "full_macro_f1": full_f1,
                "context_specific_macro_f1": own_f1,
                "transferred_macro_f1": transferred_f1,
                "context_specific_vs_full":
                    own_f1 - full_f1,
                "transferred_vs_full":
                    transferred_f1 - full_f1,
                "context_specific_minus_transferred":
                    own_f1 - transferred_f1,
            })

    consequence_df = pd.DataFrame(
        consequence_rows
    )

    consequence_df.to_csv(
        OUTPUT_DIR / "consequence_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Step 4: Compare the selected rankings themselves
    # ---------------------------------------------------------
    ranking_comparison_rows = []

    for model_name in MODELS:
        own = selected_features[
            ("novpn", model_name)
        ]

        vpn = selected_features[
            ("vpn", model_name)
        ]

        common = len(
            set(own) & set(vpn)
        )

        # Compare rankings over the 23 features by recreating
        # complete rank vectors from the saved inner rankings.
        novpn_ranking = (
            pd.concat(ranking_tables)
            .loc[
                lambda df:
                (df["context"] == "novpn")
                & (df["model"] == model_name)
            ]
            .sort_values("rank")
        )

        vpn_ranking = (
            pd.concat(ranking_tables)
            .loc[
                lambda df:
                (df["context"] == "vpn")
                & (df["model"] == model_name)
            ]
            .sort_values("rank")
        )

        novpn_rank_map = dict(
            zip(
                novpn_ranking["feature"],
                novpn_ranking["rank"],
            )
        )

        vpn_rank_map = dict(
            zip(
                vpn_ranking["feature"],
                vpn_ranking["rank"],
            )
        )

        rank_a = np.array([
            novpn_rank_map[f]
            for f in FEATURES
        ])

        rank_b = np.array([
            vpn_rank_map[f]
            for f in FEATURES
        ])

        rho, _ = spearmanr(
            rank_a,
            rank_b,
        )

        ranking_comparison_rows.append({
            "model": model_name,
            "inner_top10_overlap": common,
            "inner_top10_overlap_fraction":
                common / TOP_K,
            "inner_spearman_rho": float(rho),
        })

    pd.DataFrame(
        ranking_comparison_rows
    ).to_csv(
        OUTPUT_DIR / "inner_cross_context_selection_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Method manifest
    # ---------------------------------------------------------
    manifest = {
        "task": "F - SHAP Consequence Test",
        "contexts": list(CONTEXTS),
        "models": list(MODELS),
        "n_features_full": len(FEATURES),
        "top_k": TOP_K,
        "inner_split": {
            "type": "stratified_train_validation_split",
            "validation_fraction": INNER_TEST_SIZE,
            "random_state": INNER_SPLIT_SEED,
        },
        "selection_rule": (
            "Top-10 features ranked by mean absolute TreeSHAP "
            "on the inner-validation set"
        ),
        "shap": {
            "explainer": "TreeExplainer",
            "feature_perturbation": "tree_path_dependent",
            "check_additivity": False,
            "aggregation":
                "mean absolute SHAP across classes and samples",
        },
        "final_training": (
            "Each selected feature set is retrained on the full "
            "original training partition of the target context."
        ),
        "final_evaluation": (
            "Original held-out test sets are used only for final evaluation."
        ),
        "feature_conditions": [
            "full",
            "context_specific_top10",
            "transferred_top10",
        ],
        "primary_metric": "macro_f1",
        "secondary_metric": "per_class_recall",
    }

    with open(
        OUTPUT_DIR / "method_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(
        "\nTask F completed."
        f"\nOutputs written to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
