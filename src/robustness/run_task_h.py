from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import shap

from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "robustness"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BOOTSTRAP_SEED = 20261008

N_CLASSES = 7
SHAP_TEST_N = 500
BACKGROUND_N = 200
BOOTSTRAP_N = 2000

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

CONTEXTS = ["novpn", "vpn"]

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


def load_data(context):
    base = DATA_DIR / context

    X_train = pd.read_csv(base / "X_train.csv")[FEATURES]
    X_test = pd.read_csv(base / "X_test.csv")[FEATURES]
    y_train = pd.read_csv(base / "y_train.csv").iloc[:, 0]
    y_test = pd.read_csv(base / "y_test.csv").iloc[:, 0]

    return X_train, X_test, y_train, y_test


def stratified_sample(X, y, n_rows, seed):
    n_rows = min(n_rows, len(X))

    if n_rows == len(X):
        return X.copy()

    X_sample, _, _, _ = train_test_split(
        X,
        y,
        train_size=n_rows,
        stratify=y,
        random_state=seed,
    )

    return X_sample.copy()


def build_model(model_name):
    if model_name == "RF":
        return RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=1,
            bootstrap=True,
            class_weight=None,
            random_state=SEED,
            n_jobs=-1,
        )

    if model_name == "XGB":
        return XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            enable_categorical=False,
            objective="multi:softprob",
            num_class=N_CLASSES,
            eval_metric="mlogloss",
            random_state=SEED,
            n_jobs=-1,
        )

    if model_name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            random_state=SEED,
            n_jobs=-1,
        )

    raise ValueError(f"Unknown model: {model_name}")


def evaluate_model(model, X_test, y_test):
    pred = model.predict(X_test)

    return {
        "accuracy": float(
            accuracy_score(y_test, pred)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0,
            )
        ),
    }


def shap_matrix(model, X, method, background=None):
    if method == "tree_path_dependent":
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )

    elif method == "interventional":
        if background is None:
            raise ValueError(
                "Background data required for interventional SHAP."
            )

        masker = shap.maskers.Independent(
            background,
            max_samples=len(background),
        )

        explainer = shap.TreeExplainer(
            model,
            data=masker,
            feature_perturbation="interventional",
            model_output="raw",
        )

    else:
        raise ValueError(f"Unknown SHAP method: {method}")

    values = explainer.shap_values(X)

    if isinstance(values, (list, tuple)):
        arr = np.stack(values, axis=-1)
    else:
        arr = np.asarray(values)

    if arr.ndim == 2:
        if arr.shape != (len(X), len(FEATURES)):
            raise ValueError(
                f"Unexpected SHAP shape: {arr.shape}"
            )

        result = np.abs(arr)

    elif arr.ndim == 3:
        feature_axes = [
            i for i, size in enumerate(arr.shape)
            if size == len(FEATURES)
        ]

        if len(feature_axes) != 1:
            raise ValueError(
                f"Cannot identify feature axis in SHAP shape: {arr.shape}"
            )

        feature_axis = feature_axes[0]

        arr = np.moveaxis(
            arr,
            feature_axis,
            -1,
        )

        # samples x classes x features
        if arr.shape[0] != len(X):
            raise ValueError(
                f"Unexpected sample axis in SHAP shape: {arr.shape}"
            )

        result = np.mean(
            np.abs(arr),
            axis=1,
        )

    else:
        raise ValueError(
            f"Unsupported SHAP output shape: {arr.shape}"
        )

    if result.shape != (len(X), len(FEATURES)):
        raise ValueError(
            f"Final SHAP matrix has wrong shape: {result.shape}"
        )

    return result


def ranking_from_matrix(matrix):
    importance = matrix.mean(axis=0)

    ranking = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs_shap": importance,
    })

    ranking = ranking.sort_values(
        ["mean_abs_shap", "feature"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # Average rank handles ties correctly.
    rank_lookup = (
        ranking["mean_abs_shap"]
        .rank(
            method="average",
            ascending=False,
        )
    )

    ranking["rank"] = rank_lookup.to_numpy()

    return ranking


def compare_rankings(rank_a, rank_b):
    a = (
        rank_a
        .set_index("feature")
        .loc[FEATURES]["rank"]
        .to_numpy()
    )

    b = (
        rank_b
        .set_index("feature")
        .loc[FEATURES]["rank"]
        .to_numpy()
    )

    rho = spearmanr(a, b).statistic
    tau = kendalltau(a, b).statistic

    top5_a = set(
        rank_a.sort_values(
            ["mean_abs_shap", "feature"],
            ascending=[False, True],
        ).head(5)["feature"]
    )

    top5_b = set(
        rank_b.sort_values(
            ["mean_abs_shap", "feature"],
            ascending=[False, True],
        ).head(5)["feature"]
    )

    top10_a = set(
        rank_a.sort_values(
            ["mean_abs_shap", "feature"],
            ascending=[False, True],
        ).head(10)["feature"]
    )

    top10_b = set(
        rank_b.sort_values(
            ["mean_abs_shap", "feature"],
            ascending=[False, True],
        ).head(10)["feature"]
    )

    return {
        "spearman": float(rho),
        "kendall": float(tau),
        "rank_displacement": float(
            np.mean(np.abs(a - b))
        ),
        "top5_overlap": int(
            len(top5_a & top5_b)
        ),
        "top10_overlap": int(
            len(top10_a & top10_b)
        ),
    }


def family_ranking(feature_ranking):
    rows = []

    for family, members in FEATURE_GROUPS.items():
        value = float(
            feature_ranking.loc[
                feature_ranking["feature"].isin(members),
                "mean_abs_shap",
            ].sum()
        )

        rows.append({
            "family": family,
            "mean_abs_shap": value,
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["mean_abs_shap", "family"],
        ascending=[False, True],
    ).reset_index(drop=True)

    result["normalized_contribution"] = (
        result["mean_abs_shap"]
        / result["mean_abs_shap"].sum()
    )

    result["rank"] = (
        result["mean_abs_shap"]
        .rank(
            method="average",
            ascending=False,
        )
        .to_numpy()
    )

    return result


def compare_family_rankings(a, b):
    a_rank = (
        a
        .set_index("family")
        .loc[list(FEATURE_GROUPS.keys())]["rank"]
        .to_numpy()
    )

    b_rank = (
        b
        .set_index("family")
        .loc[list(FEATURE_GROUPS.keys())]["rank"]
        .to_numpy()
    )

    rho = spearmanr(
        a_rank,
        b_rank,
    ).statistic

    top2_a = set(a.head(2)["family"])
    top2_b = set(b.head(2)["family"])

    return {
        "family_spearman": float(rho),
        "same_top2_families": bool(
            top2_a == top2_b
        ),
    }


def bootstrap_cross_context_rho(
    matrix_a,
    matrix_b,
    n_bootstrap,
    seed,
):
    rng = np.random.default_rng(seed)

    n_a = matrix_a.shape[0]
    n_b = matrix_b.shape[0]

    values = []

    for _ in range(n_bootstrap):
        idx_a = rng.integers(
            0,
            n_a,
            size=n_a,
        )

        idx_b = rng.integers(
            0,
            n_b,
            size=n_b,
        )

        mean_a = matrix_a[idx_a].mean(axis=0)
        mean_b = matrix_b[idx_b].mean(axis=0)

        rank_a = (
            pd.Series(mean_a)
            .rank(method="average", ascending=False)
            .to_numpy()
        )

        rank_b = (
            pd.Series(mean_b)
            .rank(method="average", ascending=False)
            .to_numpy()
        )

        rho = spearmanr(
            rank_a,
            rank_b,
        ).statistic

        values.append(rho)

    values = np.asarray(values)

    return (
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


def main():
    total_start = time.perf_counter()

    matrices = {}
    rankings = {}
    family_rankings = {}

    performance_rows = []
    ranking_rows = []
    family_rows = []

    print("Starting Task H...")

    # ----------------------------------------------------------
    # Train every model and compute both SHAP formulations.
    # ----------------------------------------------------------
    for context in CONTEXTS:
        print(
            f"\n===== {context.upper()} ====="
        )

        X_train, X_test, y_train, y_test = load_data(
            context
        )

        X_background = stratified_sample(
            X_train,
            y_train,
            BACKGROUND_N,
            SEED,
        )

        X_shap = stratified_sample(
            X_test,
            y_test,
            SHAP_TEST_N,
            SEED,
        )

        print(
            f"Background rows: {len(X_background)}"
        )

        print(
            f"SHAP test rows: {len(X_shap)}"
        )

        for model_name in [
            "RF",
            "XGB",
            "ExtraTrees",
        ]:
            print(
                f"\n--- {context} | {model_name} ---"
            )

            model = build_model(
                model_name
            )

            train_start = time.perf_counter()

            model.fit(
                X_train,
                y_train,
            )

            train_time = (
                time.perf_counter()
                - train_start
            )

            metrics = evaluate_model(
                model,
                X_test,
                y_test,
            )

            print(
                f"Accuracy={metrics['accuracy']:.4f} "
                f"Macro-F1={metrics['macro_f1']:.4f}"
            )

            performance_rows.append({
                "context": context,
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "train_seconds": train_time,
            })

            for method in [
                "tree_path_dependent",
                "interventional",
            ]:
                print(
                    f"{method} SHAP running..."
                )

                shap_start = time.perf_counter()

                matrix = shap_matrix(
                    model,
                    X_shap,
                    method,
                    background=(
                        X_background
                        if method == "interventional"
                        else None
                    ),
                )

                shap_time = (
                    time.perf_counter()
                    - shap_start
                )

                ranking = ranking_from_matrix(
                    matrix
                )

                family = family_ranking(
                    ranking
                )

                key = (
                    context,
                    model_name,
                    method,
                )

                matrices[key] = matrix
                rankings[key] = ranking
                family_rankings[key] = family

                for _, row in ranking.iterrows():
                    ranking_rows.append({
                        "context": context,
                        "model": model_name,
                        "method": method,
                        "feature": row["feature"],
                        "rank": float(row["rank"]),
                        "mean_abs_shap": float(
                            row["mean_abs_shap"]
                        ),
                        "shap_seconds": shap_time,
                    })

                for _, row in family.iterrows():
                    family_rows.append({
                        "context": context,
                        "model": model_name,
                        "method": method,
                        "family": row["family"],
                        "rank": float(row["rank"]),
                        "mean_abs_shap": float(
                            row["mean_abs_shap"]
                        ),
                        "normalized_contribution": float(
                            row["normalized_contribution"]
                        ),
                    })

    # ----------------------------------------------------------
    # Cross-context comparison for all six configurations.
    # ----------------------------------------------------------
    configs = [
        ("RF", "tree_path_dependent"),
        ("XGB", "tree_path_dependent"),
        ("RF", "interventional"),
        ("XGB", "interventional"),
        ("ExtraTrees", "tree_path_dependent"),
        ("ExtraTrees", "interventional"),
    ]

    cross_rows = []

    for model_name, method in configs:
        print(
            f"\n===== CROSS-CONTEXT | "
            f"{model_name} | {method} ====="
        )

        key_nv = (
            "novpn",
            model_name,
            method,
        )

        key_vpn = (
            "vpn",
            model_name,
            method,
        )

        ranking_nv = rankings[key_nv]
        ranking_vpn = rankings[key_vpn]

        comparison = compare_rankings(
            ranking_nv,
            ranking_vpn,
        )

        family_comparison = compare_family_rankings(
            family_rankings[key_nv],
            family_rankings[key_vpn],
        )

        ci_low, ci_high = (
            bootstrap_cross_context_rho(
                matrices[key_nv],
                matrices[key_vpn],
                BOOTSTRAP_N,
                BOOTSTRAP_SEED,
            )
        )

        row = {
            "model": model_name,
            "method": method,
            **comparison,
            "spearman_ci_lower": ci_low,
            "spearman_ci_upper": ci_high,
            **family_comparison,
        }

        cross_rows.append(row)

        print(
            f"Spearman={row['spearman']:.4f} "
            f"95% CI=[{ci_low:.4f}, {ci_high:.4f}] "
            f"Kendall={row['kendall']:.4f} "
            f"Top5={row['top5_overlap']}/5 "
            f"Top10={row['top10_overlap']}/10 "
            f"Family rho={row['family_spearman']:.4f}"
        )

    # ----------------------------------------------------------
    # Alternative configuration vs primary ranking.
    # ----------------------------------------------------------
    agreement_specs = [
        (
            "RF",
            "interventional",
            "RF",
            "tree_path_dependent",
        ),
        (
            "XGB",
            "interventional",
            "XGB",
            "tree_path_dependent",
        ),
        (
            "ExtraTrees",
            "tree_path_dependent",
            "RF",
            "tree_path_dependent",
        ),
        (
            "ExtraTrees",
            "interventional",
            "RF",
            "tree_path_dependent",
        ),
    ]

    agreement_rows = []

    for alt_model, alt_method, ref_model, ref_method in (
        agreement_specs
    ):
        for context in CONTEXTS:
            alternative = rankings[
                (
                    context,
                    alt_model,
                    alt_method,
                )
            ]

            reference = rankings[
                (
                    context,
                    ref_model,
                    ref_method,
                )
            ]

            comparison = compare_rankings(
                reference,
                alternative,
            )

            agreement_rows.append({
                "context": context,
                "reference_model": ref_model,
                "reference_method": ref_method,
                "alternative_model": alt_model,
                "alternative_method": alt_method,
                **comparison,
            })

    # ----------------------------------------------------------
    # Save separate output tables.
    # ----------------------------------------------------------
    performance_df = pd.DataFrame(
        performance_rows
    )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    family_df = pd.DataFrame(
        family_rows
    )

    cross_df = pd.DataFrame(
        cross_rows
    )

    agreement_df = pd.DataFrame(
        agreement_rows
    )

    performance_df.to_csv(
        RESULTS_DIR / "alternative_model_results.csv",
        index=False,
    )

    ranking_df.to_csv(
        RESULTS_DIR / "robustness_rankings.csv",
        index=False,
    )

    family_df.to_csv(
        RESULTS_DIR / "robustness_family_rankings.csv",
        index=False,
    )

    cross_df.to_csv(
        RESULTS_DIR / "cross_context_robustness.csv",
        index=False,
    )

    agreement_df.to_csv(
        RESULTS_DIR / "configuration_agreement.csv",
        index=False,
    )

    manifest = {
        "task": "H",
        "scope": "Provided-split robustness of cross-context SHAP comparison",
        "seed": SEED,
        "shap_test_rows": SHAP_TEST_N,
        "interventional_background_rows": BACKGROUND_N,
        "bootstrap_replicates": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "contexts": CONTEXTS,
        "configurations": [
            {
                "model": "RF",
                "method": "tree_path_dependent",
            },
            {
                "model": "XGB",
                "method": "tree_path_dependent",
            },
            {
                "model": "RF",
                "method": "interventional",
            },
            {
                "model": "XGB",
                "method": "interventional",
            },
            {
                "model": "ExtraTrees",
                "method": "tree_path_dependent",
            },
            {
                "model": "ExtraTrees",
                "method": "interventional",
            },
        ],
        "models": {
            "RF": {
                "n_estimators": 200,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
                "bootstrap": True,
            },
            "XGB": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
            },
            "ExtraTrees": {
                "n_estimators": 500,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
            },
        },
        "shap": {
            "primary_method": "tree_path_dependent",
            "robustness_method": "interventional",
            "interventional_background": (
                "200 stratified training rows"
            ),
            "ranking": "23 original features",
            "rank_ties": "average rank",
        },
        "bootstrap": {
            "resampling": (
                "independent within each context; "
                "no VPN/Non-VPN row pairing assumed"
            ),
            "interpretation": (
                "uncertainty from resampling explained test "
                "records with fitted models and background fixed"
            ),
        },
    }

    with open(
        RESULTS_DIR / "method_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    summary = [
        "# Task H - SHAP Robustness",
        "",
        "## Cross-context robustness",
        cross_df.to_string(index=False),
        "",
        "## Alternative configuration agreement",
        agreement_df.to_string(index=False),
        "",
        "## Alternative model performance",
        performance_df.to_string(index=False),
        "",
        f"Total runtime: "
        f"{time.perf_counter() - total_start:.2f} seconds",
    ]

    (RESULTS_DIR / "robustness_summary.md").write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\nTask H completed successfully.")
    print(
        f"Results saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
