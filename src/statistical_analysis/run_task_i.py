"""Task I: run-level SHAP stability inference and Top-10 consequence bootstrap.

Save this file as src/statistical_analysis/run_task_i.py in the repository.
Modes:
  --mode stability     quick; uses existing Task D/E/F CSV outputs only
  --mode consequence   refits the 12 final Task F models, then bootstraps test records
  --mode all           run both (default)

Caveats built into the report:
- Run-level bootstrap intervals quantify variability across the 30 Task D training
  bootstrap runs; pairwise rows are not treated as independent observations.
- Family rankings sum member-feature mean absolute SHAP importances. This is an
  aggregated family-importance analysis, not a new Shapley-value calculation over
  grouped features.
- Task F used Top-10 features. Any H3-style inference here is exploratory and does
  not test a pre-specified Top-15 experiment.
- Consequence bootstrap intervals condition on fitted models and selected features;
  they quantify test-record sampling uncertainty, not training/selection variability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
D_DIR = RESULTS_DIR / "stability"
E_DIR = RESULTS_DIR / "group_shap"
F_DIR = RESULTS_DIR / "consequence_test"
OUT_DIR = RESULTS_DIR / "statistical_analysis"

CONTEXTS = ("novpn", "vpn")
MODELS = ("RandomForest", "XGBoost")
FEATURES = [
    "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond", "min_flowiat", "max_flowiat",
    "mean_flowiat", "std_flowiat", "min_active", "mean_active", "max_active",
    "std_active", "min_idle", "mean_idle", "max_idle", "std_idle",
]
N_BOOT = 3000
N_PERM = 20000
SEED = 20261009
FEATURE_METRICS = [
    "spearman_rho", "kendall_tau", "top5_overlap", "top10_overlap",
    "top12_overlap", "top15_overlap", "top18_overlap",
    "mean_abs_rank_displacement",
]
FAMILY_METRICS = [
    "spearman_rho", "kendall_tau", "top2_overlap", "top3_overlap",
    "mean_abs_rank_displacement",
]


def bh_adjust(p_values):
    """Benjamini-Hochberg adjusted p-values, preserving the input order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    pv = p[valid]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    sorted_p = pv[order]
    m = len(sorted_p)
    adjusted = sorted_p * m / np.arange(1, m + 1, dtype=float)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def metric_diagonal(metric):
    if metric in ("spearman_rho", "kendall_tau"):
        return 1.0
    if metric.startswith("top") and metric.endswith("_overlap"):
        return float(metric[len("top"):-len("_overlap")])
    return 0.0


def mean_off_diagonal_pairwise(matrix):
    n = matrix.shape[0]
    return float(matrix[np.triu_indices(n, k=1)].mean())


def bootstrap_gap(matrix_novpn, matrix_vpn, matrix_cross, metric, rng):
    """Estimate the within-vs-cross gap by independently resampling context runs.

    The 30 run IDs are not treated as matched experimental units merely because
    Task D used the same numeric seeds in both contexts. Each context's run list
    is resampled independently; pairwise similarities are recomputed from the
    sampled run-level matrix entries, rather than resampling pairwise rows.

    A basic confidence interval and one-sided null-centred bootstrap test are
    used. Because the statistic is pairwise, repeated sampled run IDs can appear
    as off-diagonal self-comparisons. Bootstrap errors are therefore centered
    at the empirical bootstrap mean before interval and tail calculations.
    This is an approximate bootstrap test, not a permutation test. The finite
    Monte Carlo resolution is 1 / (N_BOOT + 1).
    """
    n_novpn = matrix_novpn.shape[0]
    n_vpn = matrix_vpn.shape[0]
    if matrix_novpn.shape != (n_novpn, n_novpn):
        raise ValueError("Non-VPN within-context matrix must be square.")
    if matrix_vpn.shape != (n_vpn, n_vpn):
        raise ValueError("VPN within-context matrix must be square.")
    if matrix_cross.shape != (n_novpn, n_vpn):
        raise ValueError("Cross-context matrix shape must match the two context run counts.")
    if n_novpn < 2 or n_vpn < 2:
        raise ValueError("At least two runs per context are required.")

    within_novpn = mean_off_diagonal_pairwise(matrix_novpn)
    within_vpn = mean_off_diagonal_pairwise(matrix_vpn)
    within_point = (within_novpn + within_vpn) / 2.0
    cross_point = float(matrix_cross.mean())
    # Positive always means: more stable within context than across contexts.
    if metric == "mean_abs_rank_displacement":
        point_gap = cross_point - within_point
    else:
        point_gap = within_point - cross_point

    tri_n = np.triu_indices(n_novpn, k=1)
    tri_v = np.triu_indices(n_vpn, k=1)
    draws = np.empty(N_BOOT, dtype=float)
    for b in range(N_BOOT):
        # Independent run-level resampling for the two contexts.
        idx_n = rng.integers(0, n_novpn, size=n_novpn)
        idx_v = rng.integers(0, n_vpn, size=n_vpn)
        mn = matrix_novpn[np.ix_(idx_n, idx_n)]
        mv = matrix_vpn[np.ix_(idx_v, idx_v)]
        within_b = (
            float(mn[tri_n].mean()) + float(mv[tri_v].mean())
        ) / 2.0
        cross_b = float(matrix_cross[np.ix_(idx_n, idx_v)].mean())
        if metric == "mean_abs_rank_displacement":
            draws[b] = cross_b - within_b
        else:
            draws[b] = within_b - cross_b

    # The statistic is pairwise (a U/V-statistic), so resampling the same run
    # twice can insert self-similarity on off-diagonal positions. Use a basic
    # bootstrap interval and center bootstrap errors at their empirical mean to
    # remove this finite-sample shift before inference.
    bootstrap_center = float(np.mean(draws))
    centered_errors = draws - bootstrap_center
    q_low, q_high = np.percentile(centered_errors, [2.5, 97.5])
    ci_low = point_gap - q_high
    ci_high = point_gap - q_low
    # One-sided null-centred bootstrap test of H0: gap <= 0 vs H1: gap > 0.
    extreme = int(np.count_nonzero(centered_errors >= point_gap - 1e-12))
    p_value = (extreme + 1) / (N_BOOT + 1)

    return {
        "within_novpn_mean": within_novpn,
        "within_vpn_mean": within_vpn,
        "within_mean": within_point,
        "cross_mean": cross_point,
        "stability_gap": float(point_gap),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
        "p_one_sided_raw": float(p_value),
        "n_bootstrap": N_BOOT,
        "p_value_method": "one-sided null-centred nonparametric run bootstrap",
        "gap_sign_convention": (
            "cross_minus_within" if metric == "mean_abs_rank_displacement"
            else "within_minus_cross"
        ),
    }


def make_feature_matrices(pairwise):
    """Read D's precomputed pairwise metrics into run-by-run matrices."""
    meta = pd.read_csv(D_DIR / "run_metadata.csv")
    run_counts = meta.groupby(["context", "model"])["run"].nunique()
    if not (run_counts == run_counts.iloc[0]).all():
        raise ValueError(f"Inconsistent Task D run counts: {run_counts.to_dict()}")
    n_runs = int(run_counts.iloc[0])

    matrices = {}
    for model in MODELS:
        rows_model = pairwise[pairwise["model"] == model]
        for metric in FEATURE_METRICS:
            per_context = {}
            for context in CONTEXTS:
                mat = np.full((n_runs, n_runs), np.nan, dtype=float)
                np.fill_diagonal(mat, metric_diagonal(metric))
                rows = rows_model[
                    (rows_model["comparison_type"] == "within_context")
                    & (rows_model["context_a"] == context)
                ]
                for row in rows.itertuples(index=False):
                    i, j = int(row.run_a) - 1, int(row.run_b) - 1
                    value = float(getattr(row, metric))
                    mat[i, j] = value
                    mat[j, i] = value
                if np.isnan(mat).any():
                    raise ValueError(f"Missing pairwise cells: {model}/{context}/{metric}")
                per_context[context] = mat
            cross = np.full((n_runs, n_runs), np.nan, dtype=float)
            rows = rows_model[rows_model["comparison_type"] == "cross_context"]
            for row in rows.itertuples(index=False):
                i, j = int(row.run_a) - 1, int(row.run_b) - 1
                cross[i, j] = float(getattr(row, metric))
            if np.isnan(cross).any():
                raise ValueError(f"Missing cross-context cells: {model}/{metric}")
            matrices[(model, CONTEXTS[0], metric)] = per_context[CONTEXTS[0]]
            matrices[(model, CONTEXTS[1], metric)] = per_context[CONTEXTS[1]]
            matrices[(model, "cross", metric)] = cross
    return matrices, n_runs


def compare_orders(order_a, order_b, metrics):
    """Compare two ordered lists (highest family/feature importance first)."""
    if len(order_a) != len(order_b) or set(order_a) != set(order_b):
        raise ValueError("Rankings must contain the same items.")
    universe = list(order_a)
    ra_map = {item: i + 1 for i, item in enumerate(order_a)}
    rb_map = {item: i + 1 for i, item in enumerate(order_b)}
    ra = np.asarray([ra_map[item] for item in universe], dtype=float)
    rb = np.asarray([rb_map[item] for item in universe], dtype=float)
    result = {
        "spearman_rho": float(spearmanr(ra, rb).statistic),
        "kendall_tau": float(kendalltau(ra, rb).statistic),
        "mean_abs_rank_displacement": float(np.mean(np.abs(ra - rb))),
    }
    for metric in metrics:
        if metric.startswith("top") and metric.endswith("_overlap"):
            k = int(metric[len("top"):-len("_overlap")])
            result[metric] = float(len(set(order_a[:k]) & set(order_b[:k])))
    return result


def family_rankings_and_matrices():
    run_rankings = pd.read_csv(D_DIR / "run_rankings.csv")
    with open(E_DIR / "feature_group_definitions.json", encoding="utf-8") as f:
        group_definitions = json.load(f)
    feature_to_group = {}
    for group, members in group_definitions.items():
        for feature in members:
            if feature in feature_to_group:
                raise ValueError(f"Feature appears in multiple groups: {feature}")
            feature_to_group[feature] = group
    if set(feature_to_group) != set(FEATURES):
        missing = sorted(set(FEATURES) - set(feature_to_group))
        extra = sorted(set(feature_to_group) - set(FEATURES))
        raise ValueError(f"Feature-group coverage mismatch. Missing={missing}; extra={extra}")

    saved_orders = {}
    output_rows = []
    for (context, model, run_id), part in run_rankings.groupby(["context", "model", "run"]):
        score_by_feature = part.set_index("feature")["mean_abs_shap"].to_dict()
        group_scores = {
            group: float(sum(score_by_feature[feature] for feature in members))
            for group, members in group_definitions.items()
        }
        order = sorted(group_scores, key=lambda group: group_scores[group], reverse=True)
        saved_orders[(context, model, int(run_id))] = order
        rank_lookup = {group: rank for rank, group in enumerate(order, start=1)}
        for group, score in group_scores.items():
            output_rows.append({
                "context": context,
                "model": model,
                "run": int(run_id),
                "family": group,
                "aggregated_family_importance": score,
                "rank": rank_lookup[group],
            })

    pd.DataFrame(output_rows).sort_values(
        ["context", "model", "run", "rank"]
    ).to_csv(OUT_DIR / "family_run_rankings.csv", index=False)

    matrices = {}
    n_runs = int(run_rankings["run"].nunique())
    for model in MODELS:
        ctx_orders = {
            context: [saved_orders[(context, model, run_id)] for run_id in range(1, n_runs + 1)]
            for context in CONTEXTS
        }
        for metric in FAMILY_METRICS:
            per_context_matrices = {}
            for context in CONTEXTS:
                orders = ctx_orders[context]
                mat = np.full((n_runs, n_runs), np.nan, dtype=float)
                np.fill_diagonal(mat, metric_diagonal(metric))
                for i in range(n_runs):
                    for j in range(i + 1, n_runs):
                        val = compare_orders(orders[i], orders[j], FAMILY_METRICS)[metric]
                        mat[i, j] = mat[j, i] = val
                per_context_matrices[context] = mat
            cross = np.full((n_runs, n_runs), np.nan, dtype=float)
            for i in range(n_runs):
                for j in range(n_runs):
                    cross[i, j] = compare_orders(
                        ctx_orders[CONTEXTS[0]][i], ctx_orders[CONTEXTS[1]][j], FAMILY_METRICS
                    )[metric]
            if any(np.isnan(m).any() for m in [*per_context_matrices.values(), cross]):
                raise ValueError(f"Missing family pairwise cells: {model}/{metric}")
            matrices[(model, CONTEXTS[0], metric)] = per_context_matrices[CONTEXTS[0]]
            matrices[(model, CONTEXTS[1], metric)] = per_context_matrices[CONTEXTS[1]]
            matrices[(model, "cross", metric)] = cross
    return matrices, n_runs


def run_stability_analysis(rng):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairwise = pd.read_csv(D_DIR / "pairwise_stability.csv")
    feature_matrices, n_runs = make_feature_matrices(pairwise)
    family_matrices, n_family_runs = family_rankings_and_matrices()
    if n_runs != n_family_runs:
        raise ValueError(f"Feature/family run counts differ: {n_runs} vs {n_family_runs}")
    rows = []
    for level, metric_list, matrices in [
        ("feature", FEATURE_METRICS, feature_matrices),
        ("family", FAMILY_METRICS, family_matrices),
    ]:
        for model in MODELS:
            for metric in metric_list:
                matrix_novpn = matrices[(model, CONTEXTS[0], metric)]
                matrix_vpn = matrices[(model, CONTEXTS[1], metric)]
                matrix_cross = matrices[(model, "cross", metric)]
                estimates = bootstrap_gap(
                    matrix_novpn, matrix_vpn, matrix_cross, metric, rng
                )
                rows.append({
                    "hypothesis_level": level,
                    "model": model,
                    "metric": metric,
                    "n_runs_per_context": n_runs,
                    **estimates,
                })
    out = pd.DataFrame(rows)
    out["p_bh_primary"] = np.nan
    # H1: two model-level tests on feature-level Spearman gap.
    # H2: two model-level tests on family-level Spearman gap.
    for level in ("feature", "family"):
        mask = (out["hypothesis_level"] == level) & (out["metric"] == "spearman_rho")
        out.loc[mask, "p_bh_primary"] = bh_adjust(out.loc[mask, "p_one_sided_raw"].to_numpy())
    out.to_csv(OUT_DIR / "stability_gap_bootstrap.csv", index=False)
    return out


def create_model(model_name):
    if model_name == "RandomForest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
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


def stratified_bootstrap_indices(y, rng):
    y = np.asarray(y)
    pieces = []
    for label in np.unique(y):
        positions = np.flatnonzero(y == label)
        pieces.append(rng.choice(positions, size=len(positions), replace=True))
    indices = np.concatenate(pieces)
    rng.shuffle(indices)
    return indices


def paired_prediction_swap_pvalue(y_true, pred_left, pred_right, rng, n_perm=N_PERM, batch_size=250):
    """One-sided paired randomization test for Macro-F1(left) > Macro-F1(right).

    Under the null, the two methods' predictions are exchangeable within each test
    record. Each permutation independently swaps the two predictions for a record.
    Computation is batched to keep memory use modest.
    """
    y_true = np.asarray(y_true)
    pred_left = np.asarray(pred_left)
    pred_right = np.asarray(pred_right)
    if not (len(y_true) == len(pred_left) == len(pred_right)):
        raise ValueError("y_true and both prediction vectors must have equal lengths.")

    labels = np.sort(np.unique(y_true))
    y_codes = np.searchsorted(labels, y_true)
    left_codes = np.searchsorted(labels, pred_left)
    right_codes = np.searchsorted(labels, pred_right)
    known_labels = set(labels.tolist())
    if not set(np.unique(pred_left).tolist()).issubset(known_labels) or not set(np.unique(pred_right).tolist()).issubset(known_labels):
        raise ValueError("Prediction contains a class absent from y_true.")
    n_classes = len(labels)
    support = np.bincount(y_codes, minlength=n_classes).astype(float)

    def macro_f1_codes(pred_codes):
        scores = []
        for class_id in range(n_classes):
            tp = np.count_nonzero((y_codes == class_id) & (pred_codes == class_id))
            predicted = np.count_nonzero(pred_codes == class_id)
            denom = support[class_id] + predicted
            scores.append((2.0 * tp / denom) if denom else 0.0)
        return float(np.mean(scores))

    observed = macro_f1_codes(left_codes) - macro_f1_codes(right_codes)
    extreme = 0
    n = len(y_true)
    for start in range(0, n_perm, batch_size):
        size = min(batch_size, n_perm - start)
        swaps = rng.random((size, n)) < 0.5
        perm_left = np.where(swaps, right_codes[None, :], left_codes[None, :])
        perm_right = np.where(swaps, left_codes[None, :], right_codes[None, :])
        f1_left = np.zeros((size, n_classes), dtype=float)
        f1_right = np.zeros((size, n_classes), dtype=float)
        for class_id in range(n_classes):
            is_true = y_codes == class_id
            support_c = support[class_id]
            pred_mask_left = perm_left == class_id
            pred_mask_right = perm_right == class_id
            tp_left = (pred_mask_left & is_true[None, :]).sum(axis=1)
            tp_right = (pred_mask_right & is_true[None, :]).sum(axis=1)
            count_left = pred_mask_left.sum(axis=1)
            count_right = pred_mask_right.sum(axis=1)
            denom_left = support_c + count_left
            denom_right = support_c + count_right
            f1_left[:, class_id] = np.divide(
                2.0 * tp_left, denom_left, out=np.zeros(size), where=denom_left > 0
            )
            f1_right[:, class_id] = np.divide(
                2.0 * tp_right, denom_right, out=np.zeros(size), where=denom_right > 0
            )
        deltas = f1_left.mean(axis=1) - f1_right.mean(axis=1)
        extreme += int(np.count_nonzero(deltas >= observed - 1e-12))
    return (extreme + 1) / (n_perm + 1)


def run_consequence_analysis(rng):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    permutation_rng = np.random.default_rng(SEED + 202)
    selected_df = pd.read_csv(F_DIR / "selected_top10_features.csv")
    reference_df = pd.read_csv(F_DIR / "consequence_results.csv")
    selections = {}
    for (context, model), part in selected_df.groupby(["context", "model"]):
        ordered = part.sort_values("rank")["feature"].tolist()
        if len(ordered) != 10:
            raise ValueError(f"Expected 10 selected features for {context}/{model}; got {len(ordered)}")
        if len(set(ordered)) != 10 or not set(ordered).issubset(FEATURES):
            raise ValueError(f"Invalid saved feature selection for {context}/{model}")
        selections[(context, model)] = ordered

    dataset = {}
    for context in CONTEXTS:
        folder = DATA_DIR / context
        X_train = pd.read_csv(folder / "X_train.csv")
        X_test = pd.read_csv(folder / "X_test.csv")
        y_train = pd.read_csv(folder / "y_train.csv").iloc[:, 0].to_numpy()
        y_test = pd.read_csv(folder / "y_test.csv").iloc[:, 0].to_numpy()
        if list(X_train.columns) != FEATURES or list(X_test.columns) != FEATURES:
            raise ValueError(f"Feature schema mismatch in data/{context}")
        dataset[context] = (X_train, X_test, y_train, y_test)

    predictions = {}
    perf_rows = []
    check_rows = []
    prediction_rows = []
    for target in CONTEXTS:
        source = "vpn" if target == "novpn" else "novpn"
        X_train, X_test, y_train, y_test = dataset[target]
        labels = np.sort(np.unique(y_test))
        for model_name in MODELS:
            feature_sets = {
                "full": FEATURES,
                "context_specific_top10": selections[(target, model_name)],
                "transferred_top10": selections[(source, model_name)],
            }
            for feature_set, used_features in feature_sets.items():
                model = create_model(model_name)
                model.fit(X_train[used_features], y_train)
                pred = model.predict(X_test[used_features])
                f1 = float(f1_score(y_test, pred, labels=labels, average="macro", zero_division=0))
                acc = float(accuracy_score(y_test, pred))
                predictions[(target, model_name, feature_set)] = pred
                old = reference_df[
                    (reference_df["target_context"] == target)
                    & (reference_df["model"] == model_name)
                    & (reference_df["feature_set"] == feature_set)
                ]
                if len(old) != 1:
                    raise ValueError(f"Could not uniquely match Task F result: {target}/{model_name}/{feature_set}")
                old_f1 = float(old.iloc[0]["macro_f1"])
                discrepancy = abs(f1 - old_f1)
                check_rows.append({
                    "target_context": target,
                    "model": model_name,
                    "feature_set": feature_set,
                    "saved_macro_f1": old_f1,
                    "reproduced_macro_f1": f1,
                    "absolute_difference": discrepancy,
                    "matches_within_1e-4": bool(discrepancy <= 1e-4),
                })
                perf_rows.append({
                    "target_context": target,
                    "model": model_name,
                    "feature_set": feature_set,
                    "n_features": len(used_features),
                    "accuracy": acc,
                    "macro_f1": f1,
                })
                for row_index, (truth, guess) in enumerate(zip(y_test, pred)):
                    prediction_rows.append({
                        "target_context": target,
                        "model": model_name,
                        "feature_set": feature_set,
                        "test_row_index": row_index,
                        "y_true": int(truth),
                        "y_pred": int(guess),
                    })

    check_df = pd.DataFrame(check_rows)
    check_df.to_csv(OUT_DIR / "consequence_reproduction_check.csv", index=False)
    if not check_df["matches_within_1e-4"].all():
        worst = float(check_df["absolute_difference"].max())
        raise RuntimeError(
            f"Task F reproduction failed (max Macro-F1 difference={worst:.6g}). "
            "Inspect consequence_reproduction_check.csv and package versions before interpreting the bootstrap."
        )

    pd.DataFrame(prediction_rows).to_csv(OUT_DIR / "consequence_test_predictions.csv", index=False)
    perf_df = pd.DataFrame(perf_rows)
    bootstrap_summary_rows = []
    difference_rows = []
    h3_indices = []
    for target in CONTEXTS:
        _, X_test, _, y_test = dataset[target]
        labels = np.sort(np.unique(y_test))
        for model_name in MODELS:
            conds = ["full", "context_specific_top10", "transferred_top10"]
            pred_by_cond = {cond: predictions[(target, model_name, cond)] for cond in conds}
            point_by_cond = {
                cond: float(f1_score(y_test, pred_by_cond[cond], labels=labels, average="macro", zero_division=0))
                for cond in conds
            }
            draws_f1 = {cond: np.empty(N_BOOT, dtype=float) for cond in conds}
            draw_deltas = {
                "context_specific_minus_full": np.empty(N_BOOT, dtype=float),
                "transferred_minus_full": np.empty(N_BOOT, dtype=float),
                "context_specific_minus_transferred": np.empty(N_BOOT, dtype=float),
            }
            for b in range(N_BOOT):
                idx = stratified_bootstrap_indices(y_test, rng)
                sampled_y = y_test[idx]
                for cond in conds:
                    draws_f1[cond][b] = f1_score(
                        sampled_y, pred_by_cond[cond][idx], labels=labels,
                        average="macro", zero_division=0,
                    )
                draw_deltas["context_specific_minus_full"][b] = draws_f1["context_specific_top10"][b] - draws_f1["full"][b]
                draw_deltas["transferred_minus_full"][b] = draws_f1["transferred_top10"][b] - draws_f1["full"][b]
                draw_deltas["context_specific_minus_transferred"][b] = draws_f1["context_specific_top10"][b] - draws_f1["transferred_top10"][b]

            for cond in conds:
                low, high = np.percentile(draws_f1[cond], [2.5, 97.5])
                bootstrap_summary_rows.append({
                    "target_context": target,
                    "model": model_name,
                    "feature_set": cond,
                    "macro_f1_point": point_by_cond[cond],
                    "macro_f1_ci_95_low": float(low),
                    "macro_f1_ci_95_high": float(high),
                    "bootstrap_replicates": N_BOOT,
                    "interval_scope": "test-record uncertainty conditional on fitted model and selected features",
                })

            contrast_pairs = [
                ("context_specific_minus_full", "context_specific_top10", "full"),
                ("transferred_minus_full", "transferred_top10", "full"),
                ("context_specific_minus_transferred", "context_specific_top10", "transferred_top10"),
            ]
            for contrast, left, right in contrast_pairs:
                values = draw_deltas[contrast]
                low, high = np.percentile(values, [2.5, 97.5])
                point = point_by_cond[left] - point_by_cond[right]
                is_h3 = contrast == "context_specific_minus_transferred"
                if is_h3:
                    raw_p = paired_prediction_swap_pvalue(
                        y_test,
                        pred_by_cond[left],
                        pred_by_cond[right],
                        permutation_rng,
                        n_perm=N_PERM,
                    )
                else:
                    raw_p = np.nan
                difference_rows.append({
                    "target_context": target,
                    "model": model_name,
                    "contrast": contrast,
                    "macro_f1_difference": float(point),
                    "ci_95_low": float(low),
                    "ci_95_high": float(high),
                    "p_one_sided_raw_for_H3": float(raw_p) if is_h3 else np.nan,
                    "is_H3_top10_exploratory_test": is_h3,
                    "bootstrap_replicates": N_BOOT,
                    "permutation_replicates": N_PERM if is_h3 else np.nan,
                    "p_value_method": (
                        "paired_per_record_prediction_swap_permutation" if is_h3 else "not_tested"
                    ),
                })
                if is_h3:
                    h3_indices.append(len(difference_rows) - 1)

    summary_df = pd.DataFrame(bootstrap_summary_rows)
    summary_df.to_csv(OUT_DIR / "consequence_bootstrap_summary.csv", index=False)
    differences_df = pd.DataFrame(difference_rows)
    differences_df["p_bh_H3_top10"] = np.nan
    h3_mask = differences_df["is_H3_top10_exploratory_test"]
    differences_df.loc[h3_mask, "p_bh_H3_top10"] = bh_adjust(
        differences_df.loc[h3_mask, "p_one_sided_raw_for_H3"].to_numpy()
    )
    differences_df.to_csv(OUT_DIR / "consequence_difference_tests.csv", index=False)
    return check_df, summary_df, differences_df


def markdown_table(df, columns):
    view = df.loc[:, columns].copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            view[col] = view[col].astype(str)
    lines = ["| " + " | ".join(view.columns) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(row.tolist()) + " |")
    return "\n".join(lines)


def write_report(stability_df=None, reproduction_df=None, consequence_df=None, difference_df=None):
    parts = [
        "# Task I — Statistical analysis",
        "",
        f"Bootstrap replicates: {N_BOOT}; fixed random seed: {SEED}.",
        "",
        "## Stability inference",
        "Positive `stability_gap` means the rankings are more stable within a context than across contexts. For rank displacement, the sign is reversed (`cross_minus_within`) so positive retains this interpretation.",
        "The bootstrap resamples the Non-VPN and VPN run-level ranking lists independently. It does not treat the 435/900 repeated pairwise comparisons as independent observations, and it does not assume matching seed numbers create matched experimental units.",
        "Primary H1/H2 inference uses basic bootstrap intervals and an approximate one-sided null-centred nonparametric run bootstrap. The pairwise statistic's bootstrap errors are centered at their empirical mean to mitigate finite-sample self-comparison bias. This is not a permutation test; minimum attainable Monte Carlo p-value is 1/(3000+1).",
        "",
    ]
    if stability_df is not None:
        primary = stability_df[stability_df["metric"] == "spearman_rho"]
        parts.append("### Primary Spearman stability-gap tests")
        parts.append("")
        parts.append(markdown_table(primary, ["hypothesis_level", "model", "within_mean", "cross_mean", "stability_gap", "ci_95_low", "ci_95_high", "p_one_sided_raw", "p_bh_primary"]))
        parts.append("")
        parts.append("Benjamini–Hochberg correction is applied separately to the two model-level primary bootstrap tests within feature-level H1 and family-level H2. Other metric intervals are secondary/descriptive.")
        parts.append("")
        parts.append("Family importance is the sum of member-feature mean absolute SHAP values per run; it is not a fresh grouped-feature SHAP calculation.")
        parts.append("")
    if reproduction_df is not None:
        parts.append("## Task F reproduction check")
        parts.append("")
        parts.append(markdown_table(reproduction_df, ["target_context", "model", "feature_set", "saved_macro_f1", "reproduced_macro_f1", "absolute_difference", "matches_within_1e-4"]))
        parts.append("")
    if consequence_df is not None:
        parts.append("## Macro-F1 intervals")
        parts.append("")
        parts.append("Intervals are stratified bootstrap intervals over held-out test records, conditional on the fitted models and selected features.")
        parts.append("")
        parts.append(markdown_table(consequence_df, ["target_context", "model", "feature_set", "macro_f1_point", "macro_f1_ci_95_low", "macro_f1_ci_95_high"]))
        parts.append("")
    if difference_df is not None:
        parts.append("## Consequence contrasts")
        parts.append("")
        parts.append(markdown_table(difference_df, ["target_context", "model", "contrast", "macro_f1_difference", "ci_95_low", "ci_95_high", "p_one_sided_raw_for_H3", "p_bh_H3_top10"]))
        parts.append("")
        parts.append("The four H3-style p-values use one-sided paired prediction-swap permutation tests (20,000 permutations), with Benjamini–Hochberg correction across the four comparisons. Confidence intervals use stratified paired test-record bootstrap resampling.")
        parts.append("Caution: Task F uses Top-10 feature subsets, so the four context/model tests of context-specific minus transferred Top-10 are exploratory adaptations. They do not establish the result of a planned Top-15 experiment. The bootstrap does not include training or feature-selection variability.")
        parts.append("")
    (OUT_DIR / "task_i_report.md").write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Task I statistical analysis")
    parser.add_argument("--mode", choices=["all", "stability", "consequence"], default="all")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    stability_df = reproduction_df = consequence_df = difference_df = None

    if args.mode in ("all", "stability"):
        print("[Task I] Running run-level feature/family bootstrap; no model training or SHAP recomputation.")
        stability_df = run_stability_analysis(rng)
        print("[Task I] Wrote results/statistical_analysis/stability_gap_bootstrap.csv")

    if args.mode in ("all", "consequence"):
        print("[Task I] Re-fitting the 12 Task F final models to reproduce saved scores; this may take a few minutes.")
        reproduction_df, consequence_df, difference_df = run_consequence_analysis(np.random.default_rng(SEED + 1))
        print("[Task I] Task F scores reproduced within 1e-4; consequence bootstrap outputs written.")

    # Keep the report useful when the two modes are run separately.
    stable_path = OUT_DIR / "stability_gap_bootstrap.csv"
    if stability_df is None and stable_path.exists():
        stability_df = pd.read_csv(stable_path)
    check_path = OUT_DIR / "consequence_reproduction_check.csv"
    summary_path = OUT_DIR / "consequence_bootstrap_summary.csv"
    difference_path = OUT_DIR / "consequence_difference_tests.csv"
    if reproduction_df is None and check_path.exists():
        reproduction_df = pd.read_csv(check_path)
    if consequence_df is None and summary_path.exists():
        consequence_df = pd.read_csv(summary_path)
    if difference_df is None and difference_path.exists():
        difference_df = pd.read_csv(difference_path)
    write_report(stability_df, reproduction_df, consequence_df, difference_df)

    manifest = {
        "task": "I - Statistical analysis",
        "bootstrap_replicates": N_BOOT,
        "consequence_permutation_replicates": N_PERM,
        "random_seed": SEED,
        "primary_stability_p_value_method": "approximate one-sided null-centred nonparametric run bootstrap with empirical-centred bootstrap errors; not a permutation test",
        "consequence_p_value_method": "one-sided paired per-record prediction-swap permutation test for exploratory Top-10 contrasts",
        "stability_unit": "Task D run-level rankings; independently resample the context-specific run lists; pairwise comparisons are recomputed from run-level matrix entries",
        "stability_primary_metric": "Spearman rank correlation stability gap (within-context mean minus cross-context mean)",
        "rank_displacement_gap": "cross-context mean minus within-context mean, so positive means lower within-context displacement",
        "multiple_testing": {
            "H1_feature_primary_spearman": "Benjamini-Hochberg across RandomForest and XGBoost",
            "H2_family_primary_spearman": "Benjamini-Hochberg across RandomForest and XGBoost",
            "H3_top10_exploratory": "Benjamini-Hochberg across 4 context/model contrasts",
        },
        "family_method": "sum of member-feature mean absolute SHAP importances per Task D run; ranks six aggregate families",
        "consequence_method": "re-fit 12 Task F final models with saved Top-10 selections; class-stratified paired bootstrap of untouched test rows",
        "consequence_scope": "conditional on fitted models and selected features; no training/selection uncertainty",
        "important_caveat": "Task F tested Top-10, not the manuscript's planned Top-15; H3 Top-10 inference is exploratory.",
        "outputs": [
            "stability_gap_bootstrap.csv", "family_run_rankings.csv",
            "consequence_reproduction_check.csv", "consequence_test_predictions.csv",
            "consequence_bootstrap_summary.csv", "consequence_difference_tests.csv", "task_i_report.md",
        ],
    }
    (OUT_DIR / "method_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[Task I] Report: results/statistical_analysis/task_i_report.md")


if __name__ == "__main__":
    main()
