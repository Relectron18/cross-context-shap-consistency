"""Permutation validation using saved Task D/Task I rankings only."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
STABILITY_DIR = ROOT / "results" / "stability"
STAT_DIR = ROOT / "results" / "statistical_analysis"
CONTEXTS = ("novpn", "vpn")
MODELS = ("RandomForest", "XGBoost")
N_RUNS = 30
N_PERM = 10_000
SEED = 20261010


def bh_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    return restored


def load_rankings(path, item_column):
    df = pd.read_csv(path)
    required = {"context", "model", "run", item_column, "rank"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} lacks columns: {sorted(required - set(df.columns))}")

    output = {}
    for model in MODELS:
        part_model = df[df["model"] == model]
        if part_model.empty:
            raise ValueError(f"Model {model} is missing from {path}")

        item_sets = {
            context: sorted(
                part_model.loc[part_model["context"] == context, item_column]
                .dropna().unique().tolist()
            )
            for context in CONTEXTS
        }
        if not item_sets[CONTEXTS[0]] or item_sets[CONTEXTS[0]] != item_sets[CONTEXTS[1]]:
            raise ValueError(f"Different or empty item sets for {model} in {path}")

        items = item_sets[CONTEXTS[0]]
        vectors = []
        for context in CONTEXTS:
            context_data = part_model[part_model["context"] == context]
            run_ids = sorted(context_data["run"].astype(int).unique().tolist())
            if run_ids != list(range(1, N_RUNS + 1)):
                raise ValueError(f"Unexpected run IDs for {context}/{model}: {run_ids}")

            for run_id in range(1, N_RUNS + 1):
                run = context_data[context_data["run"].astype(int) == run_id]
                if len(run) != len(items) or run[item_column].duplicated().any():
                    raise ValueError(f"Invalid rows for {context}/{model}/run {run_id}")
                ranks = run.set_index(item_column)["rank"].reindex(items).to_numpy(float)
                if not np.isfinite(ranks).all():
                    raise ValueError(f"Missing ranks for {context}/{model}/run {run_id}")
                vectors.append(ranks)

        output[model] = vectors

    return output


def make_similarity_matrix(vectors):
    n = len(vectors)
    matrix = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(spearmanr(vectors[i], vectors[j]).statistic)
            if not np.isfinite(value):
                raise ValueError(f"Invalid Spearman correlation for runs {i} and {j}")
            matrix[i, j] = matrix[j, i] = value
    return matrix


def gap_statistics(matrix, group_a):
    group_a = np.asarray(group_a, dtype=int)
    all_ids = np.arange(2 * N_RUNS)
    group_b = np.setdiff1d(all_ids, group_a, assume_unique=True)
    if len(group_a) != N_RUNS or len(group_b) != N_RUNS:
        raise ValueError("Each group must contain 30 distinct runs.")

    tri = np.triu_indices(N_RUNS, k=1)
    within_a = float(matrix[np.ix_(group_a, group_a)][tri].mean())
    within_b = float(matrix[np.ix_(group_b, group_b)][tri].mean())
    cross = float(matrix[np.ix_(group_a, group_b)].mean())
    return {
        "within_a": within_a,
        "within_b": within_b,
        "within_mean": (within_a + within_b) / 2,
        "cross_mean": cross,
        "gap": (within_a + within_b) / 2 - cross,
    }


def main():
    feature_path = STABILITY_DIR / "run_rankings.csv"
    family_path = STAT_DIR / "family_run_rankings.csv"
    reference_path = STAT_DIR / "stability_gap_bootstrap.csv"

    for path in (feature_path, family_path, reference_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    rankings = {
        "feature": load_rankings(feature_path, "feature"),
        "family": load_rankings(family_path, "family"),
    }
    reference = pd.read_csv(reference_path)
    rng = np.random.default_rng(SEED)

    # Permute the context labels across the 60 saved run-level rankings.
    partitions = [
        rng.choice(2 * N_RUNS, size=N_RUNS, replace=False)
        for _ in range(N_PERM)
    ]

    rows = []
    for level in ("feature", "family"):
        for model in MODELS:
            matrix = make_similarity_matrix(rankings[level][model])
            observed = gap_statistics(matrix, np.arange(N_RUNS))

            saved = reference[
                (reference["hypothesis_level"] == level)
                & (reference["model"] == model)
                & (reference["metric"] == "spearman_rho")
            ]
            if len(saved) != 1:
                raise ValueError(f"Could not uniquely match saved result: {level}/{model}")

            saved_gap = float(saved.iloc[0]["stability_gap"])
            if not np.isclose(observed["gap"], saved_gap, atol=1e-6, rtol=0):
                raise ValueError(
                    f"Gap mismatch for {level}/{model}: reconstructed "
                    f"{observed['gap']:.8f}, saved {saved_gap:.8f}. "
                    "No permutation results were accepted."
                )

            null_gaps = np.empty(N_PERM)
            for i, group_a in enumerate(partitions):
                null_gaps[i] = gap_statistics(matrix, group_a)["gap"]

            exceed = int(np.count_nonzero(null_gaps >= observed["gap"] - 1e-12))
            p_value = (exceed + 1) / (N_PERM + 1)

            rows.append({
                "hypothesis_level": level,
                "model": model,
                "n_runs_per_context": N_RUNS,
                "within_novpn_mean_spearman": observed["within_a"],
                "within_vpn_mean_spearman": observed["within_b"],
                "within_mean_spearman": observed["within_mean"],
                "cross_mean_spearman": observed["cross_mean"],
                "observed_gap": observed["gap"],
                "permutation_replicates": N_PERM,
                "permutation_exceedances": exceed,
                "p_permutation_one_sided": p_value,
                "null_gap_2_5_percentile": float(np.percentile(null_gaps, 2.5)),
                "null_gap_median": float(np.median(null_gaps)),
                "null_gap_97_5_percentile": float(np.percentile(null_gaps, 97.5)),
                "seed": SEED,
                "inference_scope": "Conditional on the saved bootstrap runs",
            })

    result = pd.DataFrame(rows)
    result["p_bh_within_level"] = np.nan
    for level in ("feature", "family"):
        mask = result["hypothesis_level"] == level
        result.loc[mask, "p_bh_within_level"] = bh_adjust(
            result.loc[mask, "p_permutation_one_sided"].to_numpy()
        )

    output_path = STAT_DIR / "stability_permutation_validation.csv"
    result.to_csv(output_path, index=False)
    print(result[[
        "hypothesis_level", "model", "observed_gap",
        "p_permutation_one_sided", "p_bh_within_level"
    ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved: {output_path}")
    print("No models were trained and no SHAP values were recomputed.")


if __name__ == "__main__":
    main()
