"""
Task C - SHAP analysis for member2-shap branch.
Loads the 4 pre-trained models (Non-VPN RF, Non-VPN XGB, VPN RF, VPN XGB) saved by
member1 and explains each on its OWN held-out test set. No retraining, no changes
to baseline models or dataset.
"""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

FEATURES = ["duration","total_fiat","total_biat","min_fiat","min_biat","max_fiat","max_biat",
            "mean_fiat","mean_biat","flowPktsPerSecond","flowBytesPerSecond","min_flowiat",
            "max_flowiat","mean_flowiat","std_flowiat","min_active","mean_active","max_active",
            "std_active","min_idle","mean_idle","max_idle","std_idle"]

with open("data/label_classes.json") as f:
    label_classes = json.load(f)
K = len(label_classes)

# Confirmed model <-> context mapping (verified by test-set accuracy against each context's test set)
MODEL_MAP = {
    "novpn": {
        "RandomForest": "models/models_CN_project/random_forest.joblib",
        "XGBoost": "models/models_CN_project/xgboost.joblib",
    },
    "vpn": {
        "RandomForest": "models/models_CN_project/random_forest (1).joblib",
        "XGBoost": "models/models_CN_project/xgboost (1).joblib",
    },
}

shap_rankings = {}   # (ctx, model_name) -> pd.Series mean_abs_shap indexed by feature
shap_shares = {}

for ctx in ["novpn", "vpn"]:
    Xte = pd.read_csv(f"data/{ctx}/X_test.csv")[FEATURES]
    yte = pd.read_csv(f"data/{ctx}/y_test.csv")["label"].values
    print(f"\n=== Context: {ctx} (n_test={len(Xte)}) ===")

    for model_name, path in MODEL_MAP[ctx].items():
        model = joblib.load(path)
        acc = (model.predict(Xte) == yte).mean()
        print(f"{model_name}: loaded from {path}, test accuracy = {acc:.4f}")

        # TreeSHAP - tree_path_dependent perturbation (explicit choice, per audit requirement A7).
        # This variant uses the tree's own training-data path statistics as the background
        # and is exact + tractable at this scale; interventional TreeSHAP with an explicit
        # background sample was tested but is computationally prohibitive for RF (200 trees)
        # at this test-set size within a reasonable runtime.
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        sv = explainer.shap_values(Xte, check_additivity=False)

        sv = np.array(sv)
        # normalize shape to (K, N, F)
        if sv.ndim == 3:
            if sv.shape[0] == Xte.shape[0] and sv.shape[-1] == K:
                sv = np.transpose(sv, (2, 0, 1))  # (N,F,K) -> (K,N,F)
            # else assume already (K,N,F)
        elif sv.ndim == 2:
            # binary / single-output fallback, shouldn't happen here (K=7)
            sv = sv[None, :, :]

        mean_abs = np.mean(np.abs(sv), axis=(0, 1))  # class-macro mean |SHAP| per feature
        imp = pd.Series(mean_abs, index=FEATURES).sort_values(ascending=False)
        shap_rankings[(ctx, model_name)] = imp
        shap_shares[(ctx, model_name)] = imp / imp.sum()

        # Save raw SHAP values (compressed) and rankings
        np.save(f"SHAP/results/shap_values_{ctx}_{model_name}.npy", sv)
        imp.to_csv(f"SHAP/results/shap_ranking_{ctx}_{model_name}.csv", header=["mean_abs_shap"])
        shap_shares[(ctx, model_name)].to_csv(f"SHAP/results/shap_share_{ctx}_{model_name}.csv", header=["share_of_total_importance"])

        # Plot: top-15 bar chart
        top15 = imp.head(15)
        plt.figure(figsize=(7, 5))
        plt.barh(top15.index[::-1], top15.values[::-1], color="#4C72B0")
        plt.xlabel("Mean |SHAP value| (class-macro)")
        plt.title(f"Top-15 SHAP importance: {model_name}, {ctx}")
        plt.tight_layout()
        plt.savefig(f"SHAP/plots/shap_bar_{ctx}_{model_name}.png", dpi=150)
        plt.close()

        print(f"  Top-5: {list(imp.head(5).index)}")

# ---- Cross-context comparison for each model family ----
comparison_results = {}
for model_name in ["RandomForest", "XGBoost"]:
    non = shap_rankings[("novpn", model_name)]
    vpn = shap_rankings[("vpn", model_name)]
    rank_non = non.rank(ascending=False, method="first")
    rank_vpn = vpn.rank(ascending=False, method="first")

    rho, p_rho = spearmanr(rank_non[FEATURES], rank_vpn[FEATURES])
    tau, p_tau = kendalltau(rank_non[FEATURES], rank_vpn[FEATURES])

    def topk(k):
        a = set(non.sort_values(ascending=False).head(k).index)
        b = set(vpn.sort_values(ascending=False).head(k).index)
        return len(a & b)

    result = {
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "kendall_tau": float(tau), "kendall_p": float(p_tau),
        "top5_overlap": f"{topk(5)}/5",
        "top10_overlap": f"{topk(10)}/10",
    }
    comparison_results[model_name] = result
    print(f"\n=== Cross-context comparison: {model_name} ===")
    print(result)

    # Save standardized side-by-side ranking table
    merged = pd.DataFrame({
        "non_vpn_rank": rank_non.astype(int),
        "non_vpn_mean_abs_shap": non,
        "non_vpn_share": shap_shares[("novpn", model_name)],
        "vpn_rank": rank_vpn.astype(int),
        "vpn_mean_abs_shap": vpn,
        "vpn_share": shap_shares[("vpn", model_name)],
    })
    merged["abs_rank_shift"] = (merged["non_vpn_rank"] - merged["vpn_rank"]).abs()
    merged = merged.sort_values("non_vpn_rank")
    merged.to_csv(f"SHAP/results/standardized_ranking_comparison_{model_name}.csv")

with open("SHAP/results/cross_context_comparison_summary.json", "w") as f:
    json.dump(comparison_results, f, indent=2)

print("\nAll SHAP outputs written under SHAP/results/ and SHAP/plots/")
