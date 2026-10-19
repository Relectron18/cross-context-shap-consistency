"""Audit numbers for the v12 revision (in-project review issues M1, M2, M3, M5, M7, m1, m4).
Reads only existing result files and the deduplicated data; writes results/p7_audit.json.
The row-order part re-runs the p6 locality statistic on the deduplicated flows (the p6 version used all ARFF rows)
and measures how strongly row position tracks the features within each class."""
from c2 import *
from scipy.stats import spearmanr
S = json.load(open(OUT2 / "ref_summary.json"))
P4 = json.load(open(OUT2 / "p4_results.json")); P5 = json.load(open(OUT2 / "p5_results.json"))
out = {}
# M5: one-sided gaps (within one context minus cross) behind the mean gap
for m in ("RF", "XGB"):
    for d in ("random", "grouped", "calibrated", "orderedcal", "orderedhalf"):
        mu = S[f"{m}_{d}"]["mean"]
        out[f"onesided_{m}_{d}"] = {"nv": mu["rho_wnv"] - mu["rho_cross"], "vpn": mu["rho_wvpn"] - mu["rho_cross"], "mean": mu["rho_gap"]}
# M2: sign of the other gap metrics under the first/second-half split
for m in ("RF", "XGB"):
    df = pd.read_csv(OUT2 / f"ref_summary_{m}_orderedhalf.csv")
    out[f"orderedhalf_{m}"] = {k: {"min": float(df[k].min()), "n_neg": int((df[k] < 0).sum()), "n": len(df)}
                               for k in ("rho_gap", "rbo_gap", "top5_gap")}
# M3: whole-ranking cross-context agreement, SHAP vs permutation importance
out["cross_rho"] = {m: {"shap": P1["cross_" + m]["rho"] if (P1 := json.load(open(OUT2 / "p1_results.json"))) else None,
                        "pi_marginal": P4[f"pi_cross_{m}"]["rho"], "pi_conditional": P5[f"cpi_cross_{m}"]["rho"]} for m in ("RF", "XGB")}
# m1/m4: family-level gaps under random halves
out["family_gaps"] = {m: {k: S[f"{m}_random"]["mean"][k] for k in ("fam_gap", "fam_mean_gap", "fam_alt_gap", "clust_gap")} for m in ("RF", "XGB")}
# M1: row-order locality on deduplicated flows, position-feature dependence, share of unsplittable classes
pos = {c: np.asarray(v) for c, v in json.load(open(OUT2 / "arff_order_positions.json")).items()}
lab = {v: k for k, v in json.load(open(ROOT / "data" / "label_classes.json")).items()}
rng = np.random.default_rng(0); loc = {}
for c in CTX:
    X, y = pool(c)
    Z = StandardScaler().fit_transform(logx2(X[FEATURES].values))
    for k in range(7):
        ids = np.flatnonzero(y == k); ids = ids[np.argsort(pos[c][ids])]; z = Z[ids]
        near = [np.linalg.norm(z[[j for j in range(max(0, i - 20), min(len(z), i + 21)) if j != i]] - z[i], axis=1).min() for i in range(len(z))]
        ran = [np.linalg.norm(z[rng.choice(np.delete(np.arange(len(z)), i), 40, replace=False)] - z[i], axis=1).min() for i in range(len(z))]
        lag1 = np.median(np.linalg.norm(z[1:] - z[:-1], axis=1))
        pr = rng.integers(0, len(z), (20000, 2)); pr = pr[pr[:, 0] != pr[:, 1]]
        rnd = np.median(np.linalg.norm(z[pr[:, 0]] - z[pr[:, 1]], axis=1))
        rho_pos = max(abs(spearmanr(np.arange(len(z)), X[FEATURES].values[ids, j])[0] or 0) for j in range(len(FEATURES)))
        loc[f"{c}_{lab[k]}"] = {"n": len(z), "win20_dedup": float(np.median(near) / np.median(ran)),
                                "lag1_dedup": float(lag1 / rnd), "max_abs_rho_position_feature": float(rho_pos)}
    share = float(np.isin(y, [0, 4]).mean())
    out[f"share_browsing_p2p_{c}"] = share
out["locality_dedup"] = loc
# M7: distinct Top-k lists produced by the five selection seeds (p2 stored the full rankings)
RL = json.load(open(OUT2 / "p2_results.json"))["rank_lists"]
distinct = {f"{m}_{c}_k{k}": len({tuple(sorted(RL[f"{m}|{c}|{s}"][:k])) for s in range(5)})
            for m in ("RF", "XGB") for c in CTX for k in (5, 10, 15)}
json.dump(distinct, open(OUT2 / "p7_distinct_lists.json", "w"), indent=1)
out["distinct_lists_range"] = [min(distinct.values()), max(distinct.values())]
jdump(out, "p7_audit.json")
print(json.dumps(out, indent=1, default=float)[:6000])
