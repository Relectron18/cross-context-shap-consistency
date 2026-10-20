"""v14 leftover checks from the v13 verification. Writes results/p10_leftover_checks.json.
1. Random Forest S1/S2 sentinel fits: ties among the 23 original-feature importances (relevant to ranks() tie
   handling), and reproduction of the stored S1-vs-S2 and cross-context values in p3_results.json.
2. XGBoost tree_method: library default ('auto') against 'hist' on both contexts.
"""
from c2 import *
P3 = json.load(open(OUT2 / "p3_results.json"))
D = {c: load(c) for c in CTX}
IND = ["min_fiat", "min_biat", "max_fiat", "max_biat", "min_flowiat", "max_flowiat", "min_active", "max_active", "min_idle", "max_idle"]
def enc(X, s):                       # copied from p3_sentinel_robust.py (that script has no import guard)
    Z = X.copy()
    if s == "S2": Z = Z.mask(Z == -1, np.nan)
    for f in IND: Z[f"is_undef_{f}"] = (X[f] == -1).astype(int)
    return Z
OUT, SIMP = {}, {}
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    for s in ("S1", "S2"):
        A, B = enc(Xtr, s), enc(Xte, s)
        m = make_model("RF").fit(A, ytr)
        imp = pd.Series(np.abs(shap_values(m, B)).mean(axis=(0, 1)), index=A.columns)
        SIMP[(c, s)] = imp
        v = imp[FEATURES].to_numpy()
        OUT[f"RF_{c}_{s}_ties"] = int(len(v) - len(np.unique(np.round(v, 15))))
        OUT[f"RF_{c}_{s}_f1_diff_vs_stored"] = float(f1_score(yte, m.predict(B), average="macro") - P3[f"sent_RF_{c}_{s}"]["f1"])
    OUT[f"RF_{c}_S1vsS2_rho"] = (float(compare(SIMP[(c, "S1")][FEATURES], SIMP[(c, "S2")][FEATURES])["rho"]), P3[f"sent_RF_{c}_S1vsS2"]["rho"])
for s in ("S1", "S2"):
    OUT[f"RF_cross_{s}_rho"] = (float(compare(SIMP[("novpn", s)][FEATURES], SIMP[("vpn", s)][FEATURES])["rho"]), P3[f"sent_RF_cross_{s}"]["rho"])
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    mh = make_model("XGB"); mh.set_params(tree_method="hist")
    OUT[f"XGB_{c}_auto_vs_hist_max_abs_prob_diff"] = float(np.abs(make_model("XGB").fit(Xtr, ytr).predict_proba(Xte) - mh.fit(Xtr, ytr).predict_proba(Xte)).max())
jdump(OUT, "p10_leftover_checks.json"); print(json.dumps(OUT, indent=1))
