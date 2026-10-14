"""Sentinel encodings, model/TreeSHAP robustness, XGB seeds and depth (single environment)."""
import shap
from c2 import *
R = {}
try: R = json.load(open(OUT2 / "p3_results.json"))
except FileNotFoundError: pass
D = {c: load(c) for c in CTX}
IND = ["min_fiat", "min_biat", "max_fiat", "max_biat", "min_flowiat", "max_flowiat", "min_active", "max_active", "min_idle", "max_idle"]
def enc(X, s):
    Z = X.copy()
    if s == "S2": Z = Z.mask(Z == -1, np.nan)
    for f in IND: Z[f"is_undef_{f}"] = (X[f] == -1).astype(int)
    return Z
SIMP = {}
for mname in (("XGB", "RF") if "sent_RF_cross_S2" not in R else ()):
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        for s in ("S1", "S2"):
            A, B = enc(Xtr, s), enc(Xte, s)
            m = make_model(mname).fit(A, ytr)
            sv = shap_values(m, B); imp = pd.Series(np.abs(sv).mean(axis=(0, 1)), index=A.columns)
            SIMP[(mname, c, s)] = imp
            R[f"sent_{mname}_{c}_{s}"] = dict(f1=f1_score(yte, m.predict(B), average="macro"),
                                             ind_share=float(imp[[f"is_undef_{f}" for f in IND]].sum() / imp.sum() * 100))
        a, b = SIMP[(mname, c, "S1")][FEATURES], SIMP[(mname, c, "S2")][FEATURES]
        R[f"sent_{mname}_{c}_S1vsS2"] = compare(a, b)
        print(mname, c, R[f"sent_{mname}_{c}_S1"], R[f"sent_{mname}_{c}_S2"], R[f"sent_{mname}_{c}_S1vsS2"], flush=True)
    for s in ("S1", "S2"):
        R[f"sent_{mname}_cross_{s}"] = compare(SIMP[(mname, "novpn", s)][FEATURES], SIMP[(mname, "vpn", s)][FEATURES])
        print(mname, s, "cross", R[f"sent_{mname}_cross_{s}"], flush=True)
    jdump(R, "p3_results.json")
# ---- robustness: RF/XGB/ET x path/interventional on 500 records
REC = {}
for mname in ("RF", "XGB", "ET"):
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        m = make_model(mname).fit(Xtr, ytr)
        ex_ids = strat_sub(np.arange(len(yte)), yte, 500, 42); bg_ids = strat_sub(np.arange(len(ytr)), ytr, 200, 42)
        R[f"rob_perf_{mname}_{c}"] = dict(f1=f1_score(yte, m.predict(Xte), average="macro"), acc=accuracy_score(yte, m.predict(Xte)))
        for meth in ("path", "interv"):
            t = tic()
            if meth == "path":
                sv = shap_values(m, Xte.iloc[ex_ids])
            else:
                ex = shap.TreeExplainer(m, data=shap.maskers.Independent(Xtr.iloc[bg_ids], max_samples=200), feature_perturbation="interventional", model_output="raw")
                sv = np.asarray(ex.shap_values(Xte.iloc[ex_ids], check_additivity=False))
                if sv.ndim == 3 and sv.shape[-1] == 7: sv = np.transpose(sv, (2, 0, 1))
            REC[(mname, c, meth)] = np.abs(sv).mean(axis=0)
            print("rob", mname, c, meth, f"{tic()-t:.0f}s", flush=True)
    for meth in ("path", "interv"):
        a, b = REC[(mname, "novpn", meth)], REC[(mname, "vpn", meth)]
        ia, ib = pd.Series(a.mean(0), index=FEATURES), pd.Series(b.mean(0), index=FEATURES)
        d = cmp_full(ia, ib)
        rng = np.random.default_rng(20261008); bs = []
        for _ in range(2000):
            bs.append(compare(pd.Series(a[rng.integers(0, len(a), len(a))].mean(0), index=FEATURES),
                              pd.Series(b[rng.integers(0, len(b), len(b))].mean(0), index=FEATURES))["rho"])
        bs = np.array(bs); d["ci_pct"] = np.percentile(bs, [2.5, 97.5]).tolist()
        d["ci_basic"] = [2 * d["rho"] - np.percentile(bs, 97.5), 2 * d["rho"] - np.percentile(bs, 2.5)]
        d["boot_mean"] = float(bs.mean())
        R[f"rob_{mname}_{meth}"] = d
        print(mname, meth, d, flush=True)
    for c in CTX:
        R[f"rob_{mname}_{c}_path_vs_interv"] = compare(pd.Series(REC[(mname, c, "path")].mean(0), index=FEATURES), pd.Series(REC[(mname, c, "interv")].mean(0), index=FEATURES))["rho"]
    jdump(R, "p3_results.json")
for c in CTX:
    for meth in ("path", "interv"):
        R[f"rob_ET_vs_RF_{c}_{meth}"] = compare(pd.Series(REC[("ET", c, meth)].mean(0), index=FEATURES), pd.Series(REC[("RF", c, "path")].mean(0), index=FEATURES))["rho"]
# ---- XGB seeds and depth settings
import itertools
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    imps = [importance(shap_values(make_model("XGB", seed=s).fit(Xtr, ytr), Xte)) for s in (1, 2, 3)]
    R[f"seed_XGB_{c}"] = float(np.mean([compare(a, b)["rho"] for a, b in itertools.combinations(imps, 2)]))
for lab, kw in (("depth3", dict(max_depth=3)), ("depth10", dict(max_depth=10))):
    imps, f1 = {}, {}
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        m = make_model("XGB", **kw).fit(Xtr, ytr); f1[c] = f1_score(yte, m.predict(Xte), average="macro"); imps[c] = importance(shap_values(m, Xte))
    R[f"hp_XGB_{lab}"] = dict(f1=f1, cross=cmp_full(imps["novpn"], imps["vpn"]))
    print(lab, R[f"hp_XGB_{lab}"], flush=True)
jdump(R, "p3_results.json"); print("done")
