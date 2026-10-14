from c2 import *
import itertools
R = json.load(open(OUT2 / "p3_results.json"))
E = {c: np.load(OUT2 / f"et_rec_{c}.npz") for c in CTX}
for c in CTX: R[f"rob_perf_ET_{c}"] = json.load(open(OUT2 / f"et_perf_{c}.json"))
S_ = lambda a: pd.Series(a.mean(0), index=FEATURES)
for meth in ("path", "interv"):
    a, b = E["novpn"][f"ET_{meth}"], E["vpn"][f"ET_{meth}"]
    d = cmp_full(S_(a), S_(b)); rng = np.random.default_rng(20261008); bs = []
    for _ in range(2000):
        bs.append(compare(S_(a[rng.integers(0, len(a), len(a))]), S_(b[rng.integers(0, len(b), len(b))]))["rho"])
    bs = np.array(bs); d["ci_pct"] = np.percentile(bs, [2.5, 97.5]).tolist()
    d["ci_basic"] = [2 * d["rho"] - np.percentile(bs, 97.5), 2 * d["rho"] - np.percentile(bs, 2.5)]; d["boot_mean"] = float(bs.mean())
    R[f"rob_ET_{meth}"] = d; print("ET", meth, d)
for c in CTX:
    R[f"rob_ET_{c}_path_vs_interv"] = compare(S_(E[c]["ET_path"]), S_(E[c]["ET_interv"]))["rho"]
    for meth in ("path", "interv"):
        R[f"rob_ET_vs_RF_{c}_{meth}"] = compare(S_(E[c][f"ET_{meth}"]), S_(E[c]["RF_path"]))["rho"]
D = {c: load(c) for c in CTX}
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
