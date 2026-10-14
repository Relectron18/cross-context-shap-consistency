"""Audits, baselines and primary explanations, all in the revision-2 environment."""
from c2 import *
from scipy.stats import chi2_contingency, entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
import sklearn, xgboost, shap, platform
R = {"env": dict(python=platform.python_version(), sklearn=sklearn.__version__, xgboost=xgboost.__version__,
                 shap=shap.__version__, numpy=np.__version__, pandas=pd.__version__, os=platform.platform())}
D = {c: load(c) for c in CTX}
CLS = ["BROWSING", "CHAT", "FT", "MAIL", "P2P", "STREAMING", "VOIP"]
# ---- data audit
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    df = Xtr.copy(); df["y"] = ytr
    g = df.groupby(FEATURES)["y"].agg(["size", "nunique"])
    dup = g[g["size"] > 1]
    below = (Xtr < -1)
    R[f"audit_{c}"] = dict(
        n_train=len(ytr), n_test=len(yte), counts_train=np.bincount(ytr).tolist(), counts_test=np.bincount(yte).tolist(),
        dup_vectors_train=int(len(dup)), dup_rows_train=int(dup["size"].sum()), dup_label_conflict=int((dup["nunique"] > 1).sum()),
        unique_train=int(len(g)), unique_test=int(len(Xte.drop_duplicates())),
        below_m1_entries=int(below.values.sum()), below_m1_rows=int(below.any(axis=1).sum()),
        below_m1_by_feature={f: int(v) for f, v in below.sum().items() if v > 0},
        below_m1_by_class=np.bincount(ytr[below.any(axis=1).values], minlength=7).tolist(),
        below_m1_min=float(Xtr.values.min()),
        collisions=int(len(set(map(tuple, Xtr.round(9).values)) & set(map(tuple, Xte.round(9).values)))),
        sentinel_rate={f: float(((pd.concat([Xtr, Xte]))[f] == -1).mean()) for f in FEATURES if ((Xtr[f] == -1).any())},
    )
    print(c, {k: v for k, v in R[f"audit_{c}"].items() if k != "sentinel_rate"}, flush=True)
a = np.array([np.bincount(D[c][2]) for c in CTX]); p = a / a.sum(1, keepdims=True)
chi = chi2_contingency(a)
R["classmix"] = dict(tvd=float(0.5 * np.abs(p[0] - p[1]).sum()), chi2=float(chi[0]), dof=int(chi[2]), p=float(chi[1]),
                     cramer_v=float(np.sqrt(chi[0] / a.sum())), imb_ratio=[float(x.max() / x.min()) for x in a])
# ---- near-duplicate audit (log(x+2) and rank)
def nn_med(A, B, same):
    d, _ = NearestNeighbors(n_neighbors=2 if same else 1).fit(B).kneighbors(A)
    return float(np.median(d[:, 1] if same else d[:, 0]))
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    sc = StandardScaler().fit(Xtr); A0, B0 = sc.transform(Xtr), sc.transform(Xte)
    sc = StandardScaler().fit(logx2(Xtr)); A1, B1 = sc.transform(logx2(Xtr)), sc.transform(logx2(Xte))
    qt = QuantileTransformer(n_quantiles=1000, random_state=0).fit(Xtr); A2, B2 = qt.transform(Xtr), qt.transform(Xte)
    R[f"nn_{c}"] = {lab: dict(test_to_train=nn_med(B, A, False), train_to_train=nn_med(A, A, True))
                    for lab, A, B in (("std", A0, B0), ("log", A1, B1), ("rank", A2, B2))}
    print(c, R[f"nn_{c}"], flush=True)
# ---- cluster-grouped CV proxy (XGB and RF)
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    Z = StandardScaler().fit_transform(logx2(Xtr)); g = KMeans(n_clusters=50, n_init=3, random_state=0).fit_predict(Z)
    for mname in ("XGB", "RF"):
        for lab, spl in (("strat", StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr)), ("grouped", GroupKFold(5).split(Xtr, ytr, g))):
            f = []
            for a_, b_ in spl:
                m = make_model(mname).fit(Xtr.iloc[a_], ytr[a_]); f.append(f1_score(ytr[b_], m.predict(Xtr.iloc[b_]), average="macro"))
            R[f"cv_{mname}_{c}_{lab}"] = float(np.mean(f))
    print(c, {k: v for k, v in R.items() if k.startswith("cv_") and c in k}, flush=True)
jdump(R, "p1_results.json")
# ---- baselines + primary SHAP
IMP = {}
for mname in ("RF", "XGB"):
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        m = make_model(mname).fit(Xtr, ytr); pr = m.predict(Xte)
        R[f"base_{mname}_{c}"] = dict(acc=accuracy_score(yte, pr), prec=precision_score(yte, pr, average="macro"),
                                     rec=recall_score(yte, pr, average="macro"), f1=f1_score(yte, pr, average="macro"),
                                     recall_per_class=recall_score(yte, pr, average=None).tolist())
        t = tic(); sv = shap_values(m, Xte); rec = np.abs(sv).mean(axis=0)
        np.save(OUT2 / f"primary_rec_{mname}_{c}.npy", rec.astype(np.float32))
        np.save(OUT2 / f"primary_sv_{mname}_{c}.npy", sv.astype(np.float32))
        IMP[(mname, c)] = pd.Series(rec.mean(0), index=FEATURES)
        IMP[(mname, c, "w")] = importance(sv, y=yte)
        s = IMP[(mname, c)] / IMP[(mname, c)].sum()
        R[f"prim_{mname}_{c}"] = dict(share=s.sort_values(ascending=False).round(6).to_dict(),
                                      top5_share=float(s.sort_values(ascending=False)[:5].sum()),
                                      top10_share=float(s.sort_values(ascending=False)[:10].sum()),
                                      entropy=float(entropy(s)), family_share=(family_imp(s) * 100).round(3).to_dict())
        # sentinel records' share of min_fiat attribution
        R[f"prim_{mname}_{c}"]["minfiat_sentinel_attr_share"] = float(rec[(Xte["min_fiat"] == -1).values, FEATURES.index("min_fiat")].sum() / rec[:, FEATURES.index("min_fiat")].sum())
        print(mname, c, R[f"base_{mname}_{c}"]["f1"], f"{tic()-t:.0f}s", flush=True)
    a_, b_ = IMP[(mname, "novpn")], IMP[(mname, "vpn")]
    d = cmp_full(a_, b_); d["disp"] = float((ranks(a_) - ranks(b_)).abs().mean())
    R[f"cross_{mname}"] = d
    R[f"cross_{mname}_weighted"] = cmp_full(IMP[(mname, "novpn", "w")], IMP[(mname, "vpn", "w")])
    R[f"w_vs_unw_{mname}"] = [compare(IMP[(mname, c)], IMP[(mname, c, "w")])["rho"] for c in CTX]
    R[f"ranks_{mname}"] = {c: ranks(IMP[(mname, c)]).astype(int).to_dict() for c in CTX}
    print(mname, d, flush=True)
    jdump(R, "p1_results.json")
R["cross_model"] = {c: compare(IMP[("RF", c)], IMP[("XGB", c)])["rho"] for c in CTX}
# RF explained on 800-record samples vs full test (justifies the subsample in the repeated designs)
for c in CTX:
    rec = np.load(OUT2 / f"primary_rec_RF_{c}.npy"); yte = D[c][3]
    R[f"rf800_vs_full_{c}"] = [compare(pd.Series(rec[strat_sub(np.arange(len(yte)), yte, 800, s)].mean(0), index=FEATURES), IMP[("RF", c)])["rho"] for s in range(5)]
# LR with log(x+2) (mentioned only in one sentence)
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    sc = StandardScaler().fit(logx2(Xtr)); lr = LogisticRegression(max_iter=2000, random_state=42).fit(sc.transform(logx2(Xtr)), ytr)
    pr = lr.predict(sc.transform(logx2(Xte))); R[f"LRlog_{c}"] = dict(f1=f1_score(yte, pr, average="macro"), acc=accuracy_score(yte, pr))
# correlations among features (training partitions)
for c in CTX:
    cr = D[c][0].corr(method="spearman")
    R[f"corr_{c}"] = {f"{a}|{b}": float(cr.loc[a, b]) for a, b in [("min_fiat", "min_biat"), ("min_fiat", "min_flowiat"), ("min_biat", "min_flowiat"),
                      ("duration", "total_fiat"), ("duration", "total_biat"), ("total_fiat", "total_biat"), ("flowBytesPerSecond", "flowPktsPerSecond")]}
    X = D[c][0]; mm = np.minimum(X.min_fiat.where(X.min_fiat >= 0, np.inf), X.min_biat.where(X.min_biat >= 0, np.inf))
    R[f"minflow_eq_{c}"] = float((X.min_flowiat == mm).mean())
jdump(R, "p1_results.json"); print("done")
