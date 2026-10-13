import pickle, json, itertools
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold
from common import *
R = {}
data = {c: load(c) for c in CTX}
def save(): json.dump(R, open(OUT / "fast_results.json", "w"), indent=1, default=float)
# --- LR: convergence and log transform
for ctx in CTX:
    Xtr, Xte, ytr, yte = data[ctx]
    for lab, f in (("std", lambda X: X), ("log", lambda X: np.log1p(X.clip(lower=-1) + 1))):
        sc = StandardScaler().fit(f(Xtr)); lr = LogisticRegression(max_iter=2000, random_state=42).fit(sc.transform(f(Xtr)), ytr)
        p = lr.predict(sc.transform(f(Xte)))
        R[f"LR_{ctx}_{lab}"] = dict(f1=f1_score(yte, p, average="macro"), acc=float((p == yte).mean()), n_iter=int(np.max(lr.n_iter_)))
    print(ctx, R[f"LR_{ctx}_std"], R[f"LR_{ctx}_log"], flush=True)
save()
# --- near-duplicate audit under log and rank transforms
def nn_med(A, B, same):
    d, _ = NearestNeighbors(n_neighbors=2 if same else 1).fit(B).kneighbors(A)
    return float(np.median(d[:, 1] if same else d[:, 0]))
for ctx in CTX:
    Xtr, Xte, ytr, yte = data[ctx]
    for lab in ("log", "rank"):
        if lab == "log":
            sc = StandardScaler().fit(np.log1p(Xtr.clip(lower=-1) + 1)); A = sc.transform(np.log1p(Xtr.clip(lower=-1) + 1)); B = sc.transform(np.log1p(Xte.clip(lower=-1) + 1))
        else:
            qt = QuantileTransformer(n_quantiles=1000, random_state=0).fit(Xtr); A = qt.transform(Xtr); B = qt.transform(Xte)
        R[f"nn_{ctx}_{lab}"] = dict(train_to_test=nn_med(A, B, False), test_to_train=nn_med(B, A, False),
                                    train_to_train=nn_med(A, A, True), test_to_test=nn_med(B, B, True))
    print(ctx, R[f"nn_{ctx}_log"], R[f"nn_{ctx}_rank"], flush=True)
save()
# --- class-matched: provided controlled subsets
for m in ("XGB", "RF"):
    cm = {}
    for ctx in CTX:
        Xtr, Xte, ytr, yte = load(ctx, controlled=True)
        mdl = make_model(m).fit(Xtr, ytr)
        p = mdl.predict(Xte)
        cm[ctx] = dict(imp=importance(shap_values(mdl, Xte)), f1=f1_score(yte, p, average="macro"))
    R[f"matched_{m}"] = dict(f1={c: cm[c]["f1"] for c in CTX}, cross=compare(cm["novpn"]["imp"], cm["vpn"]["imp"]),
                            top5={c: list(cm[c]["imp"].sort_values(ascending=False).index[:5]) for c in CTX})
    print(m, R[f"matched_{m}"], flush=True)
    save()
# --- repeated matched vs natural draws (XGB; same n = 1330)
rows = []
for r in range(6):
    imps = {}
    for ci, ctx in enumerate(CTX):
        Xtr, Xte, ytr, yte = data[ctx]
        rng = np.random.default_rng(800 + 10 * r + ci)
        mi = np.concatenate([rng.choice(np.flatnonzero(ytr == c), 190, replace=False) for c in range(7)])
        props = np.bincount(ytr) / len(ytr); cnt = np.round(props * 1330).astype(int); cnt[np.argmax(cnt)] += 1330 - cnt.sum()
        ni = np.concatenate([rng.choice(np.flatnonzero(ytr == c), cnt[c], replace=False) for c in range(7)])
        for lab, ids, w in (("matched", mi, yte), ("natural", ni, None)):
            mdl = make_model("XGB", seed=42 + r, n_jobs=1).fit(Xtr.iloc[ids], ytr[ids])
            imps[(ctx, lab)] = importance(shap_values(mdl, Xte), y=w)
    for lab in ("matched", "natural"):
        rows.append(dict(r=r, cond=lab, **compare(imps[("novpn", lab)], imps[("vpn", lab)])))
df = pd.DataFrame(rows); df.to_csv(OUT / "matched_vs_natural.csv", index=False)
R["matched_vs_natural_XGB"] = df.groupby("cond")[["rho", "tau", "top5", "top10", "rbo", "l1"]].agg(["mean", "std"]).round(4).to_string()
print(df.groupby("cond")[["rho", "top5", "rbo", "l1"]].agg(["mean", "std", "min", "max"]).round(4), flush=True)
save()
# --- seed variance (XGB, full data) and hyperparameter sensitivity (XGB)
for ctx in CTX:
    Xtr, Xte, ytr, yte = data[ctx]
    imps = [importance(shap_values(make_model("XGB", seed=s).fit(Xtr, ytr), Xte)) for s in (1, 2, 3)]
    R[f"seed_XGB_{ctx}"] = float(np.mean([compare(a, b)["rho"] for a, b in itertools.combinations(imps, 2)]))
print("seed", R["seed_XGB_novpn"], R["seed_XGB_vpn"], flush=True)
for lab, kw in (("depth3", dict(max_depth=3)), ("depth10", dict(max_depth=10))):
    imps = {}; f1 = {}
    for ctx in CTX:
        Xtr, Xte, ytr, yte = data[ctx]
        mdl = make_model("XGB", **kw).fit(Xtr, ytr); f1[ctx] = f1_score(yte, mdl.predict(Xte), average="macro")
        imps[ctx] = importance(shap_values(mdl, Xte))
    R[f"hp_XGB_{lab}"] = dict(f1=f1, cross=compare(imps["novpn"], imps["vpn"]))
    print(lab, R[f"hp_XGB_{lab}"], flush=True)
save()
# --- cluster-grouped CV (proxy for session grouping), XGB
for ctx in CTX:
    Xtr, Xte, ytr, yte = data[ctx]
    Z = StandardScaler().fit_transform(np.log1p(Xtr.clip(lower=-1) + 1))
    g = KMeans(n_clusters=50, n_init=3, random_state=0).fit_predict(Z)
    out = {}
    for lab, splitter in (("stratified", StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, ytr)),
                          ("grouped", GroupKFold(5).split(Xtr, ytr, g))):
        f1s = [f1_score(ytr[te], make_model("XGB").fit(Xtr.iloc[tr], ytr[tr]).predict(Xtr.iloc[te]), average="macro")
               for tr, te in splitter]
        out[lab] = (float(np.mean(f1s)), float(np.std(f1s)))
    R[f"gcv_{ctx}"] = out
    print(ctx, out, flush=True)
save()
print("done")
