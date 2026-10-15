"""Analyses for Concerns 4: min-IAT block shares, conditional permutation importance,
list x target interaction of the Top-k transfer, class-mix spread, RF 800-record check for half-models."""
from c2 import *
R = {}
D = {c: load(c) for c in CTX}
MIN3 = ["min_fiat", "min_biat", "min_flowiat"]
# ---- 1. block-summed SHAP share of the three minima (primary fits, full test)
for m in ("RF", "XGB"):
    for c in CTX:
        s = pd.Series(np.load(OUT2 / f"primary_rec_{m}_{c}.npy").mean(0), index=FEATURES); s = s / s.sum()
        R[f"block_{m}_{c}"] = dict(block=float(s[MIN3].sum() * 100), **{f: float(s[f] * 100) for f in MIN3})
    # same in the random half-split repeats (mean over halves and repeats)
    st = pickle.load(open(OUT2 / f"ref_{m}_random.pkl", "rb"))
    for c in CTX:
        v = []
        for (cc, r, h), e in st.items():
            if cc != c: continue
            s = pd.Series(e["rec"].mean(0), index=FEATURES); v.append(float(s[MIN3].sum() / s.sum() * 100))
        R[f"block_half_{m}_{c}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
print({k: v for k, v in R.items() if k.startswith("block")}, flush=True)
# ---- 2. conditional permutation importance (Strobl et al.): permute feature j within cells defined by
# quintiles of its two most correlated features (training Spearman), on the test partition.
def cond_pi(model, Xtr, Xte, yte, reps=10, seed=0):
    rng = np.random.default_rng(seed)
    base = f1_score(yte, model.predict(Xte), average="macro")
    cr = Xtr.corr(method="spearman").abs()
    out = {}
    for f in FEATURES:
        partners = cr[f].drop(f).sort_values(ascending=False).index[:2]
        cells = np.zeros(len(Xte), dtype=int)
        for p in partners:
            edges = np.unique(np.quantile(Xtr[p], [0.2, 0.4, 0.6, 0.8]))
            cells = cells * 5 + np.searchsorted(edges, Xte[p].values, side="right")
        drops = []
        for _ in range(reps):
            Z = Xte.copy(); col = Z[f].values.copy()
            for g in np.unique(cells):
                ix = np.flatnonzero(cells == g); col[ix] = col[rng.permutation(ix)]
            Z[f] = col; drops.append(base - f1_score(yte, model.predict(Z), average="macro"))
        out[f] = float(np.mean(drops))
    return pd.Series(out)
for m in ("RF", "XGB"):
    cp = {}
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        mdl = make_model(m).fit(Xtr, ytr)
        cp[c] = cond_pi(mdl, Xtr, Xte, yte)
        R[f"cpi_{m}_{c}"] = dict(rank_min={f: int(ranks(cp[c])[f]) for f in MIN3}, top5=list(cp[c].sort_values(ascending=False).index[:5]),
                                 values=cp[c].round(5).to_dict())
        print(m, c, R[f"cpi_{m}_{c}"]["rank_min"], R[f"cpi_{m}_{c}"]["top5"], flush=True)
    R[f"cpi_cross_{m}"] = compare(cp["novpn"].clip(lower=0) + 1e-12, cp["vpn"].clip(lower=0) + 1e-12)
jdump(R, "p5_results.json")
# ---- 3. list x target 2x2 for the Top-k transfer
tk = pd.read_csv(OUT2 / "p2_topk.csv")
rows = []
for m in ("RF", "XGB"):
    for k in (5, 10, 15):
        for s in range(5):
            a = tk[(tk.model == m) & (tk.k == k) & (tk.seed == s)].set_index("target")
            # F1 of each list on each target: own list of target t = list from t
            nv_on_nv, vpn_on_nv = a.loc["novpn", "own"], a.loc["novpn", "transferred"]
            vpn_on_vpn, nv_on_vpn = a.loc["vpn", "own"], a.loc["vpn", "transferred"]
            rows.append(dict(model=m, k=k, seed=s,
                             match=(nv_on_nv - vpn_on_nv) + (vpn_on_vpn - nv_on_vpn),          # context-match (interaction) term
                             vpn_list_gain=((vpn_on_nv - nv_on_nv) + (vpn_on_vpn - nv_on_vpn)) / 2))  # main effect of list origin
df = pd.DataFrame(rows); df.to_csv(OUT2 / "p5_topk_2x2.csv", index=False)
R["topk_2x2"] = df.groupby(["model", "k"])[["match", "vpn_list_gain"]].agg(["mean", "min", "max"]).round(4).to_string()
print(R["topk_2x2"], flush=True)
# ---- 4. class-mix swap spread
cm = pd.read_csv(OUT2 / "p4_classmix.csv")
R["classmix_sd"] = cm.groupby(["model", "ctx", "mix"])[["f1", "rho_same", "rho_other", "rbo_other"]].agg(["mean", "std"]).round(4).to_string()
jdump(R, "p5_results.json")
# ---- 5. RF half-models: 800-record vs full-test rankings (random design, repeats 0-1)
P = {c: pool(c) for c in CTX}
chk = []
for r in (0, 1):
    for ci, c in enumerate(CTX):
        X, y = P[c]
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=1000 + r)
        Xtr, ytr = X.iloc[itr].reset_index(drop=True), y[itr]; Xte, yte = X.iloc[ite].reset_index(drop=True), y[ite]
        rng = np.random.default_rng(5000 + 100 * r + ci); a_, b_ = [], []
        for k in range(7):
            ids = rng.permutation(np.flatnonzero(ytr == k)); h = len(ids) // 2; a_ += list(ids[:h]); b_ += list(ids[h:2 * h])
        ex_ids = strat_sub(np.arange(len(yte)), yte, 800, 61000 + 100 * r + ci)
        for h, ids in enumerate((np.array(a_), np.array(b_))):
            ids = strat_sub(ids, ytr, 3477, 7000 + 100 * r + 10 * ci + h)
            mdl = make_model("RF", seed=int(9000 + 100 * r + 10 * ci + h)).fit(Xtr.iloc[ids], ytr[ids])
            rec = np.abs(shap_values(mdl, Xte)).mean(axis=0)
            full = pd.Series(rec.mean(0), index=FEATURES); sub = pd.Series(rec[ex_ids].mean(0), index=FEATURES)
            chk.append(dict(r=r, ctx=c, h=h, rho=compare(full, sub)["rho"], top5=compare(full, sub)["top5"]))
            print(chk[-1], flush=True)
R["rf800_halfmodels"] = chk
jdump(R, "p5_results.json"); print("done")
