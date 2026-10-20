"""v13: re-run the conditional permutation importance of p5_concerns4.py (same code, same seeds) and keep the
values at full precision. p5 stored them rounded to 5 decimals, which can turn tiny positive or negative values
into exact zeros and so change the ties that the average-rank correlation sees. Writes results/p8b_cpi_full.json
and checks that the rounded values reproduce p5_results.json."""
from c2 import *
D = {c: load(c) for c in CTX}
P5 = json.load(open(OUT2 / "p5_results.json"))
def cond_pi(model, Xtr, Xte, yte, reps=10, seed=0):          # copied verbatim from p5_concerns4.py
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
R = {}
for m in ("RF", "XGB"):
    cp = {}
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        cp[c] = cond_pi(make_model(m).fit(Xtr, ytr), Xtr, Xte, yte)
        stored = pd.Series(P5[f"cpi_{m}_{c}"]["values"])[FEATURES]
        R[f"cpi_{m}_{c}"] = dict(values=cp[c].to_dict(), max_abs_diff_vs_p5_rounded=float((cp[c].round(5) - stored).abs().max()),
                                 n_nonpositive=int((cp[c] <= 0).sum()))
    a, b = cp["novpn"], cp["vpn"]
    R[f"cpi_cross_{m}"] = dict(rho_average_clipped=float(compare(a.clip(lower=0) + 1e-12, b.clip(lower=0) + 1e-12)["rho"]),
                               tau_b_average_clipped=float(compare(a.clip(lower=0) + 1e-12, b.clip(lower=0) + 1e-12)["tau"]),
                               rho_raw_unclipped=float(spearmanr(a, b)[0]))
    print(m, R[f"cpi_cross_{m}"], {c: R[f"cpi_{m}_{c}"]["max_abs_diff_vs_p5_rounded"] for c in CTX}, flush=True)
jdump(R, "p8b_cpi_full.json")
