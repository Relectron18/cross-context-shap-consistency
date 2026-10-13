"""Light analyses on primary fits and saved bootstrap runs.
- class-reweighted importance (concern 3b)
- sentinel prevalence and min_fiat/min_biat attribution from sentinel records (3c)
- RBO and share-vector L1 for primary fits and the 30 bootstrap runs (4)
- feature correlation, min_flowiat identity, data-driven clusters (4, 5)
- per-member family means (5)
"""
import pickle, json, itertools
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from common import *

P = pickle.load(open(OUT / "primary.pkl", "rb"))
R = {}

# ---------- 3b: class reweighting on primary fits
rw = {}
for m in ("RF", "XGB"):
    imp = {}
    for ctx in CTX:
        _, Xte, _, yte = load(ctx)
        imp[ctx] = importance(P[(ctx, m)]["sv"], y=yte)
        rw[(ctx, m)] = imp[ctx]
    c = compare(imp["novpn"], imp["vpn"])
    c0 = compare(P[("novpn", m)]["imp"], P[("vpn", m)]["imp"])
    # within-context: unweighted vs reweighted
    w = {ctx: compare(P[(ctx, m)]["imp"], imp[ctx]) for ctx in CTX}
    R[f"reweight_{m}"] = dict(cross_reweighted=c, cross_unweighted=c0,
                              within_unw_vs_rw={k: v["rho"] for k, v in w.items()},
                              top5_rw={ctx: list(imp[ctx].sort_values(ascending=False).index[:5]) for ctx in CTX})
pickle.dump(rw, open(OUT / "reweighted_primary.pkl", "wb"))

# ---------- 3c: sentinel prevalence
sent = {}
for ctx in CTX:
    Xtr, Xte, ytr, yte = load(ctx)
    X = pd.concat([Xtr, Xte]); y = np.concatenate([ytr, yte])
    prev = (X == -1).mean()
    sent[ctx] = prev
    percls = pd.DataFrame({c: (X[y == c] == -1).mean() for c in range(7)})
    R[f"sentinel_percls_{ctx}"] = percls.loc[["min_fiat", "min_biat", "max_fiat", "max_biat", "min_flowiat",
                                             "min_active", "min_idle"]].round(4).to_dict()
    # share of |SHAP| of min_fiat/min_biat carried by sentinel test records
    for m in ("RF", "XGB"):
        a = np.abs(P[(ctx, m)]["sv"]).mean(axis=0)  # (N,F)
        for f in ("min_fiat", "min_biat", "min_flowiat"):
            j = FEATURES.index(f)
            mask = (Xte[f].to_numpy() == -1)
            R.setdefault("sentinel_shap_share", {})[f"{ctx}_{m}_{f}"] = dict(
                frac_records=float(mask.mean()),
                frac_abs_shap=float(a[mask, j].sum() / a[:, j].sum()),
                mean_abs_sentinel=float(a[mask, j].mean()) if mask.any() else None,
                mean_abs_other=float(a[~mask, j].mean()))
S = pd.DataFrame(sent)
S = S[(S > 0).any(axis=1)]
R["sentinel_prevalence"] = S.round(4).to_dict()
S.to_csv(OUT / "sentinel_prevalence.csv")

# ---------- 4: RBO / L1 for primary fits and saved bootstrap runs
for m in ("RF", "XGB"):
    R[f"primary_metrics_{m}"] = compare(P[("novpn", m)]["imp"], P[("vpn", m)]["imp"])
rr = pd.read_csv(str(ROOT / "results") + "/stability/run_rankings.csv")
rr["m"] = rr.model.map({"RandomForest": "RF", "XGBoost": "XGB"})
boot = {}
for (ctx, m, run), g in rr.groupby(["context", "m", "run"]):
    boot[(ctx, m, run)] = g.set_index("feature")["mean_abs_shap"].loc[FEATURES]
pickle.dump(boot, open(OUT / "boot_imp.pkl", "wb"))
bsum = {}
for m in ("RF", "XGB"):
    runs = sorted({k[2] for k in boot if k[1] == m})
    rows = []
    for ctx in CTX:
        for a, b in itertools.combinations(runs, 2):
            rows.append(dict(kind=f"within_{ctx}", **compare(boot[(ctx, m, a)], boot[(ctx, m, b)])))
    for a in runs:
        for b in runs:
            rows.append(dict(kind="cross", **compare(boot[("novpn", m, a)], boot[("vpn", m, b)])))
    df = pd.DataFrame(rows)
    bsum[m] = df.groupby("kind").mean().round(4)
    print(m, "\n", bsum[m])
    R[f"boot_metrics_{m}"] = bsum[m].to_dict()

# ---------- 4/5: correlations, identity, clusters
corr = {}
for ctx in CTX:
    Xtr, _, _, _ = load(ctx)
    corr[ctx] = Xtr.corr(method="spearman")
    ident = (Xtr.min_flowiat == np.minimum(Xtr.min_fiat.where(Xtr.min_fiat >= 0, np.inf),
                                           Xtr.min_biat.where(Xtr.min_biat >= 0, np.inf))).mean()
    ident2 = (Xtr.min_flowiat <= np.minimum(Xtr.min_fiat.where(Xtr.min_fiat >= 0, np.inf),
                                            Xtr.min_biat.where(Xtr.min_biat >= 0, np.inf))).mean()
    R[f"minflowiat_identity_{ctx}"] = dict(equal=float(ident), le=float(ident2))
    c = corr[ctx]
    R[f"corr_selected_{ctx}"] = {f"{a}~{b}": round(float(c.loc[a, b]), 3) for a, b in [
        ("min_flowiat", "min_fiat"), ("min_flowiat", "min_biat"), ("min_fiat", "min_biat"),
        ("total_fiat", "total_biat"), ("total_fiat", "duration"), ("total_biat", "duration"),
        ("flowBytesPerSecond", "flowPktsPerSecond"), ("mean_flowiat", "mean_fiat"),
        ("max_flowiat", "max_fiat"), ("max_flowiat", "max_biat"), ("mean_biat", "mean_flowiat")]}
pickle.dump(corr, open(OUT / "corr.pkl", "wb"))
avg = (corr["novpn"].abs() + corr["vpn"].abs()) / 2
D = 1 - avg.to_numpy(); np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), method="average")
clusters = {}
for thr in (0.2, 0.3):
    lab = fcluster(Z, t=thr, criterion="distance")
    cl = {}
    for f, l in zip(FEATURES, lab):
        cl.setdefault(int(l), []).append(f)
    clusters[thr] = {f"C{i+1}": v for i, v in enumerate(sorted(cl.values(), key=lambda v: -len(v)))}
R["clusters"] = {str(k): v for k, v in clusters.items()}
json.dump(clusters[0.2], open(OUT / "clusters_0.2.json", "w"), indent=1)
json.dump(clusters[0.3], open(OUT / "clusters_0.3.json", "w"), indent=1)

# ---------- 5: family per-member means, clusters, on primary + bootstrap
def fam_compare(imp_a, imp_b, fams, how):
    fa = family_imp(imp_a, fams) if how == "sum" else family_mean_imp(imp_a, fams)
    fb = family_imp(imp_b, fams) if how == "sum" else family_mean_imp(imp_b, fams)
    return spearmanr(fa.rank(ascending=False), fb.rank(ascending=False))[0], fa, fb

fam_res = {}
for m in ("RF", "XGB"):
    runs = sorted({k[2] for k in boot if k[1] == m})
    for label, fams in (("semantic", FAMILIES), ("clust0.2", clusters[0.2])):
        for how in ("sum", "mean"):
            rho_p, fa, fb = fam_compare(P[("novpn", m)]["imp"], P[("vpn", m)]["imp"], fams, how)
            wn = [fam_compare(boot[(c, m, a)], boot[(c, m, b)], fams, how)[0]
                  for c in CTX for a, b in itertools.combinations(runs, 2)]
            wn_nv = np.nanmean(wn[:435]); wn_v = np.nanmean(wn[435:])
            cr = [fam_compare(boot[("novpn", m, a)], boot[("vpn", m, b)], fams, how)[0] for a in runs for b in runs]
            fam_res[(m, label, how)] = dict(primary_rho=rho_p, within_nv=wn_nv, within_vpn=wn_v,
                                            cross=np.nanmean(cr), gap=(wn_nv + wn_v) / 2 - np.nanmean(cr),
                                            nv_order=list(fa.sort_values(ascending=False).index),
                                            vpn_order=list(fb.sort_values(ascending=False).index),
                                            nv_share=(fa / fa.sum()).round(4).to_dict() if how == "sum" else (fa / fa.sum()).round(4).to_dict(),
                                            vpn_share=(fb / fb.sum()).round(4).to_dict())
            # how many distinct orderings within VPN for RF
            if label == "semantic" and how == "sum":
                orders = {c: len({tuple(family_imp(boot[(c, m, r)]).sort_values(ascending=False).index) for r in runs}) for c in CTX}
                fam_res[(m, label, how)]["distinct_orders"] = orders
for k, v in fam_res.items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("primary_rho", "within_nv", "within_vpn", "cross", "gap", "distinct_orders")})
    print("   NV:", v["nv_order"][:3], " VPN:", v["vpn_order"][:3])
pickle.dump(fam_res, open(OUT / "family_res.pkl", "wb"))

json.dump(R, open(OUT / "light_results.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in R if not k.startswith("boot_metrics")}, indent=1, default=float)[:9000])
