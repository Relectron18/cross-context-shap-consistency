"""Summarise the repeated reference designs into gaps (feature and family level)."""
from c2 import *
from itertools import product
ALT_FAMILIES = {"Duration and totals": ["duration", "total_fiat", "total_biat"],
                "Directional IAT (min/max/mean)": ["min_fiat", "min_biat", "max_fiat", "max_biat", "mean_fiat", "mean_biat"],
                **{k: v for k, v in FAMILIES.items() if k not in ("Flow duration", "Directional IAT")}}
CLUST = json.load(open(ROOT / "revision" / "results" / "clusters_0.2.json"))
def fam_rho(a, b, fams, mean=False):
    f = family_mean_imp if mean else family_imp
    return spearmanr(f(a, fams), f(b, fams))[0]
def summarise(model, design):
    st = pickle.load(open(OUT2 / f"ref_{model}_{design}.pkl", "rb"))
    reps = sorted({k[1] for k in st if all((c, k[1], h) in st for c in CTX for h in (0, 1))})
    rows = []
    for r in reps:
        imp = {(c, h): pd.Series(st[(c, r, h)]["rec"].mean(0), index=FEATURES) for c in CTX for h in (0, 1)}
        impw = {(c, h): pd.Series(np.mean([st[(c, r, h)]["rec"][st[(c, r, h)]["y"] == k].mean(0) for k in range(7)], axis=0), index=FEATURES) for c in CTX for h in (0, 1)}
        def agg(fn, I):
            w = [fn(I[(c, 0)], I[(c, 1)]) for c in CTX]
            x = np.mean([fn(I[("novpn", a)], I[("vpn", b)]) for a, b in product((0, 1), (0, 1))])
            return w[0], w[1], x, np.mean(w) - x
        row = dict(r=r, f1_nv=np.mean([st[("novpn", r, h)]["f1"] for h in (0, 1)]), f1_vpn=np.mean([st[("vpn", r, h)]["f1"] for h in (0, 1)]),
                   n=st[("novpn", r, 0)]["n"])
        for name, fn, I in (("rho", lambda a, b: compare(a, b)["rho"], imp), ("rho_w", lambda a, b: compare(a, b)["rho"], impw),
                            ("rbo", lambda a, b: compare(a, b)["rbo"], imp), ("top5", lambda a, b: compare(a, b)["top5"], imp),
                            ("kappa5", lambda a, b: kappa(compare(a, b)["top5"], 5), imp), ("l1", lambda a, b: compare(a, b)["l1"], imp),
                            ("fam", lambda a, b: fam_rho(a, b, FAMILIES), imp), ("fam_mean", lambda a, b: fam_rho(a, b, FAMILIES, True), imp),
                            ("fam_alt", lambda a, b: fam_rho(a, b, ALT_FAMILIES), imp), ("clust", lambda a, b: fam_rho(a, b, CLUST), imp)):
            w0, w1, x, g = agg(fn, I)
            row.update({f"{name}_wnv": w0, f"{name}_wvpn": w1, f"{name}_cross": x, f"{name}_gap": g})
        rows.append(row)
    return pd.DataFrame(rows)
if __name__ == "__main__":
    S = {}
    for model in ("XGB", "RF"):
        for design in ("random", "grouped10", "grouped35", "grouped", "null"):
            try: df = summarise(model, design)
            except FileNotFoundError: continue
            df.to_csv(OUT2 / f"ref_summary_{model}_{design}.csv", index=False)
            S[f"{model}_{design}"] = dict(R=len(df), mean=df.mean().round(4).to_dict(), sd=df.std(ddof=1).round(4).to_dict(),
                                          min=df.min().round(4).to_dict(), max=df.max().round(4).to_dict())
            print(model, design, len(df), {k: round(df[k].mean(), 4) for k in ("rho_wnv", "rho_wvpn", "rho_cross", "rho_gap", "rbo_gap", "top5_gap", "fam_gap", "f1_nv", "f1_vpn")}, "sd gap", round(df.rho_gap.std(), 4))
    jdump(S, "ref_summary.json")
