"""v14: summarise the two new analyses into results/p11_v14.json (the only source of the v14 numbers).

1. Permutation null with grouped halves (designs.py nullgroup_<design>): pseudo-contexts built from whole
   capture-sized groups, then the grouped design within each class and origin. Two earlier versions are kept as
   superseded (null_<design>: halves differ in origin mix, p12_null_origin_check.py; nullstrat_<design>: flows of one
   capture in both pseudo-contexts). Per-repeat summaries are written to results/ref_summary_<MODEL>_nullgroup_<design>.csv (new files; existing summaries are not touched) and compared
   with the real-context design of the same name (ref_summary_<MODEL>_<design>.csv).
2. Common explained records (p9_common_records.py): gaps when every ranking is taken on the same records.
"""
from c2 import *
from summarize_refs import summarise
NULLS = ("grouped10", "grouped35", "calibrated", "orderedcal", "orderedhalf")
OUT = {"grouped_null": {}, "common_records": {}}
KEYS = ("rho_wnv", "rho_wvpn", "rho_cross", "rho_gap", "rbo_gap", "top5_gap", "rho_w_gap", "f1_nv", "f1_vpn")
for m in ("XGB", "RF"):
    for d in NULLS:
        df = summarise(m, f"nullgroup_{d}"); df.to_csv(OUT2 / f"ref_summary_{m}_nullgroup_{d}.csv", index=False)
        for ver in ("null", "nullstrat"):     # superseded versions, recorded for the audit trail only
            old = summarise(m, f"{ver}_{d}"); old.to_csv(OUT2 / f"ref_summary_{m}_{ver}_{d}.csv", index=False)
            OUT.setdefault(f"superseded_{ver}_gap", {})[f"{m}_{d}"] = float(old.rho_gap.mean())
        real = pd.read_csv(OUT2 / f"ref_summary_{m}_{d}.csv")
        OUT["grouped_null"][f"{m}_{d}"] = dict(
            repeats=len(df), null={k: float(df[k].mean()) for k in KEYS},
            null_gap_sd=float(df.rho_gap.std(ddof=1)), null_gap_min=float(df.rho_gap.min()), null_gap_max=float(df.rho_gap.max()),
            null_gap_n_negative=int((df.rho_gap < 0).sum()),
            real_gap=float(real.rho_gap.mean()), real_rbo_gap=float(real.rbo_gap.mean()),
            real_minus_null=float(real.rho_gap.mean() - df.rho_gap.mean()),
            real_minus_null_rbo=float(real.rbo_gap.mean() - df.rbo_gap.mean()),
            real_repeats_above_null_max=int((real.rho_gap > df.rho_gap.max()).sum()),
            real_repeats_above_null_mean=int((real.rho_gap > df.rho_gap.mean()).sum()))
    rnd = pd.read_csv(OUT2 / f"ref_summary_{m}_null.csv")
    OUT["grouped_null"][f"{m}_random"] = dict(null={k: float(rnd[k].mean()) for k in KEYS},
                                              real_gap=float(pd.read_csv(OUT2 / f"ref_summary_{m}_random.csv").rho_gap.mean()))
    c = json.load(open(OUT2 / f"p9_common_{m}.json"))
    assert c["max_abs_rec_diff_vs_stored"] < 1e-5 and c["own_gap_matches_stored"] < 1e-9, "refit does not reproduce the stored design"
    df = pd.read_csv(OUT2 / f"p9_common_{m}.csv")
    OUT["common_records"][m] = dict(
        gap={s: float(df[f"rho_gap_{s}"].mean()) for s in ("own", "nv", "vpn", "union")},
        gap_sd={s: float(df[f"rho_gap_{s}"].std(ddof=1)) for s in ("own", "nv", "vpn", "union")},
        gap_min={s: float(df[f"rho_gap_{s}"].min()) for s in ("own", "nv", "vpn", "union")},
        cross={s: float(df[f"rho_cross_{s}"].mean()) for s in ("own", "nv", "vpn", "union")},
        within_nv={s: float(df[f"rho_wnv_{s}"].mean()) for s in ("own", "nv", "vpn", "union")},
        within_vpn={s: float(df[f"rho_wvpn_{s}"].mean()) for s in ("own", "nv", "vpn", "union")},
        rbo_gap={s: float(df[f"rbo_gap_{s}"].mean()) for s in ("own", "nv", "vpn", "union")},
        n_repeats_positive_union=int((df.rho_gap_union > 0).sum()),
        primary=c["primary"], reproduction=dict(max_abs_rec_diff=c["max_abs_rec_diff_vs_stored"], own_gap_diff=c["own_gap_matches_stored"]))
jdump(OUT, "p11_v14.json"); print(json.dumps(OUT, indent=1, default=float))
