"""v14: summarise the two new analyses into results/p11_v14.json (the only source of the v14 numbers).

1. Permutation null with grouped halves, two versions (both group within each class and origin):
   nullgroup_<design>  group-level: pseudo-contexts built from whole capture-sized groups (calibrated clusters for the
                       cluster designs, calibrated row blocks for the row-block designs);
   nullstrat_<design>  flow-level: flows reassigned one by one (flows of one capture end up in both pseudo-contexts).
   For row blocks both are close to zero; for clusters they bracket the real gap (group-level clusters also make the
   pseudo-contexts differ in feature space). null_<design> (first version, halves differ in origin mix,
   p12_null_origin_check.py) is recorded as superseded. Summaries: results/ref_summary_<MODEL>_<version>_<design>.csv (new files; existing summaries are not touched) and compared
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
        real = pd.read_csv(OUT2 / f"ref_summary_{m}_{d}.csv")
        for ver in ("nullgroup", "nullstrat"):
            df = summarise(m, f"{ver}_{d}"); df.to_csv(OUT2 / f"ref_summary_{m}_{ver}_{d}.csv", index=False)
            OUT["grouped_null"].setdefault(ver, {})[f"{m}_{d}"] = dict(
                repeats=len(df), null={k: float(df[k].mean()) for k in KEYS},
                null_gap_sd=float(df.rho_gap.std(ddof=1)), null_gap_min=float(df.rho_gap.min()), null_gap_max=float(df.rho_gap.max()),
                real_gap=float(real.rho_gap.mean()), real_gap_min=float(real.rho_gap.min()), real_gap_max=float(real.rho_gap.max()),
                real_rbo_gap=float(real.rbo_gap.mean()),
                real_minus_null=float(real.rho_gap.mean() - df.rho_gap.mean()),
                real_minus_null_rbo=float(real.rbo_gap.mean() - df.rbo_gap.mean()),
                real_repeats_above_null_max=int((real.rho_gap > df.rho_gap.max()).sum()),
                real_repeats_above_null_mean=int((real.rho_gap > df.rho_gap.mean()).sum()))
        old = summarise(m, f"null_{d}"); old.to_csv(OUT2 / f"ref_summary_{m}_null_{d}.csv", index=False)
        OUT.setdefault("superseded_null_gap", {})[f"{m}_{d}"] = float(old.rho_gap.mean())
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
# ---- row-block designs on common explained records (designs.py with EXPLAIN_OTHER=1 -> ref_<MODEL>_<design>_common.pkl)
imp_ = lambda rec: pd.Series(np.asarray(rec, float).mean(0), index=FEATURES)
for m in ("XGB", "RF"):
    for d in ("orderedcal", "orderedhalf"):
        st = pickle.load(open(OUT2 / f"ref_{m}_{d}_common.pkl", "rb")); ref = pickle.load(open(OUT2 / f"ref_{m}_{d}.pkl", "rb"))
        maxdiff = max(float(np.abs(st[k]["rec"] - ref[k]["rec"]).max()) for k in ref)
        rows = []
        for r in sorted({k[1] for k in st}):
            I = {}
            for c in CTX:
                for h in (0, 1):
                    e = st[(c, r, h)]; own_, oth = e["rec"], e["rec_other"]
                    nv, vp = (own_, oth) if c == "novpn" else (oth, own_)
                    I[(c, h, "own")], I[(c, h, "nv")], I[(c, h, "vpn")] = imp_(own_), imp_(nv), imp_(vp)
                    I[(c, h, "union")] = imp_(np.vstack([nv, vp]))
            row = dict(r=r)
            for sset in ("own", "nv", "vpn", "union"):
                w = [compare(I[(c, 0, sset)], I[(c, 1, sset)])["rho"] for c in CTX]
                x = np.mean([compare(I[("novpn", a_, sset)], I[("vpn", b_, sset)])["rho"] for a_ in (0, 1) for b_ in (0, 1)])
                row[f"gap_{sset}"] = float(np.mean(w) - x)
            rows.append(row)
        df = pd.DataFrame(rows); df.to_csv(OUT2 / f"p11_common_{m}_{d}.csv", index=False)
        real = pd.read_csv(OUT2 / f"ref_summary_{m}_{d}.csv")
        OUT["common_records_rowblocks"] = OUT.get("common_records_rowblocks", {})
        OUT["common_records_rowblocks"][f"{m}_{d}"] = dict(
            max_abs_rec_diff_vs_stored=maxdiff, own_gap_diff_vs_stored=float((df.gap_own - real.rho_gap).abs().max()),
            gap={s_: float(df[f"gap_{s_}"].mean()) for s_ in ("own", "nv", "vpn", "union")},
            gap_min={s_: float(df[f"gap_{s_}"].min()) for s_ in ("own", "nv", "vpn", "union")},
            n_positive={s_: int((df[f"gap_{s_}"] > 0).sum()) for s_ in ("nv", "vpn", "union")})
# ---- evidence for the direction of each null's bias, and half sizes of the null designs
for ver in ("nullgroup", "nullstrat"):
    for k_, g in OUT["grouped_null"][ver].items():
        m, d = k_.split("_", 1)
        real = pd.read_csv(OUT2 / f"ref_summary_{m}_{d}.csv")
        g["real_cross"] = float(real.rho_cross.mean()); g["real_within_mean"] = float(((real.rho_wnv + real.rho_wvpn) / 2).mean())
        ns = [v["n"] for v in pickle.load(open(OUT2 / f"ref_{m}_{ver}_{d}.pkl", "rb")).values()]
        g["half_size_min"], g["half_size_max"] = int(min(ns)), int(max(ns))
jdump(OUT, "p11_v14.json"); print(json.dumps(OUT, indent=1, default=float))
