"""Concern 6 (class mix, direct test) and Concern 7 (minimum-IAT shifts)."""
from c2 import *
from sklearn.inspection import permutation_importance
R = {}
try: R = json.load(open(OUT2 / "p4_results.json"))
except FileNotFoundError: pass
D = {c: load(c) for c in CTX}
OTHER = {"novpn": "vpn", "vpn": "novpn"}
EX = {c: strat_sub(np.arange(len(D[c][3])), D[c][3], 800, 42) for c in CTX}
def ex_ids(mname, c): return EX[c] if mname == "RF" else np.arange(len(D[c][3]))
def prim(mname, c, ids=None):
    rec = np.load(OUT2 / f"primary_rec_{mname}_{c}.npy")
    return pd.Series(rec[ex_ids(mname, c) if ids is None else ids].mean(0), index=FEATURES)
def fit_imp(mname, c, Xtr, ytr, seed=42, cols=FEATURES):
    Xte, yte = D[c][1], D[c][3]
    m = make_model(mname, seed=seed).fit(Xtr[cols], ytr)
    ids = ex_ids(mname, c)
    imp = pd.Series(np.abs(shap_values(m, Xte[cols].iloc[ids])).mean(axis=(0, 1)), index=cols)
    return imp, f1_score(yte, m.predict(Xte[cols]), average="macro")
PRIM = {(m, c): prim(m, c) for m in ("RF", "XGB") for c in CTX}
PRIM_FULL = {(m, c): prim(m, c, np.arange(len(D[c][3]))) for m in ("RF", "XGB") for c in CTX}
# ---------------- class-mix swap
N = 3400
props = {c: np.bincount(D[c][2]) / len(D[c][2]) for c in CTX}
def counts(p):
    n = np.floor(p * N).astype(int); n[np.argmax(p)] += N - n.sum(); return n
rows = []
for mname, nr in ((("XGB", 5), ("RF", 3)) if "classmix_summary" not in R else ()):
    for c in CTX:
        Xtr, _, ytr, _ = D[c]
        for r in range(nr):
            rng = np.random.default_rng(300 + r + (0 if c == "novpn" else 50))
            imps = {}
            for lab, p in (("natural", props[c]), ("swapped", props[OTHER[c]])):
                n = counts(p); assert all(n[k] <= (ytr == k).sum() for k in range(7)), (c, lab, n)
                ids = np.concatenate([rng.choice(np.flatnonzero(ytr == k), n[k], replace=False) for k in range(7)])
                imps[lab], f1 = fit_imp(mname, c, Xtr.iloc[ids].reset_index(drop=True), ytr[ids], seed=42 + r)
                rows.append(dict(model=mname, ctx=c, r=r, mix=lab, f1=f1,
                                 rho_same=compare(imps[lab], PRIM[(mname, c)])["rho"],
                                 rho_other=compare(imps[lab], PRIM_FULL[(mname, OTHER[c])])["rho"],
                                 rbo_other=compare(imps[lab], PRIM_FULL[(mname, OTHER[c])])["rbo"],
                                 top5_other=compare(imps[lab], PRIM_FULL[(mname, OTHER[c])])["top5"]))
            rows.append(dict(model=mname, ctx=c, r=r, mix="nat_vs_swap", rho_same=compare(imps["natural"], imps["swapped"])["rho"]))
            print(rows[-3:], flush=True)
        pd.DataFrame(rows).to_csv(OUT2 / "p4_classmix.csv", index=False)
df = pd.DataFrame(rows) if rows else None
if rows: R["classmix_summary"] = df.groupby(["model", "ctx", "mix"])[["f1", "rho_same", "rho_other", "rbo_other", "top5_other"]].agg(["mean", "std"]).round(4).to_string()

jdump(R, "p4_results.json")
# ---------------- matched 1,330 with a within-context reference (XGB)
rows = []
for r in (range(6) if "matched1330" not in R else ()):
    imps = {}
    for ci, c in enumerate(CTX):
        Xtr, _, ytr, _ = D[c]; rng = np.random.default_rng(800 + 10 * r + ci)
        cnt = np.round(props[c] * 1330).astype(int); cnt[np.argmax(cnt)] += 1330 - cnt.sum()
        a, b = [], []
        for k in range(7):
            ids = rng.permutation(np.flatnonzero(ytr == k)); a += list(ids[:cnt[k]]); b += list(ids[cnt[k]:2 * cnt[k]])
        m1 = np.concatenate([rng.choice(np.flatnonzero(ytr == k), 190, replace=False) for k in range(7)])
        m2 = np.concatenate([rng.choice(np.flatnonzero(ytr == k), 190, replace=False) for k in range(7)])
        for lab, ids in (("nat1", a), ("nat2", b), ("mat1", m1), ("mat2", m2)):
            ids = np.array(ids)
            imps[(c, lab)], _ = fit_imp("XGB", c, Xtr.iloc[ids].reset_index(drop=True), ytr[ids], seed=42 + r + (lab[-1] == "2"))
    for c in CTX:
        rows.append(dict(r=r, kind=f"within_nat_{c}", rho=compare(imps[(c, "nat1")], imps[(c, "nat2")])["rho"]))
        rows.append(dict(r=r, kind=f"within_mat_{c}", rho=compare(imps[(c, "mat1")], imps[(c, "mat2")])["rho"]))
    for lab in ("nat", "mat"):
        rows.append(dict(r=r, kind=f"cross_{lab}", rho=np.mean([compare(imps[("novpn", f"{lab}{i}")], imps[("vpn", f"{lab}{j}")])["rho"] for i in (1, 2) for j in (1, 2)])))
if rows:
    df = pd.DataFrame(rows); df.to_csv(OUT2 / "p4_matched1330.csv", index=False)
    R["matched1330"] = df.groupby("kind")["rho"].agg(["mean", "std"]).round(4).to_dict()
jdump(R, "p4_results.json")
def csub(a, b):
    cols = [f for f in a.index if f in b.index]
    ra, rb = a[cols].rank(ascending=False), b[cols].rank(ascending=False)
    la, lb = list(a[cols].sort_values(ascending=False).index), list(b[cols].sort_values(ascending=False).index)
    return dict(rho=float(spearmanr(ra, rb)[0]), top5=len(set(la[:5]) & set(lb[:5])), rbo=rbo_ext(la, lb))
# ---------------- minimum-IAT: drop-one / drop-group retraining
MIN3 = ["min_fiat", "min_biat", "min_flowiat"]
for mname in ("XGB", "RF"):
    for c in CTX:
        Xtr, _, ytr, _ = D[c]
        base = PRIM[(mname, c)]; bs = base / base.sum()
        base_f1 = f1_score(D[c][3], make_model(mname).fit(Xtr, ytr).predict(D[c][1]), average="macro")
        for drop in MIN3 + ["all3"]:
            dl = MIN3 if drop == "all3" else [drop]
            cols = [f for f in FEATURES if f not in dl]
            imp, f1 = fit_imp(mname, c, Xtr, ytr, cols=cols)
            s = imp / imp.sum()
            gain = (s - bs[cols] / bs[cols].sum()).sort_values(ascending=False)
            R[f"drop_{mname}_{c}_{drop}"] = dict(df1=f1 - base_f1, f1=f1, base_f1=base_f1,
                rank_others={f: int(ranks(imp)[f]) for f in MIN3 if f in cols},
                base_rank={f: int(ranks(base)[f]) for f in MIN3},
                share_others={f: float(s[f] * 100) for f in MIN3 if f in cols},
                base_share={f: float(bs[f] * 100) for f in MIN3},
                top_gainers={f: float(v * 100) for f, v in gain[:4].items()},
                rho_rest=csub(imp, base[cols])["rho"])
            print(mname, c, drop, R[f"drop_{mname}_{c}_{drop}"], flush=True)
        jdump(R, "p4_results.json")
    # cross-context ranking with all three minima removed
    imps = {c: fit_imp(mname, c, D[c][0], D[c][2], cols=[f for f in FEATURES if f not in MIN3])[0] for c in CTX}
    R[f"drop_all3_cross_{mname}"] = csub(imps["novpn"], imps["vpn"])
# ---------------- permutation importance (macro F1) and grouped permutation
GROUPS = {"min-IAT trio": MIN3, "duration+totals": ["duration", "total_fiat", "total_biat"], "throughput pair": ["flowBytesPerSecond", "flowPktsPerSecond"]}
PI = {}
for mname in ("XGB", "RF"):
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        m = make_model(mname).fit(Xtr, ytr)
        pi = permutation_importance(m, Xte, yte, scoring="f1_macro", n_repeats=10, random_state=0, n_jobs=1)
        PI[(mname, c)] = pd.Series(pi.importances_mean, index=FEATURES)
        base = f1_score(yte, m.predict(Xte), average="macro"); rng = np.random.default_rng(0); grp = {}
        for g, fs in GROUPS.items():
            drops = []
            for _ in range(10):
                Z = Xte.copy(); perm = rng.permutation(len(Z)); Z[fs] = Xte[fs].values[perm]
                drops.append(base - f1_score(yte, m.predict(Z), average="macro"))
            grp[g] = float(np.mean(drops))
        R[f"pi_{mname}_{c}"] = dict(rank_min={f: int(ranks(PI[(mname, c)])[f]) for f in MIN3},
                                   top5=list(PI[(mname, c)].sort_values(ascending=False).index[:5]),
                                   values={f: float(v) for f, v in PI[(mname, c)].sort_values(ascending=False).items()},
                                   rho_vs_shap=compare(PI[(mname, c)].clip(lower=0) + 1e-12, PRIM_FULL[(mname, c)])["rho"], grouped=grp)
        print(mname, c, R[f"pi_{mname}_{c}"]["rank_min"], R[f"pi_{mname}_{c}"]["top5"], grp, flush=True)
    R[f"pi_cross_{mname}"] = compare(PI[(mname, "novpn")].clip(lower=0) + 1e-12, PI[(mname, "vpn")].clip(lower=0) + 1e-12)
    jdump(R, "p4_results.json")
# ---------------- entries below -1 (all in min_flowiat): share of attribution and refit without them
for mname in ("XGB", "RF"):
    for c in CTX:
        Xtr, Xte, ytr, yte = D[c]
        rec = np.load(OUT2 / f"primary_rec_{mname}_{c}.npy"); j = FEATURES.index("min_flowiat")
        bad_te = (Xte["min_flowiat"] < -1).values
        keep = ~(Xtr["min_flowiat"] < -1).values
        imp, f1 = fit_imp(mname, c, Xtr[keep].reset_index(drop=True), ytr[keep])
        R[f"below_{mname}_{c}"] = dict(n_test_below=int(bad_te.sum()), attr_share_minflowiat=float(rec[bad_te, j].sum() / rec[:, j].sum()),
                                       rho_vs_primary=compare(imp, PRIM[(mname, c)])["rho"], ranks_min={f: int(ranks(imp)[f]) for f in MIN3}, f1=f1)
        print(mname, c, R[f"below_{mname}_{c}"], flush=True)
jdump(R, "p4_results.json"); print("done")
