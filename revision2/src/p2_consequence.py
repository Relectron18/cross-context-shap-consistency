"""Concern 3: direct model transfer and a fuller Top-k transfer test (single environment)."""
from c2 import *
R = {}
D = {c: load(c) for c in CTX}
OTHER = {"novpn": "vpn", "vpn": "novpn"}
mk = lambda m, s=42: make_model(m, seed=s)
# ---- direct model transfer: train on one context, test on the other
for mname in ("RF", "XGB"):
    for src in CTX:
        Xtr, _, ytr, _ = D[src]; m = mk(mname).fit(Xtr, ytr)
        for tgt in CTX:
            Xte, yte = D[tgt][1], D[tgt][3]; pr = m.predict(Xte)
            R[f"transfer_{mname}_{src}_to_{tgt}"] = dict(f1=f1_score(yte, pr, average="macro"), acc=accuracy_score(yte, pr),
                                                        recall=recall_score(yte, pr, average=None).tolist())
        print(mname, src, {t: round(R[f"transfer_{mname}_{src}_to_{t}"]["f1"], 4) for t in CTX}, flush=True)
jdump(R, "p2_results.json")
# ---- Top-k selection on inner splits, 5 selection seeds
SEEDS = [20261008 + s for s in range(5)]
rank_lists = {}
for mname in ("XGB", "RF"):
    for c in CTX:
        Xtr, _, ytr, _ = D[c]
        for si, s in enumerate(SEEDS):
            a, b = train_test_split(np.arange(len(ytr)), test_size=0.2, stratify=ytr, random_state=s)
            sel = mk(mname).fit(Xtr.iloc[a], ytr[a])
            ev = b if mname == "XGB" else strat_sub(b, ytr, 800, s)
            imp = pd.Series(np.abs(shap_values(sel, Xtr.iloc[ev])).mean(axis=(0, 1)), index=FEATURES)
            rank_lists[(mname, c, si)] = list(imp.sort_values(ascending=False).index)
        print("selected", mname, c, flush=True)
R["rank_lists"] = {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in rank_lists.items()}
jdump(R, "p2_results.json")
rng = np.random.default_rng(777)
RANDOM10 = [sorted(rng.choice(FEATURES, 10, replace=False).tolist()) for _ in range(20)]
R["random10_sets"] = RANDOM10
cache = {}
def score(mname, c, feats):
    key = (mname, c, tuple(sorted(feats)))
    if key not in cache:
        Xtr, Xte, ytr, yte = D[c]
        m = mk(mname).fit(Xtr[list(feats)], ytr); cache[key] = m.predict(Xte[list(feats)])
    return cache[key]
def boot_diff(yte, pa, pb, B=3000, seed=0):
    rng = np.random.default_rng(seed); idx_c = [np.flatnonzero(yte == k) for k in range(7)]
    out = np.empty(B)
    for i in range(B):
        ii = np.concatenate([rng.choice(ix, len(ix)) for ix in idx_c])
        out[i] = f1_score(yte[ii], pa[ii], average="macro") - f1_score(yte[ii], pb[ii], average="macro")
    return np.percentile(out, [2.5, 97.5]).tolist()
rows = []
for mname in ("XGB", "RF"):
    for c in CTX:
        yte = D[c][3]
        full = f1_score(yte, score(mname, c, FEATURES), average="macro")
        rnd = [f1_score(yte, score(mname, c, s_), average="macro") for s_ in RANDOM10]
        for si in range(len(SEEDS)):
            own_l, tr_l = rank_lists[(mname, c, si)], rank_lists[(mname, OTHER[c], si)]
            for k in (5, 10, 15):
                po, pt = score(mname, c, own_l[:k]), score(mname, c, tr_l[:k])
                fo, ft = f1_score(yte, po, average="macro"), f1_score(yte, pt, average="macro")
                row = dict(model=mname, target=c, seed=si, k=k, shared=len(set(own_l[:k]) & set(tr_l[:k])),
                           own=fo, transferred=ft, diff=fo - ft, full=full)
                if k == 10:
                    row["bottom10"] = f1_score(yte, score(mname, c, own_l[-10:]), average="macro")
                    row["own_pct_random"] = float(np.mean(np.array(rnd) < fo)); row["tr_pct_random"] = float(np.mean(np.array(rnd) < ft))
                    if si == 0:
                        row["ci_diff"] = boot_diff(yte, po, pt)
                rows.append(row)
        R[f"random10_{mname}_{c}"] = dict(mean=float(np.mean(rnd)), min=float(np.min(rnd)), max=float(np.max(rnd)), all=rnd)
        print(mname, c, "done", flush=True)
        pd.DataFrame(rows).to_csv(OUT2 / "p2_topk.csv", index=False); jdump(R, "p2_results.json")
df = pd.DataFrame(rows)
R["topk_summary"] = df.groupby(["model", "target", "k"])[["shared", "own", "transferred", "diff"]].agg(["mean", "min", "max"]).round(4).to_string()
print(R["topk_summary"])
jdump(R, "p2_results.json"); print("done")
