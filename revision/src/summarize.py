import pickle, json, itertools
from common import *
S = {}
data = {c: load(c) for c in CTX}
yte = {c: data[c][3] for c in CTX}
rng = np.random.default_rng(11)
thalf = {}
for c in CTX:  # fixed stratified disjoint test halves
    a, b = [], []
    for k in range(7):
        ids = rng.permutation(np.flatnonzero(yte[c] == k)); h = len(ids) // 2
        a += list(ids[:h]); b += list(ids[h:])
    thalf[c] = (np.array(a), np.array(b))

def imp_from(rec, idx=None, y=None):
    r = rec if idx is None else rec[idx]
    if y is None:
        return pd.Series(r.mean(0), index=FEATURES)
    yy = y if idx is None else y[idx]
    return pd.Series(np.mean([r[yy == k].mean(0) for k in range(7)], axis=0), index=FEATURES)

for m in ("XGB", "RF"):
    try:
        st = pickle.load(open(OUT / f"halfsplit_{m}.pkl", "rb"))
    except FileNotFoundError:
        continue
    reps = sorted({k[1] for k in st if all((c, k[1], h) in st for c in CTX for h in (0, 1))})
    res = {}
    for variant in ("fulltest", "disjointtest", "classweighted"):
        rows = []
        for r in reps:
            I = {}
            for c in CTX:
                for h in (0, 1):
                    rec = st[(c, r, h)]["rec"]
                    if variant == "fulltest":
                        I[(c, h)] = imp_from(rec)
                    elif variant == "disjointtest":
                        I[(c, h)] = imp_from(rec, thalf[c][h])
                    else:
                        I[(c, h)] = imp_from(rec, y=yte[c])
            w = {c: compare(I[(c, 0)], I[(c, 1)]) for c in CTX}
            cr = [compare(I[("novpn", a)], I[("vpn", b)]) for a in (0, 1) for b in (0, 1)]
            row = {"r": r}
            for met in ("rho", "tau", "top5", "top10", "rbo", "l1"):
                row[f"w_nv_{met}"] = w["novpn"][met]; row[f"w_vpn_{met}"] = w["vpn"][met]
                row[f"cross_{met}"] = np.mean([x[met] for x in cr])
                row[f"gap_{met}"] = (w["novpn"][met] + w["vpn"][met]) / 2 - row[f"cross_{met}"]
            rows.append(row)
        df = pd.DataFrame(rows)
        summ = {}
        for met in ("rho", "tau", "top5", "top10", "rbo", "l1"):
            g = df[f"gap_{met}"]
            summ[met] = dict(w_nv=df[f"w_nv_{met}"].mean(), w_vpn=df[f"w_vpn_{met}"].mean(), cross=df[f"cross_{met}"].mean(),
                             gap=g.mean(), sd=g.std(ddof=1), lo=g.min(), hi=g.max(),
                             se=g.std(ddof=1) / np.sqrt(len(g)), d=g.mean() / g.std(ddof=1))
        res[variant] = summ
    f1 = {c: np.mean([st[(c, r, h)]["f1"] for r in reps for h in (0, 1)]) for c in CTX}
    S[m] = dict(R=len(reps), n={c: st[(c, reps[0], 0)]["n"] for c in CTX}, f1=f1, res=res)
    print(m, "R =", len(reps), "f1", {c: round(v, 4) for c, v in f1.items()})
    for v, summ in res.items():
        for met in ("rho", "top5", "rbo", "l1"):
            s = summ[met]
            print(f"  {v:13s} {met:5s} wNV={s['w_nv']:.4f} wVPN={s['w_vpn']:.4f} cross={s['cross']:.4f} gap={s['gap']:.4f} sd={s['sd']:.4f} [{s['lo']:.4f},{s['hi']:.4f}]")
json.dump(S, open(OUT / "halfsplit_summary.json", "w"), indent=1, default=float)

# consequence
try:
    C = pickle.load(open(OUT / "consequence.pkl", "rb"))
    out = {}
    for m in ("RF", "XGB"):
        for t in CTX:
            keys = [k for k in C if k[0] == t and k[1] == m]
            if not keys: continue
            d = {k[2]: C[k]["f1"] for k in keys}
            rand = np.array([v for k, v in d.items() if k.startswith("rand")])
            row = {k: v for k, v in d.items() if not k.startswith("rand")}
            if len(rand):
                row.update(rand_mean=rand.mean(), rand_min=rand.min(), rand_max=rand.max(), rand_n=len(rand),
                           own10_pct=(rand < d.get("own10", np.nan)).mean(), trans10_pct=(rand < d.get("trans10", np.nan)).mean())
            for k in (5, 10, 15):
                if f"own{k}" in d and f"trans{k}" in d:
                    o = set(C[(t, m, f"own{k}")]["feats"]); tr = set(C[(t, m, f"trans{k}")]["feats"])
                    row[f"overlap{k}"] = len(o & tr)
            out[f"{t}_{m}"] = row
            print(t, m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    json.dump(out, open(OUT / "consequence_summary.json", "w"), indent=1, default=float)
except FileNotFoundError:
    pass
