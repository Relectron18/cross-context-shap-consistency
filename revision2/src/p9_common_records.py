"""v14: cross-context agreement when both models are explained on the same records.

In the reference designs, within-context pairs are explained on the same test records, cross-context pairs on
the two contexts' different test partitions. This script refits the flow-level random-halves design with the
seeds of designs.py (same partitions, halves, subsamples and model seeds) and explains every half-model on
both contexts' explained records (XGBoost: full test partitions; Random Forest: the same 800-record
class-stratified samples as designs.py). It then recomputes within- and cross-context agreement with every
ranking taken on one common record set: the Non-VPN records, the VPN records, or their union.
It also does this for the primary models (full test partitions).
Checks: each refit reproduces the stored explanation on its own records (results/ref_<MODEL>_random.pkl) and the
stored primary rankings. Usage: python p9_common_records.py <RF|XGB>   ->  results/p9_common_<MODEL>.json
"""
from c2 import *
MODEL = sys.argv[1]; R = 11; N_H = 3477; N_RF_EXPLAIN = 800
P = {c: pool(c) for c in CTX}
stored = pickle.load(open(OUT2 / f"ref_{MODEL}_random.pkl", "rb"))
imp = lambda rec: pd.Series(rec.mean(0), index=FEATURES)
SETS = ("nv", "vpn", "union")
rows, maxdiff = [], 0.0
for r in range(R):
    parts, halves, ex = {}, {}, {}
    for ci, c in enumerate(CTX):
        X, y = P[c]
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=1000 + r)
        Xtr, ytr = X.iloc[itr].reset_index(drop=True), y[itr]
        Xte, yte = X.iloc[ite].reset_index(drop=True), y[ite]
        parts[c] = (Xtr, ytr, Xte, yte)
        rng = np.random.default_rng(5000 + 100 * r + ci); a, b = [], []
        for k in range(7):
            ids = rng.permutation(np.flatnonzero(ytr == k)); h = len(ids) // 2
            a += list(ids[:h]); b += list(ids[h:2 * h])
        halves[c] = (np.array(a), np.array(b))
        ex_ids = strat_sub(np.arange(len(yte)), yte, N_RF_EXPLAIN, 61000 + 100 * r + ci) if MODEL == "RF" else np.arange(len(yte))
        ex[c] = Xte.iloc[ex_ids]
    I = {}
    for ci, c in enumerate(CTX):
        Xtr, ytr, Xte, yte = parts[c]
        for h, ids in enumerate(halves[c]):
            ids = strat_sub(ids, ytr, N_H, 7000 + 100 * r + 10 * ci + h)
            m = make_model(MODEL, seed=int(9000 + 100 * r + 10 * ci + h), **({"n_jobs": 1} if MODEL == "XGB" else {})).fit(Xtr.iloc[ids], ytr[ids])
            rec = {s: np.abs(shap_values(m, ex[s2])).mean(axis=0) for s, s2 in (("nv", "novpn"), ("vpn", "vpn"))}
            own = rec["nv"] if c == "novpn" else rec["vpn"]
            maxdiff = max(maxdiff, float(np.abs(own.astype(np.float32) - stored[(c, r, h)]["rec"]).max()))
            I[(c, h, "nv")], I[(c, h, "vpn")] = imp(rec["nv"]), imp(rec["vpn"])
            I[(c, h, "union")] = imp(np.vstack([rec["nv"], rec["vpn"]]))
            I[(c, h, "own")] = imp(own)
    row = dict(r=r)
    for s in SETS + ("own",):
        w = {c: compare(I[(c, 0, s)], I[(c, 1, s)]) for c in CTX}
        cr = [compare(I[("novpn", a, s)], I[("vpn", b, s)]) for a in (0, 1) for b in (0, 1)]
        for met in ("rho", "rbo"):
            row[f"{met}_wnv_{s}"], row[f"{met}_wvpn_{s}"] = w["novpn"][met], w["vpn"][met]
            row[f"{met}_cross_{s}"] = float(np.mean([x[met] for x in cr]))
            row[f"{met}_gap_{s}"] = (row[f"{met}_wnv_{s}"] + row[f"{met}_wvpn_{s}"]) / 2 - row[f"{met}_cross_{s}"]
    rows.append(row); print(MODEL, r, {s: round(row[f"rho_gap_{s}"], 3) for s in SETS + ("own",)}, flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT2 / f"p9_common_{MODEL}.csv", index=False)
ref = pd.read_csv(OUT2 / f"ref_summary_{MODEL}_random.csv")
out = dict(model=MODEL, repeats=R, max_abs_rec_diff_vs_stored=maxdiff,
           own_gap_matches_stored=float((df.rho_gap_own - ref.rho_gap).abs().max()),
           mean={k: float(v) for k, v in df.mean().items() if k != "r"},
           sd={k: float(v) for k, v in df.std(ddof=1).items() if k != "r"},
           min={k: float(v) for k, v in df.min().items() if k != "r"}, max={k: float(v) for k, v in df.max().items() if k != "r"})
# primary models on common record sets (full test partitions)
D = {c: load(c) for c in CTX}
M_ = {c: make_model(MODEL).fit(D[c][0], D[c][2]) for c in CTX}
PR = {}
for c in CTX:
    for s, s2 in (("nv", "novpn"), ("vpn", "vpn")):
        PR[(c, s)] = np.abs(shap_values(M_[c], D[s2][1])).mean(axis=0)
prim = {}
for s in ("nv", "vpn", "union"):
    get = (lambda c: imp(np.vstack([PR[(c, "nv")], PR[(c, "vpn")]]))) if s == "union" else (lambda c, s=s: imp(PR[(c, s)]))
    prim[s] = {k: float(v) for k, v in compare(get("novpn"), get("vpn")).items()}
stored_rho = json.load(open(OUT2 / "p1_results.json"))[f"cross_{MODEL}"]["rho"]
own_rho = compare(imp(PR[("novpn", "nv")]), imp(PR[("vpn", "vpn")]))["rho"]
out["primary"] = dict(on_common=prim, own_records_rho=float(own_rho), stored_rho=float(stored_rho))
jdump(out, f"p9_common_{MODEL}.json")
print(json.dumps({k: out[k] for k in ("max_abs_rec_diff_vs_stored", "own_gap_matches_stored", "primary")}, default=float))
