"""Repeated within-context reference designs (Concerns 1 and 2).

design = random  : each repeat redraws a stratified 80/20 partition of the context, then splits the
                   training part into two disjoint class-stratified halves (size-matched across contexts).
design = grouped : same partition, but halves are formed from whole k-means clusters fitted within each
                   class on log(x+2)-standardised training features (proxy for capture sessions), so
                   near-duplicate flows cannot sit in both halves.
design = null    : permutation null. Flows are pooled within class across both contexts and randomly
                   reassigned to two pseudo-contexts with the original per-class counts; the random
                   design is then applied to the pseudo-contexts.
Every half-model has its own seed. XGB explains the full test partition of the repeat; RF explains a
class-stratified sample of 800 test records (RF TreeSHAP cost).
"""
from c2 import *
MODEL, DESIGN, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
N_H = 3477
N_RF_EXPLAIN = 800
path = OUT2 / f"ref_{MODEL}_{DESIGN}.pkl"
try: store = pickle.load(open(path, "rb"))
except FileNotFoundError: store = {}
P = {c: pool(c) for c in CTX}

def contexts_for_repeat(r):
    if DESIGN != "null": return P
    rng = np.random.default_rng(31000 + r)
    out = {c: ([], []) for c in CTX}
    for k in range(7):
        Xs = pd.concat([P[c][0][P[c][1] == k] for c in CTX], ignore_index=True)
        n_nv = int((P["novpn"][1] == k).sum())
        perm = rng.permutation(len(Xs))
        for c, ids in (("novpn", perm[:n_nv]), ("vpn", perm[n_nv:])):
            out[c][0].append(Xs.iloc[ids]); out[c][1].append(np.full(len(ids), k))
    return {c: (pd.concat(out[c][0], ignore_index=True), np.concatenate(out[c][1])) for c in CTX}

CL_SIZE = int(DESIGN[7:]) if DESIGN.startswith("grouped") and len(DESIGN) > 7 else 140
# Calibrated design: number of clusters per class = number of capture files of that class in the
# released PCAPs (prefix list of the Deep Packet code, Lotfollahi et al.), at least 4. BROWSING has no
# dedicated captures, so it uses the context's median flows per capture file (see revision2/README.md).
CAPTURE_FILES = {"novpn": {1: 20, 2: 32, 3: 4, 4: 1, 5: 19, 6: 34}, "vpn": {1: 10, 2: 6, 3: 2, 4: 1, 5: 5, 6: 7}}
ARFF_COUNTS = {"novpn": {1: 890, 2: 1018, 3: 249, 4: 1000, 5: 482, 6: 2826, 0: 2500},
               "vpn": {1: 1196, 2: 1932, 3: 491, 4: 928, 5: 475, 6: 2271, 0: 2500}}
def calibrated_k(ctx, k):
    if k == 0:
        per_file = np.median([ARFF_COUNTS[ctx][c] / CAPTURE_FILES[ctx][c] for c in (1, 2, 3, 5, 6)])
        return max(4, int(round(ARFF_COUNTS[ctx][0] / per_file)))
    return max(4, CAPTURE_FILES[ctx][k])

def grouped_halves(Xtr, ytr, seed, ctx=None):
    Z = StandardScaler().fit_transform(logx2(Xtr))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for k in range(7):
        ids = np.flatnonzero(ytr == k)
        kc = calibrated_k(ctx, k) if DESIGN == "calibrated" else max(4, int(round(len(ids) / CL_SIZE)))
        lab = KMeans(n_clusters=kc, n_init=3, random_state=int(seed) + k).fit_predict(Z[ids])
        groups = [ids[lab == g] for g in range(kc)]
        order = rng.permutation(len(groups))
        groups = sorted([groups[i] for i in order], key=len, reverse=True)
        a, b = [], []
        for g in groups:
            (a if len(a) <= len(b) else b).extend(g)
        n = min(len(a), len(b))
        A += list(rng.choice(a, n, replace=False)); B += list(rng.choice(b, n, replace=False))
    return np.array(A), np.array(B)

for r in range(R):
    if all((c, r, h) in store for c in CTX for h in (0, 1)): continue
    D = contexts_for_repeat(r)
    halves, parts = {}, {}
    for ci, c in enumerate(CTX):
        X, y = D[c]
        itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=1000 + r)
        Xtr, ytr = X.iloc[itr].reset_index(drop=True), y[itr]
        Xte, yte = X.iloc[ite].reset_index(drop=True), y[ite]
        parts[c] = (Xtr, ytr, Xte, yte)
        if DESIGN.startswith("grouped") or DESIGN == "calibrated":
            halves[c] = grouped_halves(Xtr, ytr, 41000 + 100 * r + ci, ctx=c)
        else:
            rng = np.random.default_rng(5000 + 100 * r + ci)
            a, b = [], []
            for k in range(7):
                ids = rng.permutation(np.flatnonzero(ytr == k)); h = len(ids) // 2
                a += list(ids[:h]); b += list(ids[h:2 * h])
            halves[c] = (np.array(a), np.array(b))
    n_common = N_H if not (DESIGN.startswith("grouped") or DESIGN == "calibrated") else min(min(len(h) for h in halves[c]) for c in CTX)
    for ci, c in enumerate(CTX):
        Xtr, ytr, Xte, yte = parts[c]
        if MODEL == "RF":
            ex_ids = strat_sub(np.arange(len(yte)), yte, N_RF_EXPLAIN, 61000 + 100 * r + ci)
        else:
            ex_ids = np.arange(len(yte))
        for h, ids in enumerate(halves[c]):
            ids = strat_sub(ids, ytr, n_common, 7000 + 100 * r + 10 * ci + h)
            seed = int(9000 + 100 * r + 10 * ci + h)
            t = tic()
            m = make_model(MODEL, seed=seed, **({"n_jobs": 1} if MODEL == "XGB" else {})).fit(Xtr.iloc[ids], ytr[ids])
            pred = m.predict(Xte)
            sv = shap_values(m, Xte.iloc[ex_ids])
            store[(c, r, h)] = dict(rec=np.abs(sv).mean(axis=0).astype(np.float32), y=yte[ex_ids], n=len(ids),
                                    seed=seed, f1=f1_score(yte, pred, average="macro"))
            print(MODEL, DESIGN, c, r, h, len(ids), f"f1={store[(c, r, h)]['f1']:.4f}", f"{tic() - t:.0f}s", flush=True)
    pickle.dump(store, open(path, "wb"))
print("done")
