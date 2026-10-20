"""Repeated within-context reference designs (Concerns 1 and 2).

design = random  : each repeat redraws a stratified 80/20 partition of the context, then splits the
                   training part into two disjoint class-stratified halves (size-matched across contexts).
design = grouped : same partition, but halves are formed from whole k-means clusters fitted within each
                   class on log(x+2)-standardised training features (proxy for capture sessions), so
                   near-duplicate flows cannot sit in both halves.
design = ordered<B> / orderedcal / orderedhalf : halves are formed from whole blocks of consecutive rows of the original
                   ARFF files (within class, about B training flows per block, or the capture-count
                   calibrated number of blocks; orderedhalf = first versus second half of each class). Row order keeps capture locality (p6_arff_order.py), so this
                   grouping does not depend on feature-space clustering.
design = null    : permutation null. Flows are pooled within class across both contexts and randomly
                   reassigned to two pseudo-contexts with the original per-class counts; the random
                   design is then applied to the pseudo-contexts.
design = null_<grouped design> (v14), e.g. null_grouped35, null_calibrated, null_orderedcal, null_orderedhalf:
                   the same pseudo-contexts as `null` (same seeds), with halves formed by that grouped design.
                   For row blocks, flows keep their position in their original ARFF file and are ordered by
                   (source file, position), so a block holds consecutive rows of one file.
                   SUPERSEDED: these halves can differ in how many flows come from the Non-VPN and the VPN file
                   (first-versus-second-half: almost completely), so a within pair is partly a cross-context pair.
design = nullstrat_<grouped design> (v14): same pseudo-contexts, but the grouping is applied separately to the
                   flows of each origin (Non-VPN file, VPN file) within each class, so both halves have the same
                   origin mix. Cluster counts and calibrated counts follow the origin context.
                   Still not a fair reference: flows are reassigned one by one, so flows of one capture sit in
                   both pseudo-contexts; grouped halves share none, but pseudo cross pairs do.
design = nullgroup_<grouped design> (v14, used in the paper): pseudo-contexts are built from whole capture-sized
                   groups (the calibrated clusters, or for row-block designs the calibrated row blocks, formed per
                   class and origin on the pooled flows), split within each origin in proportion to the original
                   per-class counts; then the grouped design runs within each class and origin as in nullstrat.
Every half-model has its own seed. XGB explains the full test partition of the repeat; RF explains a
class-stratified sample of 800 test records (RF TreeSHAP cost).
"""
from c2 import *
import os
MODEL, DESIGN, R = sys.argv[1], sys.argv[2], int(sys.argv[3])
N_H = 3477
N_RF_EXPLAIN = 800
path = Path(os.environ["REF_OUT"]) if os.environ.get("REF_OUT") else OUT2 / f"ref_{MODEL}_{DESIGN}.pkl"   # REF_OUT: reproduction checks
IS_NULL = DESIGN == "null" or DESIGN.startswith(("null_", "nullstrat_", "nullgroup_"))
BASE = DESIGN.split("_", 1)[1] if DESIGN.startswith(("null_", "nullstrat_", "nullgroup_")) else DESIGN
STRAT = DESIGN.startswith(("nullstrat_", "nullgroup_"))
GROUPNULL = DESIGN.startswith("nullgroup_")
EXPLAIN_OTHER = os.environ.get("EXPLAIN_OTHER") == "1"   # also store rec_other: explanation on the other context's explained records
try: store = pickle.load(open(path, "rb"))
except FileNotFoundError: store = {}
P = {c: pool(c) for c in CTX}

POSR = {}   # ARFF row position of every flow of the (pseudo-)contexts of the current repeat (row-block designs)
ORIGINR = {}  # origin of every flow of the pseudo-contexts (0 = Non-VPN file, 1 = VPN file); nullstrat designs
def unit_groups(Xk, posk, ctx, k, seed):
    """Capture-sized units of one class and origin on the pooled flows: calibrated row blocks or calibrated clusters."""
    kc = calibrated_k(ctx, k)
    if ORDERED: return [g for g in np.array_split(np.argsort(posk), kc) if len(g)]
    lab = KMeans(n_clusters=min(kc, len(Xk)), n_init=3, random_state=seed).fit_predict(StandardScaler().fit_transform(logx2(Xk)))
    return [np.flatnonzero(lab == g) for g in np.unique(lab)]

def contexts_for_group_null(r):
    rng = np.random.default_rng(31000 + r)
    out = {c: ([], [], [], []) for c in CTX}     # X, y, origin, pos
    for k in range(7):
        n_src = [int((P[c][1] == k).sum()) for c in CTX]; f_nv = n_src[0] / sum(n_src)
        for o, c in enumerate(CTX):
            m = P[c][1] == k; Xk = P[c][0][m].reset_index(drop=True)
            posk = POS[c][m] if ORDERED else np.zeros(len(Xk))
            groups = unit_groups(Xk, posk, c, k, 31000 + r + 7 * o + k)
            groups = [groups[i] for i in rng.permutation(len(groups))]
            target = f_nv * len(Xk); got = {"novpn": 0, "vpn": 0}
            for g in groups:                      # each group goes to the pseudo-context further below its target
                side = "novpn" if (target - got["novpn"]) >= ((len(Xk) - target) - got["vpn"]) else "vpn"
                got[side] += len(g)
                out[side][0].append(Xk.iloc[g]); out[side][1].append(np.full(len(g), k))
                out[side][2].append(np.full(len(g), o)); out[side][3].append(o * 10**7 + posk[g])
    res = {}
    for c in CTX:
        res[c] = (pd.concat(out[c][0], ignore_index=True), np.concatenate(out[c][1]))
        ORIGINR[c] = np.concatenate(out[c][2]); POSR[c] = np.concatenate(out[c][3])
    return res

def contexts_for_repeat(r):
    if GROUPNULL: return contexts_for_group_null(r)
    if not IS_NULL:
        if ORDERED: POSR.update(POS)
        return P
    rng = np.random.default_rng(31000 + r)
    out = {c: ([], []) for c in CTX}; pos_k = {}; org_k = {}
    for k in range(7):
        Xs = pd.concat([P[c][0][P[c][1] == k] for c in CTX], ignore_index=True)
        if ORDERED: src_pos = np.concatenate([ci * 10**7 + POS[c][P[c][1] == k] for ci, c in enumerate(CTX)])
        n_nv = int((P["novpn"][1] == k).sum())
        perm = rng.permutation(len(Xs))
        for c, ids in (("novpn", perm[:n_nv]), ("vpn", perm[n_nv:])):
            out[c][0].append(Xs.iloc[ids]); out[c][1].append(np.full(len(ids), k))
            if ORDERED: pos_k.setdefault(c, []).append(src_pos[ids])
            org_k.setdefault(c, []).append((ids >= int((P["novpn"][1] == k).sum())).astype(int))
    if ORDERED: POSR.update({c: np.concatenate(v) for c, v in pos_k.items()})
    ORIGINR.update({c: np.concatenate(v) for c, v in org_k.items()})
    return {c: (pd.concat(out[c][0], ignore_index=True), np.concatenate(out[c][1])) for c in CTX}

CL_SIZE = int(BASE[7:]) if BASE.startswith("grouped") and len(BASE) > 7 else 140
# Calibrated design: number of clusters per class = number of capture files of that class in the
# released PCAPs, at least 4. The counts come from the capture-prefix list of munhouiani's independent
# Deep Packet implementation on GitHub (paper ref. [56]), not from the Deep Packet paper itself; see
# designs_meta.py. BROWSING has no dedicated captures, so it uses the context's median flows per capture file.
from designs_meta import CAPTURE_FILES, ARFF_COUNTS
def calibrated_k(ctx, k):
    if k == 0:
        per_file = np.median([ARFF_COUNTS[ctx][c] / CAPTURE_FILES[ctx][c] for c in (1, 2, 3, 5, 6)])
        return max(4, int(round(ARFF_COUNTS[ctx][0] / per_file)))
    return max(4, CAPTURE_FILES[ctx][k])

ORDERED = BASE.startswith("ordered")
if ORDERED:
    POS = {c: np.asarray(v) for c, v in json.load(open(OUT2 / "arff_order_positions.json")).items()}
    BLOCK = None if BASE == "orderedcal" else "half" if BASE == "orderedhalf" else int(BASE[7:])

def grouped_halves(Xtr, ytr, seed, ctx=None, pos=None, origin=None):
    """origin (nullstrat designs only): 0/1 per training flow; the grouping then runs within each class and origin,
    with cluster and calibrated counts of the origin context. With origin=None this is the original procedure."""
    if not ORDERED: Z = StandardScaler().fit_transform(logx2(Xtr))
    rng = np.random.default_rng(seed)
    A, B = [], []
    for k in range(7):
      for o in ((None,) if origin is None else (0, 1)):
        ids = np.flatnonzero(ytr == k) if o is None else np.flatnonzero((ytr == k) & (origin == o))
        cx = ctx if o is None else CTX[o]
        def cal(cx, k):
            if o is None: return calibrated_k(cx, k)
            # null strata hold only part of the origin's flows: keep groups capture-sized by scaling the count
            return max(2, int(round(calibrated_k(cx, k) * len(ids) / (0.8 * int((P[cx][1] == k).sum())))))
        if ORDERED:
            kc = (cal(cx, k) if BLOCK is None else 2 if BLOCK == "half"
                  else max(4, int(round(len(ids) / BLOCK))))
            groups = np.array_split(ids[np.argsort(pos[ids])], kc)   # contiguous runs of ARFF rows
        else:
            kc = cal(cx, k) if BASE == "calibrated" else max(4, int(round(len(ids) / CL_SIZE)))
            if o is not None: kc = max(1, min(kc, len(ids)))
            if len(ids) == 0: continue
            lab = KMeans(n_clusters=kc, n_init=3, random_state=int(seed) + k + (0 if o is None else 50 * o)).fit_predict(Z[ids])
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
        if BASE.startswith("grouped") or BASE == "calibrated" or ORDERED:
            halves[c] = grouped_halves(Xtr, ytr, 41000 + 100 * r + ci, ctx=c,
                                       pos=POSR[c][itr] if ORDERED else None, origin=ORIGINR[c][itr] if STRAT else None)
        else:
            rng = np.random.default_rng(5000 + 100 * r + ci)
            a, b = [], []
            for k in range(7):
                ids = rng.permutation(np.flatnonzero(ytr == k)); h = len(ids) // 2
                a += list(ids[:h]); b += list(ids[h:2 * h])
            halves[c] = (np.array(a), np.array(b))
    n_common = N_H if not (BASE.startswith("grouped") or BASE == "calibrated" or ORDERED) else min(min(len(h) for h in halves[c]) for c in CTX)
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
            if EXPLAIN_OTHER:
                oi = 1 - ci; oc = CTX[oi]; Xte_o, yte_o = parts[oc][2], parts[oc][3]
                ex_o = strat_sub(np.arange(len(yte_o)), yte_o, N_RF_EXPLAIN, 61000 + 100 * r + oi) if MODEL == "RF" else np.arange(len(yte_o))
                store[(c, r, h)]["rec_other"] = np.abs(shap_values(m, Xte_o.iloc[ex_o])).mean(axis=0).astype(np.float32)
            print(MODEL, DESIGN, c, r, h, len(ids), f"f1={store[(c, r, h)]['f1']:.4f}", f"{tic() - t:.0f}s", flush=True)
    pickle.dump(store, open(path, "wb"))
print("done")
