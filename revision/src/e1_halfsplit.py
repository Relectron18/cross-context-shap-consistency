"""Concern 1/2 + seed variance: repeated disjoint, size-matched half-splits.

For each repeat r and context, the original training partition is split (class-stratified)
into two disjoint halves; each half is subsampled (stratified) to a common size N_H so that
halves are size-matched across contexts. Each fit uses its own model seed. Every fitted model
is explained on the full fixed test set of its context. We store the per-record mean |SHAP|
(averaged over output classes) so that importances can later be computed (a) on the full
test set, (b) on disjoint test halves, and (c) with true classes weighted equally.
"""
import sys, pickle
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from common import *

MODEL = sys.argv[1]
R = int(sys.argv[2])
N_H = 3477  # floor(6955 / 2): smallest half across contexts

out_path = OUT / f"halfsplit_{MODEL}.pkl"
try:
    store = pickle.load(open(out_path, "rb"))
except FileNotFoundError:
    store = {}

data = {c: load(c) for c in CTX}
for r in range(R):
    for ci, ctx in enumerate(CTX):
        if (ctx, r, 0) in store:
            continue
        Xtr, Xte, ytr, yte = data[ctx]
        rng = np.random.default_rng(5000 + 100 * r + ci)
        # stratified disjoint halves
        idx_a, idx_b = [], []
        for c in np.unique(ytr):
            ids = rng.permutation(np.flatnonzero(ytr == c))
            h = len(ids) // 2
            idx_a += list(ids[:h]); idx_b += list(ids[h:2 * h])
        for h, ids in enumerate((np.array(idx_a), np.array(idx_b))):
            # stratified subsample to N_H
            if len(ids) > N_H:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=N_H,
                                             random_state=int(7000 + 100 * r + 10 * ci + h))
                keep, _ = next(sss.split(ids, ytr[ids]))
                ids = ids[keep]
            seed = int(9000 + 100 * r + 10 * ci + h)
            t = tic()
            m = make_model(MODEL, seed=seed).fit(Xtr.iloc[ids], ytr[ids])
            pred = m.predict(Xte)
            sv = shap_values(m, Xte)
            rec = np.abs(sv).mean(axis=0).astype(np.float32)  # (N_test, F)
            store[(ctx, r, h)] = dict(rec=rec, n=len(ids), seed=seed,
                                      f1=f1_score(yte, pred, average="macro"))
            print(MODEL, ctx, r, h, len(ids), f"f1={store[(ctx, r, h)]['f1']:.4f}",
                  f"{tic() - t:.0f}s", flush=True)
        pickle.dump(store, open(out_path, "wb"))
print("done")
