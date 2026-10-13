"""Concern 6: consequence test with random-10 / bottom-10 controls and second/third cutoffs."""
import sys, pickle
from sklearn.metrics import f1_score, accuracy_score
from common import *

S = str(ROOT / "results") + "/consequence_test/"
inner = pd.read_csv(S + "inner_shap_rankings.csv")
MN = {"RandomForest": "RF", "XGBoost": "XGB"}
inner["m"] = inner.model.map(MN)
order = {(c, m): list(g.sort_values("rank").feature) for (c, m), g in inner.groupby(["context", "m"])}

rng = np.random.default_rng(314)
RANDOM_SETS = [sorted(rng.choice(FEATURES, 10, replace=False)) for _ in range(50)]

out_path = OUT / "consequence.pkl"
try:
    res = pickle.load(open(out_path, "rb"))
except FileNotFoundError:
    res = {}
data = {c: load(c) for c in CTX}
models = sys.argv[1:] or ["XGB", "RF"]
for m in models:
    for tgt in CTX:
        src = "vpn" if tgt == "novpn" else "novpn"
        Xtr, Xte, ytr, yte = data[tgt]
        conds = {"full": FEATURES}
        for k in ((5, 10, 15) if m == "XGB" else (5, 10)):
            conds[f"own{k}"] = order[(tgt, m)][:k]
            conds[f"trans{k}"] = order[(src, m)][:k]
        conds["bottom10"] = order[(tgt, m)][-10:]
        for i, s in enumerate(RANDOM_SETS[:(20 if m == "XGB" else 10)]):
            conds[f"rand{i}"] = list(s)
        for name, feats in conds.items():
            key = (tgt, m, name)
            if key in res:
                continue
            mdl = make_model(m).fit(Xtr[feats], ytr)
            p = mdl.predict(Xte[feats])
            res[key] = dict(feats=feats, f1=f1_score(yte, p, average="macro"),
                            acc=accuracy_score(yte, p), pred=p)
            print(tgt, m, name, f"{res[key]['f1']:.4f}", flush=True)
        pickle.dump(res, open(out_path, "wb"))
print("done")
