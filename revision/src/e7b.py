import sys, pickle
from sklearn.metrics import f1_score, accuracy_score
from common import *
exec(open(Path(__file__).with_name("e7_consequence.py")).read().split("out_path =")[0])  # reuse order, RANDOM_SETS
out_path = OUT / "consequence.pkl"
try: res = pickle.load(open(out_path, "rb"))
except FileNotFoundError: res = {}
data = {c: load(c) for c in CTX}
plan = [("vpn", "XGB", 6), ("novpn", "RF", 3), ("vpn", "RF", 3)]
for tgt, m, nr in plan:
    src = "vpn" if tgt == "novpn" else "novpn"
    Xtr, Xte, ytr, yte = data[tgt]
    conds = {"own10": order[(tgt, m)][:10], "trans10": order[(src, m)][:10], "bottom10": order[(tgt, m)][-10:]}
    if m == "XGB": conds.update(own5=order[(tgt, m)][:5], trans5=order[(src, m)][:5])
    for i, s in enumerate(RANDOM_SETS[:nr]): conds[f"rand{i}"] = list(s)
    for name, feats in conds.items():
        if (tgt, m, name) in res: continue
        mdl = make_model(m).fit(Xtr[feats], ytr); p = mdl.predict(Xte[feats])
        res[(tgt, m, name)] = dict(feats=feats, f1=f1_score(yte, p, average="macro"), acc=accuracy_score(yte, p), pred=p)
        print(tgt, m, name, f"{res[(tgt, m, name)]['f1']:.4f}", flush=True)
        pickle.dump(res, open(out_path, "wb"))
print("done")
