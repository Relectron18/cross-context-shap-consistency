"""Reproduce primary models, save SHAP values; baseline metrics."""
import pickle
from sklearn.metrics import accuracy_score, f1_score
from common import *

res = {}
for ctx in CTX:
    Xtr, Xte, ytr, yte = load(ctx)
    for m in ("RF", "XGB"):
        t = tic()
        model = make_model(m).fit(Xtr, ytr)
        pred = model.predict(Xte)
        sv = shap_values(model, Xte)
        imp = importance(sv)
        res[(ctx, m)] = dict(sv=sv, imp=imp, acc=accuracy_score(yte, pred),
                             f1=f1_score(yte, pred, average="macro"), pred=pred)
        print(ctx, m, f"acc={res[(ctx,m)]['acc']:.4f} f1={res[(ctx,m)]['f1']:.4f} t={tic()-t:.0f}s")
        print((imp / imp.sum() * 100).sort_values(ascending=False).head(6).round(2).to_dict())
pickle.dump(res, open(OUT / "primary.pkl", "wb"))
for m in ("RF", "XGB"):
    print(m, compare(res[("novpn", m)]["imp"], res[("vpn", m)]["imp"]))
