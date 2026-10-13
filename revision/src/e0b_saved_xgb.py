import pickle, joblib
from sklearn.metrics import accuracy_score, f1_score
from common import *
res = pickle.load(open(OUT/"primary.pkl","rb"))
for ctx in CTX:
    Xtr, Xte, ytr, yte = load(ctx)
    model = joblib.load(ROOT / "models" / ctx / "xgboost.joblib")
    pred = model.predict(Xte)
    sv = shap_values(model, Xte); imp = importance(sv)
    res[(ctx,"XGB")] = dict(sv=sv, imp=imp, acc=accuracy_score(yte,pred), f1=f1_score(yte,pred,average="macro"), pred=pred)
    print(ctx, res[(ctx,"XGB")]["acc"], res[(ctx,"XGB")]["f1"])
    print((imp/imp.sum()*100).sort_values(ascending=False).head(6).round(2).to_dict())
    lr = joblib.load(f"models/lr_{ctx}.joblib"); print("LR n_iter", lr.n_iter_, lr.max_iter)
pickle.dump(res, open(OUT/"primary.pkl","wb"))
for m in ("RF","XGB"): print(m, compare(res[("novpn",m)]["imp"], res[("vpn",m)]["imp"]))
