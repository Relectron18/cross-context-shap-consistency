"""Extra Trees robustness per context (run the two contexts in parallel), saving per-record |SHAP|."""
import shap
from c2 import *
c = sys.argv[1]
Xtr, Xte, ytr, yte = load(c)
ex_ids = strat_sub(np.arange(len(yte)), yte, 500, 42); bg_ids = strat_sub(np.arange(len(ytr)), ytr, 200, 42)
out = {}
for mname in (["ET", "RF"] if sys.argv[2:] == [] else sys.argv[2:]):
    m = make_model(mname).fit(Xtr, ytr)
    perf = dict(f1=f1_score(yte, m.predict(Xte), average="macro"), acc=accuracy_score(yte, m.predict(Xte)))
    out[f"{mname}_path"] = np.abs(shap_values(m, Xte.iloc[ex_ids])).mean(axis=0)
    if mname == "ET":
        e = shap.TreeExplainer(m, data=shap.maskers.Independent(Xtr.iloc[bg_ids], max_samples=200), feature_perturbation="interventional", model_output="raw")
        sv = np.asarray(e.shap_values(Xte.iloc[ex_ids], check_additivity=False))
        if sv.ndim == 3 and sv.shape[-1] == 7: sv = np.transpose(sv, (2, 0, 1))
        out["ET_interv"] = np.abs(sv).mean(axis=0)
        json.dump(perf, open(OUT2 / f"et_perf_{c}.json", "w"))
np.savez(OUT2 / f"et_rec_{c}.npz", **out); print("done", c)
