"""v13: interventional TreeSHAP for the primary Random Forest on the same 500 explained records and 200
background records as the robustness check (p3_sentinel_robust.py / p3c_et.py), saved per record so that
Extra Trees and Random Forest can be compared under the same formulation. p3 did not save these arrays.
Usage: python p8a_rf_interv.py <novpn|vpn>
"""
import shap
from c2 import *
c = sys.argv[1]
Xtr, Xte, ytr, yte = load(c)
ex_ids = strat_sub(np.arange(len(yte)), yte, 500, 42); bg_ids = strat_sub(np.arange(len(ytr)), ytr, 200, 42)
m = make_model("RF").fit(Xtr, ytr)
t = tic()
e = shap.TreeExplainer(m, data=shap.maskers.Independent(Xtr.iloc[bg_ids], max_samples=200),
                       feature_perturbation="interventional", model_output="raw")
sv = np.asarray(e.shap_values(Xte.iloc[ex_ids], check_additivity=False))
if sv.ndim == 3 and sv.shape[-1] == 7: sv = np.transpose(sv, (2, 0, 1))
np.savez(OUT2 / f"rf_interv_rec_{c}.npz", RF_interv=np.abs(sv).mean(axis=0), RF_path=np.abs(shap_values(m, Xte.iloc[ex_ids])).mean(axis=0),
         ex_ids=ex_ids, bg_ids=bg_ids)
print("done", c, f"{tic() - t:.0f}s", flush=True)
