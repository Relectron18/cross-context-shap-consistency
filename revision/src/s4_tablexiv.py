import joblib, pickle
from sklearn.model_selection import train_test_split
from common import *
P = pickle.load(open(OUT/"primary.pkl","rb"))
mats = {}
for ctx in CTX:
    Xtr, Xte, ytr, yte = load(ctx)
    Xs, _, ys, _ = train_test_split(Xte, yte, train_size=500, stratify=yte, random_state=42)
    for m in ("XGB","RF"):
        model = joblib.load(ROOT / "models" / ctx / "xgboost.joblib") if m=="XGB" else make_model("RF").fit(Xtr, ytr)
        sv = shap_values(model, Xs)
        mats[(ctx,m)] = np.abs(sv).mean(axis=0)
def rk(v): return pd.Series(v).rank(method="average", ascending=False).to_numpy()
out = {}
for m in ("RF","XGB"):
    a, b = mats[("novpn",m)], mats[("vpn",m)]
    point = spearmanr(rk(a.mean(0)), rk(b.mean(0)))[0]
    tau = kendalltau(rk(a.mean(0)), rk(b.mean(0)))[0]
    rng = np.random.default_rng(20261008); vals=[]
    for _ in range(2000):
        ia = rng.integers(0,len(a),len(a)); ib = rng.integers(0,len(b),len(b))
        vals.append(spearmanr(rk(a[ia].mean(0)), rk(b[ib].mean(0)))[0])
    vals = np.array(vals)
    lo, hi = np.percentile(vals,[2.5,97.5])
    print(m, f"point={point:.4f} tau={tau:.4f} pct=[{lo:.4f},{hi:.4f}] boot_mean={vals.mean():.4f} boot_median={np.median(vals):.4f} frac_below_point={np.mean(vals<point):.3f}",
          f"basic=[{2*point-hi:.4f},{2*point-lo:.4f}]")
    out[m] = dict(point=point, vals=vals)
pickle.dump((mats,out), open(OUT/"tablexiv_check.pkl","wb"))
