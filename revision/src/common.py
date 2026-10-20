import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr, kendalltau
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]   # repository root
DATA = ROOT / "data"
OUT = ROOT / "revision" / "results"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "duration", "total_fiat", "total_biat", "min_fiat", "min_biat",
    "max_fiat", "max_biat", "mean_fiat", "mean_biat",
    "flowPktsPerSecond", "flowBytesPerSecond", "min_flowiat",
    "max_flowiat", "mean_flowiat", "std_flowiat",
    "min_active", "mean_active", "max_active", "std_active",
    "min_idle", "mean_idle", "max_idle", "std_idle",
]
FAMILIES = {
    "Flow duration": ["duration"],
    "Directional IAT": ["total_fiat", "total_biat", "min_fiat", "min_biat",
                        "max_fiat", "max_biat", "mean_fiat", "mean_biat"],
    "Flow throughput": ["flowPktsPerSecond", "flowBytesPerSecond"],
    "Flow IAT": ["min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat"],
    "Active period": ["min_active", "mean_active", "max_active", "std_active"],
    "Idle period": ["min_idle", "mean_idle", "max_idle", "std_idle"],
}
CTX = ("novpn", "vpn")


def load(ctx, controlled=False):
    d = DATA / "controlled" / ctx if controlled else DATA / ctx
    Xtr = pd.read_csv(d / "X_train.csv")[FEATURES]
    Xte = pd.read_csv(d / "X_test.csv")[FEATURES]
    ytr = pd.read_csv(d / "y_train.csv").iloc[:, 0].to_numpy()
    yte = pd.read_csv(d / "y_test.csv").iloc[:, 0].to_numpy()
    return Xtr, Xte, ytr, yte


def make_model(name, seed=42, **kw):
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1, **kw)
    if name == "XGB":
        params = dict(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                      colsample_bytree=0.8, objective="multi:softprob", num_class=7,
                      eval_metric="mlogloss", random_state=seed, n_jobs=-1)
        params.update(kw)
        return XGBClassifier(**params)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=500, random_state=seed, n_jobs=-1, **kw)
    raise ValueError(name)


def shap_values(model, X, n_classes=7):
    ex = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    sv = np.asarray(ex.shap_values(X, check_additivity=False))
    if sv.ndim == 3 and sv.shape[0] == len(X) and sv.shape[-1] == n_classes:
        sv = np.transpose(sv, (2, 0, 1))
    return sv  # (C, N, F)


def importance(sv, cols=FEATURES, y=None):
    """Mean |SHAP| over records and classes; if y given, weight true classes equally."""
    a = np.abs(sv).mean(axis=0)  # (N, F)
    if y is None:
        imp = a.mean(axis=0)
    else:
        imp = np.mean([a[y == c].mean(axis=0) for c in np.unique(y)], axis=0)
    return pd.Series(imp, index=cols)


def ranks(imp):
    # Tied values get their average rank (v13). Up to v12 this used method="first", which breaks ties by
    # feature order. SHAP importances have no ties, so SHAP results are unchanged (checked in
    # revision2/src/p8_v13_audit.py); permutation importance clipped at 0 has many ties, and its
    # cross-context agreement was recomputed in p8_v13_audit.py.
    return imp.rank(ascending=False, method="average")


def rbo_ext(list_a, list_b, p=0.9):
    """Extrapolated rank-biased overlap (Webber et al. 2010, eq. 32) for equal-length lists."""
    k = len(list_a)
    sa, sb = set(), set()
    s = 0.0
    x_d = 0
    for d in range(1, k + 1):
        sa.add(list_a[d - 1]); sb.add(list_b[d - 1])
        x_d = len(sa & sb)
        s += (x_d / d) * p ** d
    return x_d / k * p ** k + (1 - p) / p * s


def compare(imp_a, imp_b, p=0.9):
    ra, rb = ranks(imp_a)[FEATURES], ranks(imp_b)[FEATURES]
    la = list(imp_a.sort_values(ascending=False).index)
    lb = list(imp_b.sort_values(ascending=False).index)
    sa, sb = imp_a[FEATURES] / imp_a.sum(), imp_b[FEATURES] / imp_b.sum()
    out = {
        "rho": spearmanr(ra, rb)[0],
        "tau": kendalltau(ra, rb)[0],
        "top5": len(set(la[:5]) & set(lb[:5])),
        "top10": len(set(la[:10]) & set(lb[:10])),
        "rbo": rbo_ext(la, lb, p),
        "l1": float(np.abs(sa - sb).sum()),  # = 2 x total variation distance of share vectors
    }
    return out


def family_imp(imp, families=FAMILIES):
    return pd.Series({f: imp[m].sum() for f, m in families.items()})


def family_mean_imp(imp, families=FAMILIES):
    return pd.Series({f: imp[m].mean() for f, m in families.items()})


def tic():
    return time.time()
