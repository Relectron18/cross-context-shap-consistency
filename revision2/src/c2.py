"""Shared helpers for the second revision (single environment for every result)."""
import sys, json, time, warnings, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "revision" / "src"))
from common import (FEATURES, FAMILIES, CTX, load, make_model, shap_values, importance,
                    ranks, rbo_ext, compare, family_imp, family_mean_imp, tic)
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score, recall_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
OUT2 = ROOT / "revision2" / "results"; OUT2.mkdir(parents=True, exist_ok=True)
FIG2 = ROOT / "revision2" / "figures"; FIG2.mkdir(parents=True, exist_ok=True)
M = len(FEATURES)

def logx2(X):
    """log(x + 2) with values clipped at -1 (sentinel -> log 1 = 0)."""
    return np.log(np.asarray(X, float).clip(min=-1) + 2)

def pool(ctx):
    """All deduplicated flows of a context (primary train + test)."""
    Xtr, Xte, ytr, yte = load(ctx)
    return pd.concat([Xtr, Xte], ignore_index=True), np.concatenate([ytr, yte])

def kappa(o, k, m=M):
    e = k * k / m
    return (o - e) / (k - e)

def cmp_full(ia, ib):
    d = compare(ia, ib)
    d["kappa5"] = kappa(d["top5"], 5); d["kappa10"] = kappa(d["top10"], 10)
    d["fam_rho"] = spearmanr(family_imp(ia), family_imp(ib))[0]
    return d

def strat_sub(ids, y, n, seed):
    if len(ids) <= n: return np.asarray(ids)
    keep, _ = next(StratifiedShuffleSplit(1, train_size=n, random_state=int(seed)).split(ids, y[ids]))
    return np.asarray(ids)[keep]

def jdump(obj, name):
    json.dump(obj, open(OUT2 / name, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))

_make_model = make_model
def make_model(name, seed=42, **kw):
    """XGBoost is run single-threaded: multi-threaded fits under CPU contention were ~10x slower.
    Thread count does not change XGBoost's hist results (checked: identical predictions)."""
    if name == "XGB": kw.setdefault("n_jobs", 1)
    return _make_model(name, seed=seed, **kw)
