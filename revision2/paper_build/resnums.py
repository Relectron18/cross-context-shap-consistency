"""Load every result used in the paper text (all from revision2/results)."""
import json, numpy as np, pandas as pd, sys
from pathlib import Path
RES = Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
P1 = json.load(open(RES / "p1_results.json")); P2 = json.load(open(RES / "p2_results.json"))
P3 = json.load(open(RES / "p3_results.json")); P4 = json.load(open(RES / "p4_results.json"))
S = json.load(open(RES / "ref_summary.json"))
TK = pd.read_csv(RES / "p2_topk.csv")
CM = pd.read_csv(RES / "p4_classmix.csv"); M13 = pd.read_csv(RES / "p4_matched1330.csv")
def f(x, d=3):
    s = f"{x:.{d}f}"
    if float(s) == 0: s = s.lstrip("-")
    return s.replace("-", "−")
def sg(x, d=3):
    if float(f"{abs(x):.{d}f}") == 0: return f"{0:.{d}f}"
    return ("+" if x >= 0 else "−") + f"{abs(x):.{d}f}"
def pct(x, d=1): return f"{100*x:.{d}f}"
def ref(m, d, key, stat="mean"): return S[f"{m}_{d}"][stat][key]
from scipy.stats import spearmanr
FEATURES = ["duration", "total_fiat", "total_biat", "min_fiat", "min_biat", "max_fiat", "max_biat", "mean_fiat", "mean_biat",
            "flowPktsPerSecond", "flowBytesPerSecond", "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat",
            "min_active", "mean_active", "max_active", "std_active", "min_idle", "mean_idle", "max_idle", "std_idle"]
FAMILIES = {"Flow duration": ["duration"], "Directional IAT": ["total_fiat", "total_biat", "min_fiat", "min_biat", "max_fiat", "max_biat", "mean_fiat", "mean_biat"],
            "Flow throughput": ["flowPktsPerSecond", "flowBytesPerSecond"], "Flow IAT": ["min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat"],
            "Active period": ["min_active", "mean_active", "max_active", "std_active"], "Idle period": ["min_idle", "mean_idle", "max_idle", "std_idle"]}
ALT = {"Duration and totals": ["duration", "total_fiat", "total_biat"], "Directional IAT": ["min_fiat", "min_biat", "max_fiat", "max_biat", "mean_fiat", "mean_biat"],
       **{k: v for k, v in FAMILIES.items() if k not in ("Flow duration", "Directional IAT")}}
CLUST = json.load(open(RES.parent.parent / "revision" / "results" / "clusters_0.2.json"))
def prim(m, c):
    return pd.Series(np.load(RES / f"primary_rec_{m}_{c}.npy").mean(0), index=FEATURES)
def fam(imp, fams, mean=False):
    return pd.Series({k: (imp[v].mean() if mean else imp[v].sum()) for k, v in fams.items()})
def fam_rho(m, fams, mean=False):
    return spearmanr(fam(prim(m, "novpn"), fams, mean), fam(prim(m, "vpn"), fams, mean))[0]
