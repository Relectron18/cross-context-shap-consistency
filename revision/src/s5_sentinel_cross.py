"""Cross-context agreement under the S1/S2 sentinel encodings (from saved sentinel rankings)."""
from common import *
d = pd.read_csv(ROOT / "results" / "sentinel" / "sentinel_rankings.csv")
d = d[d.feature_type == "original"]
for m in ("RF", "XGB"):
    for c in ("S1", "S2"):
        a = d[(d.context == "novpn") & (d.model == m) & (d.condition == c)].set_index("feature").mean_abs_shap
        b = d[(d.context == "vpn") & (d.model == m) & (d.condition == c)].set_index("feature").mean_abs_shap
        print(m, c, {k: round(float(v), 4) for k, v in compare(a, b).items()})
