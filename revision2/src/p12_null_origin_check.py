"""v14: origin balance of the grouped-null halves (repeat 0, XGBoost settings; the halves do not depend on the model).

In the permutation null each pseudo-context mixes flows from the Non-VPN and the VPN file. If a grouped design puts
mostly Non-VPN-origin flows of a class in one half and VPN-origin flows in the other, its "within" pair is partly a
cross-context pair. This script rebuilds the halves of repeat 0 for each grouped null and reports, per class, the
absolute difference between the two halves in their share of VPN-origin flows. null_<design> (first version) is
confounded this way; nullstrat_<design> groups within each origin and is balanced.
Writes results/p12_null_origin_check.json.
"""
import json, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
CODE = r'''
import sys, numpy as np
sys.argv = ["designs.py", "XGB", DESIGN_NAME, "0"]
exec(open("designs.py").read().split("for r in range(R):")[0])
r = 0; D = contexts_for_repeat(r); out = {}
P_orig = {c: pool(c) for c in CTX}
for ci, c in enumerate(CTX):
    X, y = D[c]
    rng = np.random.default_rng(31000 + r); origin = []
    for k in range(7):
        n_src = [int((P_orig[cc][1] == k).sum()) for cc in CTX]
        perm = rng.permutation(sum(n_src)); ids = perm[:n_src[0]] if c == "novpn" else perm[n_src[0]:]
        origin.append((ids >= n_src[0]).astype(int))
    origin = np.concatenate(origin)
    itr, ite = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=1000 + r)
    Xtr, ytr = X.iloc[itr].reset_index(drop=True), y[itr]
    a, b = grouped_halves(Xtr, ytr, 41000 + 100 * r + ci, ctx=c, pos=POSR[c][itr] if ORDERED else None,
                          origin=ORIGINR[c][itr] if STRAT else None)
    o = origin[itr]; d = [abs(o[a][ytr[a] == k].mean() - o[b][ytr[b] == k].mean()) for k in range(7)]
    out[c] = dict(max=float(max(d)), mean=float(np.mean(d)))
print("RESULT", json.dumps(out))
'''
res = {}
for prefix in ("null_", "nullstrat_"):
    for d in ("grouped10", "grouped35", "calibrated", "orderedcal", "orderedhalf"):
        name = prefix + d
        p = subprocess.run([sys.executable, "-c", "import json\nDESIGN_NAME=%r\n" % name + CODE], cwd=HERE, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        assert line, p.stderr[-2000:]
        res[name] = json.loads(line[0][7:]); print(name, res[name], flush=True)
json.dump(res, open(HERE.parent / "results" / "p12_null_origin_check.json", "w"), indent=1)
