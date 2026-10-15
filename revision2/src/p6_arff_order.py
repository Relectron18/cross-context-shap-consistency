"""Row order of the original ARFF files as a capture proxy (Concern 1, round 4).

The processed CSVs used everywhere else were shuffled by the train/test split, but the original
Scenario A2 ARFF files keep the order in which ISCXFlowMeter wrote the flows. Rows are sorted by class,
and within a class neighbouring rows turn out to be far more alike than random pairs, so row order carries
capture locality that does not depend on feature-space clustering.

Usage: python p6_arff_order.py <folder with TimeBasedFeatures-Dataset-15s-NO-VPN.arff and -VPN.arff>
Outputs: results/arff_order_positions.json (ARFF row index of every pooled flow, in pool() order)
         results/arff_order_locality.csv  (neighbour-distance ratios per class and lag)
"""
from c2 import *
import io, os

FILES = {"novpn": "TimeBasedFeatures-Dataset-15s-NO-VPN.arff", "vpn": "TimeBasedFeatures-Dataset-15s-VPN.arff"}
LAGS = (1, 2, 5, 10, 20, 50, 100, 200)


def read_arff(fn):
    head = [l for l in open(fn) if l.startswith("@ATTRIBUTE")]
    cols = [l.split()[1] for l in head]
    rows = [l.strip().rstrip(",") for l in open(fn)
            if l.strip(", \n") and not l.startswith(("@", "%"))]
    df = pd.read_csv(io.StringIO("\n".join(rows)), header=None)
    df.columns = cols
    df = df.rename(columns={cols[-1]: "cls"})
    df["cls"] = df["cls"].astype(str).str.replace("VPN-", "", regex=False)
    return df


def key(M):
    return pd.util.hash_pandas_object(pd.DataFrame(np.round(np.asarray(M, float), 3)), index=False).values


if __name__ == "__main__":
    folder = sys.argv[1]
    lab = json.load(open(ROOT / "data" / "label_classes.json"))
    names = {v: k for k, v in lab.items()}
    positions, loc = {}, []
    for c in CTX:
        a = read_arff(os.path.join(folder, FILES[c]))
        X, y = pool(c)
        first = {}
        for i, (k, cl) in enumerate(zip(key(a[FEATURES].values), a.cls)):
            first.setdefault((k, cl), i)
        pos = np.array([first.get((k, names[int(t)]), -1) for k, t in zip(key(X[FEATURES].values), y)])
        assert (pos >= 0).all() and len(set(pos)) == len(pos), "every pooled flow must map to one ARFF row"
        positions[c] = pos.tolist()
        runs = int((a.cls != a.cls.shift()).sum())
        print(c, len(a), "ARFF rows;", len(X), "pooled flows mapped one-to-one;", runs, "class runs")
        # locality: median distance between rows `lag` apart within a class, relative to random pairs
        Z = logx2(a[FEATURES].values)
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
        rng = np.random.default_rng(0)
        for cl, g in a.groupby("cls", sort=False):
            z = Z[g.index.values]
            p = rng.integers(0, len(z), (20000, 2)); p = p[p[:, 0] != p[:, 1]]
            rnd = np.median(np.linalg.norm(z[p[:, 0]] - z[p[:, 1]], axis=1))
            for L in LAGS:
                if L < len(z):
                    loc.append(dict(ctx=c, cls=cl, n=len(z), lag=L,
                                    ratio=float(np.median(np.linalg.norm(z[L:] - z[:-L], axis=1)) / rnd)))
            # window statistic, robust to interleaved flows: distance to the nearest of the 40 rows within
            # +-20 positions, relative to the nearest of 40 random rows of the same class
            W = 20; near, ran = [], []
            for i in range(len(z)):
                nb = [j for j in range(max(0, i - W), min(len(z), i + W + 1)) if j != i]
                near.append(np.linalg.norm(z[nb] - z[i], axis=1).min())
                rr = rng.choice(np.delete(np.arange(len(z)), i), size=min(len(nb), len(z) - 1), replace=False)
                ran.append(np.linalg.norm(z[rr] - z[i], axis=1).min())
            loc.append(dict(ctx=c, cls=cl, n=len(z), lag="win20", ratio=float(np.median(near) / np.median(ran))))
    json.dump(positions, open(OUT2 / "arff_order_positions.json", "w"))
    L = pd.DataFrame(loc); L.to_csv(OUT2 / "arff_order_locality.csv", index=False)
    print(L.pivot_table(index=["ctx", "cls"], columns="lag", values="ratio").round(2).to_string())
