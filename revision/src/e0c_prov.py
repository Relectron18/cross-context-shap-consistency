import pandas as pd, numpy as np, io
from common import *
import sys
A = sys.argv[1].rstrip("/") + "/"  # path to the "Scenario A2-ARFF" folder
for ctx, fn in (("novpn","TimeBasedFeatures-Dataset-15s-NO-VPN.arff"),("vpn","TimeBasedFeatures-Dataset-15s-VPN.arff")):
    lines = [l for l in open(A+fn) if l.strip() and not l.startswith(("@","%"))]
    df = pd.read_csv(io.StringIO("".join(lines)), header=None)
    df = df.iloc[:, :24]; df.columns = FEATURES+["cls"]
    df = df.dropna(subset=["cls"])
    df["cls"] = df["cls"].str.replace("VPN-","")
    print(ctx, "raw", len(df), "dedup full row", len(df.drop_duplicates()), "dedup feat", len(df.drop_duplicates(subset=FEATURES)))
    d = df.drop_duplicates()
    print(d.cls.value_counts().to_dict())
    Xtr,Xte,ytr,yte = load(ctx)
    both = pd.concat([Xtr,Xte]).round(6)
    m = d[FEATURES].round(6)
    key = lambda X: set(map(tuple, X.to_numpy()))
    print("processed rows", len(both), "unique processed vectors in arff:", len(key(both) & key(m)), "of", len(key(both)))
    print("sentinel -1 present", (df[FEATURES]==-1).any().sum())
