"""Class counts in the original Scenario A2 ARFF files (Concern 6: BROWSING cap).
Usage: python p0_arff_counts.py "<path to Scenario A2-ARFF>"  -> results/arff_class_counts.csv
Counts in the committed CSV were produced from the files distributed by CIC/UNB."""
import sys, io, glob, os
import pandas as pd
rows = []
for fn in sorted(glob.glob(os.path.join(sys.argv[1], "TimeBasedFeatures-Dataset-*.arff"))):
    lines = [l for l in open(fn) if l.strip() and not l.startswith(("@", "%"))]
    df = pd.read_csv(io.StringIO("".join(lines)), header=None)
    cls = df.iloc[:, -1].astype(str).str.replace("VPN-", "", regex=False)
    r = cls.value_counts().to_dict(); r["file"] = os.path.basename(fn); r["total"] = len(df)
    r["exact_duplicate_surplus_rows"] = int(df.duplicated().sum()); rows.append(r)
out = os.path.join(os.path.dirname(__file__), "..", "results", "arff_class_counts.csv")
pd.DataFrame(rows)[["file", "BROWSING", "CHAT", "FT", "MAIL", "P2P", "STREAMING", "VOIP", "total", "exact_duplicate_surplus_rows"]].to_csv(out, index=False)
print(open(out).read())
