"""Check v14 against the result files. Usage: python check_v14_numbers.py <manuscript.docx>. Exits 1 on any failure.

1. Every v13 check still passes (check_v13_numbers.py): the editorial pass must not have moved a checked number.
2. The new numbers (common explained records, grouped permutation null) appear as derived from results/p11_v14.json.
3. Every numeric cell of the 20 new Table VII rows (two null versions x five designs x two models) matches p11_v14.json.
4. Statements that the new analyses retired are gone, and the abstract stays within 250 words.
"""
import sys, json, zipfile, subprocess
from pathlib import Path
from lxml import etree
HERE = Path(__file__).resolve().parent
G = json.load(open(HERE.parent / "results" / "p11_v14.json"))
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
doc = etree.fromstring(zipfile.ZipFile(sys.argv[1]).read("word/document.xml"))
ptext = lambda p: "".join(e.text or "" for e in p.iter(f"{{{W}}}t", f"{{{M}}}t"))
paras = [ptext(p) for p in doc.iter(f"{{{W}}}p")]; t = "\n".join(paras)
f3 = lambda x: f"{x:.3f}"
def sg(x, d=3):
    s = f"{x:+.{d}f}"; return s[1:] if float(s) == 0 else s
fails = []
r = subprocess.run([sys.executable, str(HERE / "check_v13_numbers.py"), sys.argv[1]], capture_output=True, text=True)
if r.returncode: fails.append("v13 checks fail:\n" + "\n".join(l for l in r.stdout.splitlines() if l.startswith(("MISS", "FAIL"))))

CR, GN = G["common_records"], G["grouped_null"]
rng = lambda m, k="gap": (min(CR[m][k][s] for s in ("nv", "vpn", "union")), max(CR[m][k][s] for s in ("nv", "vpn", "union")))
own = CR["RF"]["gap"]["own"]; rg = rng("RF")
pct = f"about {100 * (own - rg[1]) / own:.0f}–{100 * (own - rg[0]) / own:.0f}%"
prim = {m: sorted(CR[m]["primary"]["on_common"][s]["rho"] for s in ("nv", "vpn", "union")) for m in ("RF", "XGB")}
NG, NS = GN["nullgroup"], GN["nullstrat"]
ROWD, CLUD = ("orderedcal", "orderedhalf"), ("grouped10", "grouped35", "calibrated")
row_nulls = [V[f"{m}_{d}"]["null"]["rho_gap"] for V in (NG, NS) for m in ("XGB", "RF") for d in ROWD]
expect = {
 "common records gaps": f"Δ is {f3(rng('XGB')[0])}–{f3(rng('XGB')[1])} for XGBoost and {f3(rg[0])}–{f3(rg[1])} for Random Forest across the three common record sets",
 "common records own": f"against {f3(CR['XGB']['gap']['own'])} and {f3(own)} when each model is explained on its own context's records",
 "common records RBO": f"the RBO gap falls from {f3(CR['RF']['rbo_gap']['own'])} to {f3(rng('RF', 'rbo_gap')[0])}–{f3(rng('RF', 'rbo_gap')[1])}",
 "primary common": f"{f3(prim['RF'][0])}–{f3(prim['RF'][2])} for Random Forest and {f3(prim['XGB'][0])}–{f3(prim['XGB'][2])} for XGBoost, against 0.830 and 0.786",
 "row-block null range": f"({min(row_nulls):.3f} to {max(row_nulls):.3f})",
 "row-block excess": f"by {f3(NG['XGB_orderedcal']['real_minus_null'])} and {f3(NG['XGB_orderedhalf']['real_minus_null'])} for XGBoost and by {f3(NG['RF_orderedcal']['real_minus_null'])} and {f3(NG['RF_orderedhalf']['real_minus_null'])} for Random Forest",
 "cluster flow null": f"the flow-level null gives {min(NS[f'{m}_{d}']['null']['rho_gap'] for m in ('XGB', 'RF') for d in CLUD):.3f} to {max(NS[f'{m}_{d}']['null']['rho_gap'] for m in ('XGB', 'RF') for d in CLUD):.3f}",
}
for k, v in expect.items():
    if v not in t: fails.append(f"MISS {k}: {v}")
if t.count(pct) < 4: fails.append(f"share of the RF gap from explained records ({pct}) appears {t.count(pct)} times, expected 4 (VI-C, X-A, X-E, XII)")
for g in ("the permutation null was run only for random halves", "is left for future work", "no design tests that remainder directly",
          "the permutation null covers only random halves", "none of the designs controls"):
    if g in t: fails.append(f"retired statement still present: {g}")
ab = [p for p in paras if p.startswith("Abstract")][0]
if len(ab.split()) > 250: fails.append(f"abstract has {len(ab.split())} words")

# Table VII: new rows
cap = [p for p in doc.iter(f"{{{W}}}p") if ptext(p).strip() == "TABLE VII"][0]
tbl = cap.getnext()
while tbl.tag != f"{{{W}}}tbl": tbl = tbl.getnext()
LAB = {"grouped ~10 per cluster": "grouped10", "grouped ~35 per cluster": "grouped35", "grouped calibrated": "calibrated",
       "row blocks calibrated": "orderedcal", "row blocks first vs second half": "orderedhalf"}
model, n = None, 0
for tr in tbl.iter(f"{{{W}}}tr"):
    cells = [[ptext(p).strip() for p in tc.iter(f"{{{W}}}p") if ptext(p).strip()] for tc in tr.iter(f"{{{W}}}tc")]
    first = cells[0][0] if cells and cells[0] else ""
    if first.startswith("XGBoost ("): model = "XGB"; continue
    if first.startswith("Random Forest ("): model = "RF"; continue
    if not first.startswith("Null ("): continue
    ver = "nullstrat" if first.startswith("Null (flows), ") else "nullgroup"
    d = LAB[first.split(", ", 1)[1]]; g = GN[ver][f"{model}_{d}"]; s = g["null"]
    want = [[f"{f3(s['f1_nv'])} / {f3(s['f1_vpn'])}"], [f3(s["rho_wnv"])], [f3(s["rho_wvpn"])], [f3(s["rho_cross"])],
            [f"{sg(s['rho_gap'])} ({f3(g['null_gap_sd'])})", f"{g['null_gap_min']:.3f} to {g['null_gap_max']:.3f}"],
            [sg(s["rbo_gap"])], [sg(s["top5_gap"], 2)], [sg(s["rho_w_gap"])]]
    n += 1
    if cells[1:] != want: fails.append(f"Table VII {first} ({model}): {cells[1:]} != {want}")
if n != 20: fails.append(f"expected 20 new Table VII rows, found {n}")
print(f"new Table VII rows checked: {n}")
for f in fails: print("FAIL", f)
print("OK" if not fails else f"{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
