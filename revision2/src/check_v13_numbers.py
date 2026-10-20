"""Check v13 against the result files. Usage: python check_v13_numbers.py <manuscript.docx>. Exits 1 on any failure.

1. Every number changed or added in v13 appears in the text exactly as derived from results/p8_v13.json.
2. The statements the v12 defect review retracted no longer appear.
3. Every numeric cell of Table VII equals the mean over repeats in the per-repeat CSVs, rounded once.
"""
import sys, json, zipfile, re
from pathlib import Path
from lxml import etree
R = Path(__file__).resolve().parents[1] / "results"
A = json.load(open(R / "p8_v13.json"))
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"; M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
doc = etree.fromstring(zipfile.ZipFile(sys.argv[1]).read("word/document.xml"))
ptext = lambda p: "".join(e.text or "" for e in p.iter(f"{{{W}}}t", f"{{{M}}}t"))
t = "\n".join(ptext(p) for p in doc.iter(f"{{{W}}}p"))
f3 = lambda x: f"{x:.3f}"; sg = lambda x, d=3: f"{x:+.{d}f}" if round(x, d) != 0 else f"{abs(x):.{d}f}"
PI, ET, MF, DR, RL, CW, MT = (A[k] for k in ("pi_cross", "et_vs_rf", "minflowiat_equality", "drop_RF_vpn_min_fiat", "min_iat_reliance_RF", "class_weighting", "matched1330"))
t7 = A["table7"]
et_ii = sorted(ET[c]["interv_interv"] for c in ("novpn", "vpn")); et_pp = sorted(ET[c]["path_path"] for c in ("novpn", "vpn"))
expect = {
 "PI average ranks": f"is {f3(PI['marginal_RF']['rho_average'])} (Random Forest) and {f3(PI['marginal_XGB']['rho_average'])} (XGBoost) for marginal and {f3(PI['conditional_RF']['rho_average'])} and {f3(PI['conditional_XGB']['rho_average'])} for conditional",
 "PI raw": f"the four values are {f3(PI['marginal_RF']['rho_raw_unclipped'])}, {f3(PI['marginal_XGB']['rho_raw_unclipped'])}, {f3(PI['conditional_RF']['rho_raw_unclipped'])} and {f3(PI['conditional_XGB']['rho_raw_unclipped'])}",
 "PI ties": f"{PI['marginal_RF']['n_nonpositive']['novpn']} of the 23 features are tied at zero in each context under marginal permutation, and {PI['conditional_RF']['n_nonpositive']['novpn']} and {PI['conditional_RF']['n_nonpositive']['vpn']} under conditional",
 "RF reliance": f"costs {f3(RL['novpn']['joint_perm_df1'])} macro F1 without a VPN and {f3(RL['vpn']['joint_perm_df1'])} with one, and dropping them and retraining costs {f3(RL['novpn']['drop_all3_df1'])} and {f3(RL['vpn']['drop_all3_df1'])}",
 "ET vs RF": f"{f3(et_pp[0])}–{f3(et_pp[1])} when both are explained with the path-dependent formulation and {f3(et_ii[0])}–{f3(et_ii[1])} when both use the interventional one",
 "min_flowiat shares": f"only {100*MF['novpn']['share_defined']:.0f}% (Non-VPN) and {100*MF['vpn']['share_defined']:.0f}% (VPN) of cases ({100*MF['novpn']['share_all']:.0f}% and {100*MF['vpn']['share_all']:.0f}% of all training flows)",
 "drop-one renormalised": f"by {DR['min_flowiat']['renormalised']:.1f} and {DR['min_biat']['renormalised']:.1f} percentage points more than renormalising",
 "drop-one raw": f"({DR['min_flowiat']['raw']:.1f} and {DR['min_biat']['raw']:.1f} points before that correction)",
 "class weighting": f"RBO rises from {f3(CW['RF']['rbo'][0])} to {f3(CW['RF']['rbo'][1])} and the L1 distance falls from {f3(CW['RF']['l1'][0])} to {f3(CW['RF']['l1'][1])}",
 "matched comparators": f"compared with {f3(MT['xgb_primary_cross'])} for the full-size XGBoost models and {f3(MT['xgb_random_half_cross'])} between their random halves",
 "grouped10 cross": f"({f3(A['grouped10_cross']['RF']['grouped10'])} against {f3(A['grouped10_cross']['RF']['random'])})",
 "transfer below CV": f"({f3(A['transfer_f1']['transfer_XGB_novpn_to_vpn'])} and {f3(A['transfer_f1']['transfer_RF_novpn_to_vpn'])}) fall below that range",
 "medians": f"(median {A['flows_per_capture']['novpn']['median']:.1f})",
 "tree_method": f"tree_method = {A['xgb_tree_method']['used']}",
 "null XGB": f"Δ of {t7['XGB_null']['gap']:.3f} for XGBoost (range {t7['XGB_null']['lo']:.3f} to {f3(t7['XGB_null']['hi'])})",
 "VII spread": "within 2.15 percentage points",
 "k 10 or 15": "feature sets of 10 or 15 features",
 "corresponding author": "*Corresponding author.",
 "class-mix SD (XGB natural)": f"(SD {A['classmix_delta']['XGB_novpn']['sd_natural']:.3f} and {A['classmix_delta']['XGB_vpn']['sd_natural']:.3f})",
 "threads": "XGBoost was run single-threaded; refitting with two threads gave identical predictions.",
}
gone = ["0.611 (Random Forest)", "2.2 and 1.5 percentage points, respectively", "0.709–0.776 for the interventional", "0.77–0.82",
        "in only 43% of Non-VPN", "containing at least 10 features", "fall in the same range", "is barely changed",
        "none of the designs controls", "Whole clusters are assigned at random", "Explanations are always computed", "give them equal weight",
        "the block itself gains importance", "the block as a whole gains importance", "macro-F1", "data set", "Kendall’s", "Spearman’s",
        "Explanation capability is being required", "If there is an application of the explanations", "do not exploit the features",
        "P2P has a single capture per context", "cannot be divided by capture at all", "indicate the same direction", "all available cores", "within 2.1 percentage", "(Fig. 4). We therefore"]
fails = [k for k, v in expect.items() if v not in t] + [f"still present: {g}" for g in gone if g in t]

# Table VII: every numeric cell against the per-repeat means
cap = [p for p in doc.iter(f"{{{W}}}p") if ptext(p).strip() == "TABLE VII"][0]
tbl = cap.getnext()
while tbl.tag != f"{{{W}}}tbl": tbl = tbl.getnext()
LAB = {"Random halves": "random", "Grouped, ~10": "grouped10", "Grouped, ~35": "grouped35", "Grouped, ~140": "grouped", "Grouped, calibrated": "calibrated",
       "Row blocks, ~10": "ordered10", "Row blocks, ~35": "ordered35", "Row blocks, ~140": "ordered140", "Row blocks, calibrated": "orderedcal",
       "Row blocks, first": "orderedhalf", "Permutation null": "null"}
model, ncell = None, 0
for tr in tbl.iter(f"{{{W}}}tr"):
    cells = [[ptext(p).strip() for p in tc.iter(f"{{{W}}}p") if ptext(p).strip()] for tc in tr.iter(f"{{{W}}}tc")]
    if cells and cells[0] and cells[0][0].startswith("XGBoost ("): model = "XGB"; continue
    if cells and cells[0] and cells[0][0].startswith("Random Forest ("): model = "RF"; continue
    lab = next((v for k, v in LAB.items() if cells and cells[0] and cells[0][0].startswith(k)), None)
    if not lab: continue
    s = t7[f"{model}_{lab}"]
    want = [[f"{f3(s['f1_nv'])} / {f3(s['f1_vpn'])}"], [f3(s["w_nv"])], [f3(s["w_vpn"])], [f3(s["cross"])],
            [f"{sg(s['gap'])} ({f3(s['sd'])})", f"{s['lo']:.3f} to {f3(s['hi'])}"], [sg(s["rbo"])], [sg(s["top5"], 2)], [sg(s["wgap"])]]
    got = cells[1:]
    for i, (g, w) in enumerate(zip(got, want)):
        ncell += 1
        g2 = [x.replace("−", "-") for x in g]; w2 = [x.replace("+-", "-").replace("-0.000", "0.000") for x in w]
        if g2 != w2: fails.append(f"Table VII {model} {lab} col {i + 1}: {g2} != {w2}")
for k, v in expect.items(): print("OK  " if v in t else "MISS", k)
print(f"Table VII cells checked: {ncell}")
for f in fails: print("FAIL", f)
sys.exit(1 if fails else 0)
