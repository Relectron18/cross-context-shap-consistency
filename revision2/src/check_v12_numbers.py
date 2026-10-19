"""Check that every number added in v12 appears in the manuscript text exactly as derived from the result files.
Usage: python check_v12_numbers.py <extracted manuscript text>. Exits 1 on any mismatch."""
import sys, json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
A = json.load(open(R / "p7_audit.json")); P3 = json.load(open(R / "p3_results.json")); D = json.load(open(R / "p7_distinct_lists.json"))
t = open(sys.argv[1]).read()
L = A["locality_dedup"]; f2 = lambda x: f"{x:.2f}"; f3 = lambda x: f"{x:.3f}"
weak = ["vpn_FT", "vpn_MAIL", "vpn_P2P", "vpn_VOIP"]
lag_w = [L[k]["lag1_dedup"] for k in weak]; lag_o = [v["lag1_dedup"] for k, v in L.items() if k not in weak]
win = [v["win20_dedup"] for v in L.values()]
r = lambda k: P3[k]["rho"] if isinstance(P3[k], dict) else P3[k]
oh = lambda m, k: A[f"orderedhalf_{m}"][k]["n_neg"]
fam = A["family_gaps"]
expect = {
 "row-order win20": f"{f2(min(win))}–{f2(max(win))} times the distance",
 "lag-1 weak classes": f"median distance ratio {f2(min(lag_w))}–{f2(max(lag_w))}, against {f2(min(lag_o))}–{f2(max(lag_o))}",
 "position-feature rho": f"up to {f2(max(v['max_abs_rho_position_feature'] for v in L.values()))})",
 "BROWSING+P2P share": f"hold {round(100*min(A['share_browsing_p2p_novpn'],A['share_browsing_p2p_vpn']))}–{round(100*max(A['share_browsing_p2p_novpn'],A['share_browsing_p2p_vpn']))}% of the training flows",
 "one-sided RF": f"by {f3(A['onesided_RF_calibrated']['nv'])}, whereas within-VPN agreement lies {f3(-A['onesided_RF_calibrated']['vpn'])} below",
 "one-sided XGB": f"(XGBoost: +{f3(A['onesided_XGB_calibrated']['nv'])} and -{f3(-A['onesided_XGB_calibrated']['vpn'])})",
 "orderedhalf min": f"smallest XGBoost value there is {A['orderedhalf_XGB']['rho_gap']['min']:.4f}",
 "orderedhalf signs": f"RBO in {oh('XGB','rbo_gap')} of 11 for XGBoost and {oh('RF','rbo_gap')} of 11 for Random Forest, Top-5 in {oh('XGB','top5_gap')} and {oh('RF','top5_gap')}",
 "PI cross": f"{f3(A['cross_rho']['RF']['pi_marginal'])} (Random Forest) and {f3(A['cross_rho']['XGB']['pi_marginal'])} (XGBoost) for marginal and {f3(A['cross_rho']['RF']['pi_conditional'])} and {f3(A['cross_rho']['XGB']['pi_conditional'])} for conditional permutation importance, against {f3(A['cross_rho']['RF']['shap'])} and {f3(A['cross_rho']['XGB']['shap'])} for SHAP",
 "family XGB": f"every semantic-family gap ({f3(min(fam['XGB']['fam_mean_gap'],fam['XGB']['fam_gap'],fam['XGB']['fam_alt_gap']))}–{f3(max(fam['XGB']['fam_mean_gap'],fam['XGB']['fam_gap'],fam['XGB']['fam_alt_gap']))})",
 "family RF": f"it is not ({f3(min(fam['RF']['fam_mean_gap'],fam['RF']['fam_gap'],fam['RF']['fam_alt_gap']))}–{f3(max(fam['RF']['fam_mean_gap'],fam['RF']['fam_gap'],fam['RF']['fam_alt_gap']))} under",
 "path vs interv": f"{f3(min(r(f'rob_{m}_{c}_path_vs_interv') for m in ('RF','XGB') for c in ('novpn','vpn')))}–{f3(max(r(f'rob_{m}_{c}_path_vs_interv') for m in ('RF','XGB') for c in ('novpn','vpn')))} for Random Forest and XGBoost and {f3(min(r(f'rob_ET_{c}_path_vs_interv') for c in ('novpn','vpn')))}–{f3(max(r(f'rob_ET_{c}_path_vs_interv') for c in ('novpn','vpn')))} for Extra Trees",
 "ET vs RF": f"= {f3(min(r('rob_ET_vs_RF_novpn_path'),r('rob_ET_vs_RF_vpn_path')))}–{f3(max(r('rob_ET_vs_RF_novpn_path'),r('rob_ET_vs_RF_vpn_path')))} for the path-dependent and {f3(min(r('rob_ET_vs_RF_novpn_interv'),r('rob_ET_vs_RF_vpn_interv')))}–{f3(max(r('rob_ET_vs_RF_novpn_interv'),r('rob_ET_vs_RF_vpn_interv')))} for the interventional",
 "distinct lists": f"one to three distinct" if (min(D.values()), max(D.values())) == (1, 3) else "<<distinct-list range changed>>",
}
bad = [k for k, v in expect.items() if v not in t]
for k, v in expect.items(): print(("OK  " if k not in bad else "MISS"), k, "|", v)
sys.exit(1 if bad else 0)
