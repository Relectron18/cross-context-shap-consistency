"""v13: correct the verified defects of the standalone review of v12 (paper/reviews/2026-10-20-v12-defect-review.md;
audit ledger in paper/reviews/2026-10-20-v13-audit-ledger.md).

Numbers that changed are read from results/p8_v13.json (written by src/p8_v13_audit.py) and formatted here, so
the text cannot drift from the result file. Each replacement must match exactly once in one paragraph; text with
a mathematical symbol (rendered as a Word equation) is never inside an old string. New text may contain symbols;
mathfix.fix turns them into equations at the end.
Usage: python fix_v13.py <v12 docx> <output docx>
"""
import sys, copy, json, re, zipfile
from pathlib import Path
from lxml import etree
import mathfix, edits_lib as E
from edits_lib import W, MN, q, ptext, replace, rewrite

SRC, OUT = sys.argv[1], sys.argv[2]
RES = Path(__file__).resolve().parents[1] / "results"
A = json.load(open(RES / "p8_v13.json"))
zin = zipfile.ZipFile(SRC)
doc = etree.fromstring(zin.read("word/document.xml"))
body = E.body = doc.find(q(W, "body"))
for e in list(body.iter(q(W, "lastRenderedPageBreak"))): e.getparent().remove(e)
f3 = lambda x: f"{x:.3f}"

def find_c(s):
    hits = [p for p in body.iter(q(W, "p")) if s in ptext(p)]
    assert len(hits) == 1, ("paragraph not unique", s[:70], len(hits))
    return hits[0]

def rep(old, new, where=None):
    replace(find_c(where or old), old, new)

PI = A["pi_cross"]; ET = A["et_vs_rf"]; MF = A["minflowiat_equality"]; DR = A["drop_RF_vpn_min_fiat"]; RL = A["min_iat_reliance_RF"]
CW = A["class_weighting"]; MT = A["matched1330"]; G10 = A["grouped10_cross"]; CM = A["classmix_delta"]
assert A["xgb_tree_method"]["auto_vs_hist_max_abs_prob_diff_novpn"] == 0.0
assert all(v["identical_predictions"] and v["all_cores_vs_one_max_abs_prob_diff"] == 0.0 for v in A["xgb_threads"].values())
et_ii = sorted(ET[c]["interv_interv"] for c in ("novpn", "vpn")); et_pp = sorted(ET[c]["path_path"] for c in ("novpn", "vpn"))
pct = lambda x: f"{100 * x:.0f}%"
HS = A["half_sizes"]
HS_O = (min(HS[f"{m}_{d}"][0] for m in ("RF", "XGB") for d in ("ordered10", "ordered35", "ordered140", "orderedcal", "orderedhalf")),
        max(HS[f"{m}_{d}"][1] for m in ("RF", "XGB") for d in ("ordered10", "ordered35", "ordered140", "orderedcal", "orderedhalf")))
assert HS["RF_ordered140"][0] == HS_O[0]
assert HS["XGB_random"] == HS["RF_random"] == [3476, 3477]   # the range stated in IV-D for random halves
F1H = (min(A["table7"][f"{m}_orderedhalf"][k] for m in ("RF", "XGB") for k in ("f1_nv", "f1_vpn")),
       max(A["table7"][f"{m}_orderedhalf"][k] for m in ("RF", "XGB") for k in ("f1_nv", "f1_vpn")))

R = [
 # ---- II: wording, "data set"
 ("Explanation capability is being required of security classifiers for both analyst support and the uncovering of spurious correlations",
  "Security classifiers are increasingly expected to explain their decisions, both to support analysts and to expose spurious correlations"),
 ("The same data set was later used", "The same dataset was later used"),
 ("that the data set was constructed to contrast", "that the dataset was constructed to contrast"),
 ("does not require an additional background data set", "does not require an additional background dataset"),
 # ---- IV-B: where explanations are computed
 ("Explanations are always computed on the held-out test partition of the context in question.",
  "Except in the feature-selection step of Section IV-G, explanations are computed on the held-out test partition of the context in question."),
 # ---- IV-C: ties, figure order
 ("Spearman correlation is computed on ranks with tied values given their average rank, so (3) is just the familiar no-tie formula.",
  "Both coefficients are computed on ranks in which tied values receive their average rank; Spearman's coefficient is then the correlation of these ranks, which equals (3) when there are no ties. SHAP importances contain no ties, and ties arise only for permutation importance (Section VI-E)."),
 ("importance per feature (Fig. 4).", "importance per feature (Section VI-A)."),
 # ---- IV-D: clusters, calibration, row order, explained records
 ("with at least four clusters per class. Whole clusters are assigned at random to the two halves, larger clusters first, each to the currently smaller half.",
  "with at least four clusters per class (scikit-learn k-means with three initialisations and a seed fixed per repeat, context and class). Whole clusters are then assigned to the halves greedily: from the largest to the smallest, with clusters of equal size in random order, each goes to the half that currently holds fewer flows."),
 ("which is why the half-models' macro F1 falls as the clusters grow (Table VII).",
  "which is why the half-models' macro F1 falls as the clusters grow (Section VI-C)."),
 ("for CHAT, FT, MAIL, STREAMING and VOIP, plus one VPN P2P capture; BROWSING has no dedicated captures.",
  "for CHAT, FT, MAIL, STREAMING and VOIP and one VPN P2P capture; it lists no Non-VPN P2P capture and no dedicated BROWSING captures."),
 ("(median 44)", f"(median {A['flows_per_capture']['novpn']['median']:.1f})"),
 ("(median 246)", f"(median {A['flows_per_capture']['vpn']['median']:.1f})"),
 ("divided by the context's median flows per capture.",
  "divided by the context's median flows per capture. Non-VPN P2P, which the list omits, is counted as one capture; with the minimum of four clusters this has no effect."),
 ("the prefix list comes from third-party code, P2P has at most one capture per context,",
  "the prefix list comes from third-party code, it lists at most one P2P capture per context,"),
 ("A second proxy does not use the features.", "A second proxy is not built from the features."),
 ("appear in the order in which the flow meter wrote them.", "appear to be in the order in which the flow meter wrote them."),
 ("lies at 0.37–0.76 times the distance of the nearest of 40 random rows,",
  "lies at about 0.36–0.76 times the distance of the nearest of 40 random rows (the exact range depends on which random rows are drawn),"),
 ("Blocks are assigned to halves in the same way as clusters.",
  f"Blocks are assigned to halves in the same way as clusters. The resulting halves hold {HS_O[0]:,}–{HS_O[1]:,} training flows ({HS_O[0]:,} with blocks of about 140 flows), against {HS['RF_random'][0]:,}–{HS['RF_random'][1]:,} for random halves."),
 ("BROWSING has no capture files of its own and P2P has a single capture per context, so for these two classes, which hold 34–40% of the training flows, no within-class grouping can separate captures.",
  "BROWSING and P2P, which hold 34–40% of the training flows, cannot be divided by capture at all: BROWSING has no capture files of its own in the prefix list, so its flows cannot be traced to captures, and P2P has at most one listed capture per context."),
 ("rather than from the models, and none of the designs controls for this.",
  f"rather than from the models. The permutation null shares this asymmetry, since each pseudo-context is explained on its own test partition; with records that are exchangeable between pseudo-contexts, the asymmetry and the class-mix difference together give a mean gap of +{f3(A['table7']['RF_null']['gap'])} for Random Forest and {A['table7']['XGB_null']['gap']:.3f} for XGBoost. No grouped design controls for it, and in the real contexts the explained records also differ in distribution."),
 # ---- IV-E, IV-G, IV-H
 ("XGBoost was run single-threaded; refitting with two threads gave identical predictions.",
  "XGBoost used one thread in the repeated designs and all available cores in the other fits; refitting the primary models with one thread and with two gave identical predictions."),
 ("Third, class-matched training uses 190 training flows per class",
  "Third, class-matched training, run for XGBoost only, uses 190 training flows per class"),
 ("while in S2, -1 is replaced by NaN, which is handled natively", "while in S2, -1 is replaced by NaN in every feature, which is handled natively"),
 # ---- VI-B
 ("and the mean absolute rank change is 3.04 and 3.39.", "and the mean absolute rank change is 3.04 and 3.39 positions."),
 ("and the cross-context values stay at about the same level.",
  f"and cross-context ρ stays at about the same level ({f3(CW['RF']['rho'][1])} and {f3(CW['XGB']['rho'][1])}). The top-weighted measures move more: for Random Forest, RBO rises from {f3(CW['RF']['rbo'][0])} to {f3(CW['RF']['rbo'][1])} and the L1 distance falls from {f3(CW['RF']['l1'][0])} to {f3(CW['RF']['l1'][1])}, and the Top-10 overlap drops by one feature for both models (Table VI)."),
 ("but it equals that minimum in only 43% of Non-VPN and 31% of VPN training flows, because",
  f"but among the training flows in which at least one directional minimum is defined, it equals that minimum in only {pct(MF['novpn']['share_defined'])} (Non-VPN) and {pct(MF['vpn']['share_defined'])} (VPN) of cases ({pct(MF['novpn']['share_all'])} and {pct(MF['vpn']['share_all'])} of all training flows), because"),
 # ---- VI-C
 ("-0.002 for XGBoost (range -0.018 to 0.009)",
  f"{A['table7']['XGB_null']['gap']:.3f} for XGBoost (range {A['table7']['XGB_null']['lo']:.3f} to {A['table7']['XGB_null']['hi']:.3f})"),
 ("while cross-context agreement is barely changed, and",
  f"while cross-context agreement changes little for XGBoost ({f3(G10['XGB']['grouped10'])} against {f3(G10['XGB']['random'])}) but falls for Random Forest ({f3(G10['RF']['grouped10'])} against {f3(G10['RF']['random'])}), and"),
 ("0.751 for XGBoost and 0.296 for Random Forest", f"{f3(A['table7']['XGB_calibrated']['w_vpn'])} for XGBoost and {f3(A['table7']['RF_calibrated']['w_vpn'])} for Random Forest"),
 ("with RBO gaps of +0.091 and +0.046.", f"with RBO gaps of +{f3(A['table7']['XGB_orderedhalf']['rbo'])} and +{f3(A['table7']['RF_orderedhalf']['rbo'])}."),
 ("so only the outer partition and the seeds change.",
  "so what changes from repeat to repeat is mainly the outer partition, the subsample drawn from each half and the seeds."),
 ("Clusters may overcorrect, and blocks may undercorrect wherever a capture crosses a block boundary. Reading them as rough bounds,",
  f"Clusters may overcorrect. Blocks may undercorrect wherever a capture crosses a block boundary, but they are not independent of the features either: row position is correlated with some features within a class, and the first-versus-second-half split lowers half-model macro F1 to {F1H[0]:.2f}–{F1H[1]:.2f}, so it may also separate regions of feature space. Reading the two proxies, with these caveats, as rough bounds,"),
 # ---- VI-D
 ("The swap does not bring a context closer to the other one.", "The swap does not bring a context materially closer to the other one."),
 ("against a cross-context gap of 0.13–0.19 under random halves.",
  f"against a cross-context gap of 0.13–0.19 under random halves. The two XGBoost changes are small moves toward the other context, no larger than the spread across draws (SD {min(v[s] for v in CM.values() for s in ('sd_natural', 'sd_swapped')):.3f}–{max(v[s] for v in CM.values() for s in ('sd_natural', 'sd_swapped')):.3f})."),
 ("the class-mix swap, class-matched training with a same-size reference", "the class-mix swap, class-matched training of XGBoost with a same-size reference"),
 ("Class-matched training yields the same answer given comparison with a reference of the same size.",
  "Class-matched training of XGBoost gives the same answer when it is compared with a reference of the same size."),
 ("(approximately 0.85, compared to 0.77–0.82 for full-size models)",
  f"(approximately 0.85, compared with {f3(MT['xgb_primary_cross'])} for the full-size XGBoost models and {f3(MT['xgb_random_half_cross'])} between their random halves)"),
 # ---- VI-E
 ("For Random Forest the block as a whole gains about six percentage points under VPN.",
  "For Random Forest the block's combined SHAP share rises by about six percentage points under VPN."),
 ("dropping min_fiat increases the shares of min_flowiat and min_biat by 2.2 and 1.5 percentage points, respectively;",
  f"dropping min_fiat raises the shares of min_flowiat and min_biat by {DR['min_flowiat']['renormalised']:.1f} and {DR['min_biat']['renormalised']:.1f} percentage points more than renormalising the remaining shares after the loss of min_fiat's {A['drop_RF_vpn_min_fiat_base_share']:.1f}% would give ({DR['min_flowiat']['raw']:.1f} and {DR['min_biat']['raw']:.1f} points before that correction);"),
 ("We therefore report a conditional variant as well, which permutes each feature only within quintile cells of its two most correlated features [36].",
  "We therefore also report a simplified conditional variant inspired by [36], which permutes each feature only within the cells formed by the training-data quintiles of the two features most correlated with it (absolute Spearman correlation on the training partition)."),
 ("is 0.611 (Random Forest) and 0.719 (XGBoost) for marginal and 0.657 and 0.575 for conditional permutation importance, against 0.830 and 0.786 for SHAP.",
  f"is {f3(PI['marginal_RF']['rho_average'])} (Random Forest) and {f3(PI['marginal_XGB']['rho_average'])} (XGBoost) for marginal and {f3(PI['conditional_RF']['rho_average'])} and {f3(PI['conditional_XGB']['rho_average'])} for conditional permutation importance, against 0.830 and 0.786 for SHAP. Here negative permutation importances are set to zero and tied features receive their average rank; for Random Forest, {PI['marginal_RF']['n_nonpositive']['novpn']} of the 23 features are tied at zero in each context under marginal permutation, and {PI['conditional_RF']['n_nonpositive']['novpn']} and {PI['conditional_RF']['n_nonpositive']['vpn']} under conditional permutation. Without setting negative values to zero, the four values are {f3(PI['marginal_RF']['rho_raw_unclipped'])}, {f3(PI['marginal_XGB']['rho_raw_unclipped'])}, {f3(PI['conditional_RF']['rho_raw_unclipped'])} and {f3(PI['conditional_XGB']['rho_raw_unclipped'])}."),
 ("For Random Forest, credit is redistributed within the block, but the block itself gains importance under VPN, so part of the shift is a change in what the models rely on.",
  f"For Random Forest, credit is redistributed within the block and the block's combined SHAP share also rises under VPN, but the measures based on predictive performance do not show greater reliance on it: permuting the three minima together costs {f3(RL['novpn']['joint_perm_df1'])} macro F1 without a VPN and {f3(RL['vpn']['joint_perm_df1'])} with one, and dropping them and retraining costs {f3(RL['novpn']['drop_all3_df1'])} and {f3(RL['vpn']['drop_all3_df1'])}. The SHAP share and these measures therefore disagree on whether the Random Forest relies more on the block under VPN."),
 ("the conditional version permutes within quintile cells of each feature's two most correlated features.",
  "the conditional version permutes within cells formed by the training-data quintiles of each feature's two most correlated features."),
 # ---- VI-F
 ("With blocks of consecutive rows, however, within-VPN rankings are again steadier (calibrated blocks: 0.942 against 0.919 for XGBoost, 0.906 against 0.884 for Random Forest), so that the reversal in trend depends on the nature of the grouping.",
  "With blocks of consecutive rows, within-VPN rankings are again the steadier ones (calibrated blocks: 0.942 against 0.919 for XGBoost, 0.906 against 0.884 for Random Forest). Row blocks are, however, the weaker proxy in four VPN classes: adjacent rows of VPN FT, MAIL, P2P and VOIP are nearly as different as random rows (Section IV-D), so the blocks may keep fewer related VPN flows together than intended. Which context gives the steadier rankings therefore depends on the grouping, and neither proxy settles it."),
 # ---- VII
 ("within 2.1 percentage points of one another", "within 2.15 percentage points of one another"),
 ("Therefore, redundancy among related flows does not fully explain the observed differences.",
  "At the finest grouping, therefore, redundancy among related flows does not explain the XGBoost family-level difference; at coarser grouping it may."),
 # ---- VIII-A
 ("These scores fall in the same range as the cluster-grouped cross-validation of Section III-C (0.47–0.64), but the comparison is only indicative.",
  f"These scores are close to those of the cluster-grouped cross-validation of Section III-C (0.47–0.64), although both Non-VPN-to-VPN scores ({f3(A['transfer_f1']['transfer_XGB_novpn_to_vpn'])} and {f3(A['transfer_f1']['transfer_RF_novpn_to_vpn'])}) fall below that range, and the comparison is only indicative."),
 # ---- IX
 ("0.730–0.801 for the path-dependent and 0.709–0.776 for the interventional formulation).",
  f"{f3(et_pp[0])}–{f3(et_pp[1])} when both are explained with the path-dependent formulation and {f3(et_ii[0])}–{f3(et_ii[1])} when both use the interventional one)."),
 # ---- X
 ("Blocks of consecutive rows in the original files do not exploit the features and leave the half-models accurate.",
  "Blocks of consecutive rows in the original files are not built from the features, although row position is correlated with some of them, and they leave the half-models accurate."),
 ("clusters may overcorrect and blocks may undercorrect.",
  "clusters may overcorrect, and blocks may undercorrect or, at their strictest, also separate regions of feature space."),
 ("whereas for Random Forest the block as a whole gains importance under VPN.",
  "whereas for Random Forest the block's combined SHAP share rises under VPN, although permuting or removing the block costs slightly less macro F1 under VPN than without it."),
 ("If there is an application of the explanations for decision-making tasks, such as in selecting features, the performance of these explanations should be tested on out-of-sample data drawn from the target context.",
  "When explanations drive a decision, such as which features to keep, that decision should be tested on held-out data from the target context."),
 # ---- XI
 ("and BROWSING and P2P cannot be split by capture at all.",
  "and BROWSING (no listed captures) and P2P (at most one listed capture per context) cannot be split by capture with the information available."),
 # ---- XII
 ("XGBoost's macro F1-score falls from", "XGBoost's macro F1 falls from"),
 ("with macro F1-scores ranging from 0.43 to 0.61", "with macro F1 ranging from 0.43 to 0.61"),
 ("Unlike feature-based clustering, this approach does not rely on the input features to construct groups.",
  "Unlike feature-based clustering, this approach does not use the input features to construct groups, although row position is itself correlated with some features."),
 ("Random Forest, in contrast, shows an increase in the importance of the minimum-IAT feature group itself.",
  "For Random Forest, the combined SHAP share of the minimum-IAT group rises, but permuting or removing the group costs less macro F1 under VPN than without it, so the two kinds of measure disagree on whether the model relies on the group more."),
 ("For SHAP-selected feature sets containing at least 10 features, transferring the selected set between contexts results in an average macro-F1 loss",
  "For SHAP-selected feature sets of 10 or 15 features, transferring the selected set between contexts results in an average macro F1 loss"),
 # ---- Table notes
 ("The matched-subset rows give the per-class sizes used for class-matched training in Section VI-D.",
  "The matched-subset rows give the per-class training sizes used for class-matched training of XGBoost in Section VI-D; matched models are explained on the full test partition, so there is no matched test set."),
]
for old, new in R:
    rep(old, new)

# ---- IV-G: the feature-selection paragraph, rewritten whole (it contained garbled clauses and "give them equal weight")
rewrite(find_c("The selection process in the feature list transfer"), [
 "In the feature-list transfer, features are selected from the training data only [57–59]. Each training partition is split, stratified by class, into an 80% inner-training part and a 20% inner-validation part. A selector model is fitted on the inner-training part, its TreeSHAP importance is computed on the inner-validation part (for Random Forest on a stratified sample of 800 inner-validation records), and the k most important features are kept. The reduced model is then refitted on the whole training partition with these features and evaluated on the test partition. The own list comes from the target context, and the transferred list comes from the other context with the same inner-split seed. We run k = 5, 10 and 15 for both models and repeat the selection with five inner-split seeds. Since each list is scored on both targets, the four scores of a model form a 2 × 2 design of list origin by target, which separates a context-match effect from a difference in overall list quality."])

# ---- Data and code availability (four kinds of material kept apart; nothing is claimed to be public that is not)
rewrite(find_c("The ISCX VPN-nonVPN data, including the Scenario A2"), [
 "The ISCX VPN-nonVPN benchmark, including the Scenario A2 time-based ARFF files used here, is publicly available from the Canadian Institute for Cybersecurity [50]. The analysis code (Python scripts for preprocessing, the reference designs, the explanation analyses, the tables and the figures) is kept in a private repository and has not been publicly released; it is available from the corresponding author on reasonable request. The processed artifacts (the deduplicated partitions, the mapping of each flow to its row in the original ARFF files, the per-record SHAP values and the result files) are available on the same terms, and the row mapping can be rebuilt from the public ARFF files with that code. Every table and figure that reports results was generated from these result files by the same code."])

# ---- title page: corresponding-author note for the asterisk
em = find_c("ganesh.khekare@vit.ac.in*")
note = copy.deepcopy(em); em.addnext(note); rewrite(note, ["*Corresponding author."])

# ---- Table III: XGBoost tree method (verified from the trained booster's configuration)
tm = A["xgb_tree_method"]["used"]
rep("eval_metric = mlogloss; random_state = 42", f"eval_metric = mlogloss; tree_method = {tm} (library default, identical to hist here); random_state = 42")

# ---- table cells (double rounding in v12: 4-decimal summaries rounded again to 3)
def table_after(caption):
    cap = [p for p in body.iter(q(W, "p")) if ptext(p).strip() == caption]; assert len(cap) == 1, caption
    el = cap[0].getnext()
    while el.tag != q(W, "tbl"): el = el.getnext()
    return el
def rows(tbl): return [[ptext(c).strip() for c in tr.iter(q(W, "tc"))] for tr in tbl.iter(q(W, "tr"))], list(tbl.iter(q(W, "tr")))
def set_cell(tbl, section, label, col, old, new):
    txt, trs = rows(tbl); sec = section is None
    for t, tr in zip(txt, trs):
        if section is not None and t and t[0].startswith(section): sec = True; continue
        if sec and t and t[0].startswith(label):
            tc = list(tr.iter(q(W, "tc")))[col]; ps = list(tc.iter(q(W, "p")))
            p = [x for x in ps if ptext(x).strip() == old]; assert len(p) == 1, (label, col, old, [ptext(x) for x in ps])
            replace(p[0], old, new); return
    raise AssertionError((section, label, col))
T2 = table_after("TABLE II")
for lab in ("Matched subset (per class)", "Matched subset (total)"):
    old = "48" if "per class" in lab else "336"
    set_cell(T2, None, lab, 2, old, "—"); set_cell(T2, None, lab, 5, old, "—")
T7 = table_after("TABLE VII"); t7 = A["table7"]
XS, RS = "XGBoost (11", "Random Forest (11"
fix7 = [(XS, "Grouped, calibrated", 3, "0.751", f3(t7["XGB_calibrated"]["w_vpn"])),
        (XS, "Row blocks, ~10", 2, "0.947", f3(t7["XGB_ordered10"]["w_nv"])),
        (XS, "Permutation null", 3, "0.944", f3(t7["XGB_null"]["w_vpn"])),
        (XS, "Permutation null", 5, "-0.002 (0.009)", f"{t7['XGB_null']['gap']:.3f} ({f3(t7['XGB_null']['sd'])})"),
        (XS, "Permutation null", 5, "-0.018 to 0.009", f"{t7['XGB_null']['lo']:.3f} to {f3(t7['XGB_null']['hi'])}"),
        (RS, "Grouped, calibrated", 3, "0.296", f3(t7["RF_calibrated"]["w_vpn"])),
        (RS, "Row blocks, ~10", 5, "+0.168 (0.029)", f"+{f3(t7['RF_ordered10']['gap'])} ({f3(t7['RF_ordered10']['sd'])})"),
        (RS, "Row blocks, calibrated", 4, "0.743", f3(t7["RF_orderedcal"]["cross"])),
        (RS, "Row blocks, first", 6, "+0.046", f"+{f3(t7['RF_orderedhalf']['rbo'])}")]
for sec, lab, col, old, new in fix7: set_cell(T7, sec, lab, col, old, new)
T11 = table_after("TABLE XI")
set_cell(T11, None, "RF / Six families, mean per member", 4, "0.754", "0.755")
set_cell(T11, None, "XGB / Six families, mean per member", 6, "+0.054", "+0.055")

# ---- apostrophes: one style (ASCII) for apostrophes between letters
na = 0
for t in body.iter(q(W, "t")):
    if t.text and "’" in t.text:
        new = re.sub(r"(?<=[A-Za-z])’(?=[A-Za-z]|\s|$)", "'", t.text); na += new != t.text; t.text = new
print("runs with apostrophes normalised", na)
# ---- body paragraphs that the team edits left with other paragraph formatting (a Heading 1 style, the BodyText
# style, keep-with-next or no first-line indent); they get the paragraph properties of an ordinary body paragraph
tmpl = find_c("A VPN does not change which application a flow comes from").find(q(W, "pPr"))
nf = 0
for p in body.findall(q(W, "p")):
    s_ = ptext(p)
    if len(s_) < 150 or s_.lstrip().startswith(("Note.", "Fig.", "[", "Abstract", "Keywords", "prabhav.")): continue
    pp = p.find(q(W, "pPr")); st = pp.find(q(W, "pStyle")) if pp is not None else None
    sty = st.get(q(W, "val")) if st is not None else None
    ind = pp.find(q(W, "ind")) if pp is not None else None
    if sty in ("Heading1", "BodyText") or ind is None:
        if s_.startswith(("where ",)): continue            # continuation lines after an equation keep no indent
        new = copy.deepcopy(tmpl)
        if pp is not None: p.replace(pp, new)
        else: p.insert(0, new)
        nf += 1
for t in body.iter(q(W, "t")):
    if t.text and t.text.startswith(" The answer to the question posed"): t.text = t.text.lstrip()
print("body paragraphs given standard formatting", nf)
# ---- empty paragraphs left before subsection headings by the team edits (they show as blank lines)
def is_empty(p): return not ptext(p).strip() and not list(p.iter(q(W, "drawing"))) and not list(p.iter(q(W, "br")))
ne = 0
for p in list(body.findall(q(W, "p"))):
    if not is_empty(p): continue
    nxt = p.getnext()
    while nxt is not None and nxt.tag == q(W, "p") and is_empty(nxt): nxt = nxt.getnext()
    if nxt is not None and nxt.tag == q(W, "p") and re.match(r"^[A-I]\. [A-Z][a-z]", ptext(nxt)):
        p.getparent().remove(p); ne += 1
print("empty paragraphs before subsection headings removed", ne)
# ---- run-level language: team-edited paragraphs carried en-IN; the document language is en-US everywhere else
nl = 0
for e in body.iter(q(W, "lang")):
    for k, v in list(e.attrib.items()):
        if v == "en-IN": e.set(k, "en-US"); nl += 1
print("en-IN language tags set to en-US", nl)
print("equations added", mathfix.fix([body]))

xml = etree.tostring(doc, xml_declaration=True, encoding="UTF-8", standalone=True)
zo = zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED)
for item in zin.infolist():
    zo.writestr(item, xml if item.filename == "word/document.xml" else zin.read(item.filename))
zo.close()
print("->", OUT)
