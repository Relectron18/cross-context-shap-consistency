# Part B of fix_v14.py (executed inside it): the two new analyses. Every number is read from
# results/p11_v14.json (src/p11_v14_summary.py). Uses rep / rew / guard / find_c / body from fix_v14.py.
import numpy as np
G = json.load(open(RES / "p11_v14.json"))
GN, CR = G["grouped_null"], G["common_records"]
f3 = lambda x: f"{x:.3f}"
def sg(x, d=3):
    s = f"{x:+.{d}f}"
    return s[1:] if float(s) == 0 else s
NAMES = {"grouped10": "clusters of about 10 flows", "grouped35": "clusters of about 35 flows", "calibrated": "calibrated clusters",
         "orderedcal": "calibrated row blocks", "orderedhalf": "the first-versus-second-half split"}
ND = ("grouped10", "grouped35", "calibrated", "orderedcal", "orderedhalf")
NEW = "results (v14)"

# ---- I. contributions: the two new references
rep("and a permutation null in which flows are reassigned at random between two pseudo-contexts.",
    "and a permutation null in which flows are reassigned at random between two pseudo-contexts, run with random and with grouped halves. It also checks whether the gap survives when the models of both contexts are explained on the same records.")

# ---- IV-D: grouped null and common explained records (method)
P8 = json.load(open(RES / "p8_v13.json"))["half_sizes"]
real_hs = [x for m in ("RF", "XGB") for d in ND for x in P8[f"{m}_{d}"]]
null_hs = [GN["nullgroup"][f"{m}_{d}"][k] for m in ("RF", "XGB") for d in ND for k in ("half_size_min", "half_size_max")]
rep("shows what the half-split procedure and the class-mix difference produce when nothing else separates the two contexts.",
    "shows what the half-split procedure and the class-mix difference produce when nothing else separates the two contexts. "
    "For five grouped designs (clusters of about 10 and 35 flows, calibrated clusters, calibrated row blocks and the first-versus-second-half split) we build the null in two ways, because neither is neutral on its own. "
    "In the flow-level version, flows are reassigned one by one as above and the grouped design is then applied within each pseudo-context. Flows of one capture then sit in both pseudo-contexts, so the pseudo cross pairs share more related flows than the grouped within pairs do, which pushes Δ down. "
    "In the group-level version, whole capture-sized groups move instead: within each class and source file, the pooled flows are divided into the calibrated clusters (for the cluster designs) or the calibrated row blocks (for the row-block designs), and the groups are taken in random order, each going to the pseudo-context that is further below its original share of the class. "
    f"Because whole groups move, class counts are matched only approximately, and the halves hold {min(null_hs):,}–{max(null_hs):,} training flows, against {min(real_hs):,}–{max(real_hs):,} in the real grouped designs. "
    "Calibrated clusters are regions of feature space, so for the cluster designs this version also gives the two pseudo-contexts different feature distributions, which pushes Δ up; row blocks are not built from the features, although row position is correlated with some of them. "
    "In both versions the grouped design runs separately on the flows of each source file, so that both halves hold the same mix of Non-VPN and VPN rows, and in the calibrated designs the number of groups is scaled to the share of the file's flows that a pseudo-context holds, so that groups stay about capture-sized. "
    "An earlier version without the per-file split was discarded because its two halves could hold very different shares of Non-VPN and VPN rows.",
    allow=NEW)
rep("Part of the cross-context disagreement may therefore come from the difference in explained records rather than from the models. The permutation null shares this asymmetry, since each pseudo-context is explained on its own test partition; with records that are exchangeable between pseudo-contexts, the asymmetry and the class-mix difference together give a mean gap of +0.028 for Random Forest and -0.001 for XGBoost. No grouped design controls for it, and in the real contexts the explained records also differ in distribution.",
    "Part of the cross-context disagreement could therefore come from the difference in explained records rather than from the models. To measure this, we refit the random-halves design with the same seeds and explain every half-model on the explained records of both contexts, so that within- and cross-context pairs can be compared on one common record set: the Non-VPN records, the VPN records, or both together. Each refitted half-model reproduces the stored explanation on its own records exactly. The permutation null shares the asymmetry, since each pseudo-context is explained on its own test partition, but there the records of the two pseudo-contexts are exchangeable.",
    allow=NEW)

# ---- Table VII: grouped-null rows after each model's "Permutation null" row
def table_after(caption):
    cap = [p for p in body.iter(q(W, "p")) if ptext(p).strip() == caption]; assert len(cap) == 1, caption
    el = cap[0].getnext()
    while el.tag != q(W, "tbl"): el = el.getnext()
    return el
def set_par(p, text):
    rs = list(p.iter(q(W, "r"))); assert rs, "empty cell paragraph"
    for r in rs[1:]: r.getparent().remove(r)
    t = rs[0].find(q(W, "t")); t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
T7 = table_after("TABLE VII")
LABEL = {"grouped10": "grouped ~10 per cluster", "grouped35": "grouped ~35 per cluster", "calibrated": "grouped calibrated",
         "orderedcal": "row blocks calibrated", "orderedhalf": "row blocks first vs second half"}
VERS = (("nullstrat", "Null (flows), "), ("nullgroup", "Null (groups), "))
model = None; anchors = {}
for tr in T7.iter(q(W, "tr")):
    first = ptext(tr).strip()
    if first.startswith("XGBoost ("): model = "XGB"
    elif first.startswith("Random Forest ("): model = "RF"
    elif first.startswith("Permutation null") and model: anchors[model] = tr
assert set(anchors) == {"XGB", "RF"}
for m, tr0 in anchors.items():
    prev = tr0
    for ver, lab in VERS:
      for d in ND:
        g = GN[ver][f"{m}_{d}"]; s = g["null"]
        vals = [[lab + LABEL[d]], [f"{f3(s['f1_nv'])} / {f3(s['f1_vpn'])}"], [f3(s["rho_wnv"])], [f3(s["rho_wvpn"])], [f3(s["rho_cross"])],
                [f"{sg(s['rho_gap'])} ({f3(g['null_gap_sd'])})", f"{g['null_gap_min']:.3f} to {g['null_gap_max']:.3f}"],
                [sg(s["rbo_gap"])], [sg(s["top5_gap"], 2)], [sg(s["rho_w_gap"])]]
        new = copy.deepcopy(tr0)
        cells = list(new.iter(q(W, "tc"))); assert len(cells) == len(vals)
        for tc, v in zip(cells, vals):
            ps = [p for p in tc.iter(q(W, "p")) if ptext(p).strip()]
            assert len(ps) == len(v), (d, v, [ptext(p) for p in ps])
            for p, text in zip(ps, v): set_par(p, text)
        prev.addnext(new); prev = new
rep("The null design pools the flows of each class across contexts and reassigns them at random, without replacement, to pseudo-contexts with the original class counts.",
    "The null design pools the flows of each class across contexts and reassigns them at random, without replacement, to pseudo-contexts with the original class counts. Rows marked Null (flows) apply a grouped design to those pseudo-contexts; rows marked Null (groups) build the pseudo-contexts from whole calibrated clusters or row blocks first, so their class counts match the original ones only approximately (Section IV-D).")

# ---- Fig. 2: pipeline figure regenerated by src/figs.py (lists the grouped null and the same-record check)
FIG2_BYTES = (RES.parent / "figures" / "fig2_pipeline.png").read_bytes()
REPLACE_MEDIA = {"word/media/image2.png": FIG2_BYTES}

# ---- VI-C: common explained records (result)
def rng_(m, key="gap", sets=("nv", "vpn", "union")):
    v = [CR[m][key][s] for s in sets]; return min(v), max(v)
xg, rg = rng_("XGB"), rng_("RF")
rb = rng_("RF", "rbo_gap")
prim = {m: sorted(CR[m]["primary"]["on_common"][s]["rho"] for s in ("nv", "vpn", "union")) for m in ("RF", "XGB")}
assert all(CR[m]["n_repeats_positive_union"] == 11 for m in ("RF", "XGB"))
assert all(min(CR[m]["gap_min"][s] for s in ("nv", "vpn", "union")) > 0 for m in ("RF", "XGB"))
own = {m: CR[m]["gap"]["own"] for m in ("RF", "XGB")}
share_lo, share_hi = (own["RF"] - rg[1]) / own["RF"], (own["RF"] - rg[0]) / own["RF"]
rep("of -0.001 for XGBoost (range -0.019 to 0.009) and +0.028 for Random Forest (range -0.010 to 0.052).",
    "of -0.001 for XGBoost (range -0.019 to 0.009) and +0.028 for Random Forest (range -0.010 to 0.052)."
    f" Nor does the gap come mainly from explaining the two contexts on different records. When every half-model is explained on the same records, Δ is {f3(xg[0])}–{f3(xg[1])} for XGBoost and {f3(rg[0])}–{f3(rg[1])} for Random Forest across the three common record sets (the Non-VPN records, the VPN records and both together), against {f3(own['XGB'])} and {f3(own['RF'])} when each model is explained on its own context's records, and it is positive in all 11 repeats for both models."
    f" For XGBoost the explained records make little difference to ρ, although the RBO gap falls from {f3(CR['XGB']['rbo_gap']['own'])} to {f3(rng_('XGB', 'rbo_gap')[0])}–{f3(rng_('XGB', 'rbo_gap')[1])}. For Random Forest they account for about {100*share_lo:.0f}–{100*share_hi:.0f}% of the gap, and the RBO gap falls from {f3(CR['RF']['rbo_gap']['own'])} to {f3(rb[0])}–{f3(rb[1])}."
    f" The primary models agree: on common records their cross-context ρ is {f3(prim['RF'][0])}–{f3(prim['RF'][2])} for Random Forest and {f3(prim['XGB'][0])}–{f3(prim['XGB'][2])} for XGBoost, against 0.830 and 0.786 on their own test partitions.",
    allow=NEW)

# ---- helpers for new paragraphs
def new_par_after(anchor_text, text):
    a = find_c(anchor_text); p = copy.deepcopy(a); a.addnext(p); rewrite(p, [text]); return p

# ---- VI-C: grouped null and row blocks on common records (result); claims are asserted before they are written
NG, NS = GN["nullgroup"], GN["nullstrat"]; CRB = G["common_records_rowblocks"]
ROWD, CLUD = ("orderedcal", "orderedhalf"), ("grouped10", "grouped35", "calibrated")
MM = ("XGB", "RF")
row_nulls = [V[f"{m}_{d}"]["null"]["rho_gap"] for V in (NG, NS) for m in MM for d in ROWD]
assert max(abs(x) for x in row_nulls) < 0.04, row_nulls                     # "close to zero" (real row-block gaps are 0.03-0.15)
exc = {m: [NG[f"{m}_{d}"]["real_minus_null"] for d in ROWD] for m in MM}
assert all(x > 0 for v in exc.values() for x in v)
above = [NG[f"{m}_{d}"]["real_repeats_above_null_max"] for m in MM for d in ROWD]
crb = {(m, d): (min(CRB[f"{m}_{d}"]["gap"][x] for x in ("nv", "vpn", "union")), max(CRB[f"{m}_{d}"]["gap"][x] for x in ("nv", "vpn", "union"))) for m in MM for d in ROWD}
assert all(CRB[k]["max_abs_rec_diff_vs_stored"] == 0.0 for k in CRB) and all(v[0] > 0 for v in crb.values())
crb_pos = [CRB[f"{m}_{d}"]["n_positive"][x] for m in MM for d in ROWD for x in ("nv", "vpn", "union")]
rf_rb_red = [(CRB[f"RF_{d}"]["gap"]["own"] - g) / CRB[f"RF_{d}"]["gap"]["own"] for d in ROWD for g in crb[("RF", d)]]
cl_flow = [NS[f"{m}_{d}"]["null"]["rho_gap"] for m in MM for d in CLUD]
cl_grp = [NG[f"{m}_{d}"]["null"]["rho_gap"] for m in MM for d in CLUD]
cl_real = [NG[f"{m}_{d}"]["real_gap"] for m in MM for d in CLUD]
assert max(cl_flow) < 0 < min(cl_grp), (cl_flow, cl_grp)
assert all(NS[f"{m}_{d}"]["null"]["rho_gap"] < NG[f"{m}_{d}"]["real_gap"] < NG[f"{m}_{d}"]["null"]["rho_gap"] for m in MM for d in CLUD)
flow_cross = np.mean([NS[f"{m}_{d}"]["null"]["rho_cross"] for m in MM for d in CLUD])
flow_within = np.mean([(NS[f"{m}_{d}"]["null"]["rho_wnv"] + NS[f"{m}_{d}"]["null"]["rho_wvpn"]) / 2 for m in MM for d in CLUD])
assert flow_cross > flow_within
grp_lower = [(m, d) for m in MM for d in CLUD if NG[f"{m}_{d}"]["null"]["rho_cross"] < NG[f"{m}_{d}"]["real_cross"]]
sd_grp = max(NG[f"{m}_{d}"]["null_gap_sd"] for m in MM for d in CLUD)
closest = min(((m, d) for m in MM for d in CLUD), key=lambda k: NG[f"{k[0]}_{k[1]}"]["null"]["rho_gap"] - NG[f"{k[0]}_{k[1]}"]["real_gap"])
cg = NG[f"{closest[0]}_{closest[1]}"]
nm = {"XGB": "XGBoost", "RF": "Random Forest"}; dn = {"grouped10": "about 10 flows", "grouped35": "about 35 flows", "calibrated": "calibrated clusters"}
ex_k = ("RF", "grouped10"); exg = NG[f"{ex_k[0]}_{ex_k[1]}"]
new_par_after("(RBO in 1 of 11 for XGBoost and 2 of 11 for Random Forest, Top-5 in 3 and 2).",
    "The grouped null separates the two proxies (Table VII). For row blocks, both versions of the null give a mean Δ close to zero "
    f"({min(row_nulls):.3f} to {max(row_nulls):.3f}) at the two settings tested, so this grouping produces almost no gap of its own, and the real row-block gaps lie above it: "
    f"against the group-level null by {f3(exc['XGB'][0])} and {f3(exc['XGB'][1])} for XGBoost and by {f3(exc['RF'][0])} and {f3(exc['RF'][1])} for Random Forest "
    f"at the calibrated and first-versus-second-half settings, with {min(above)}–{max(above)} of the 11 real repeats above the largest null repeat. "
    "The row-block gaps also remain when both contexts are explained on the same records: across the three common record sets they are "
    f"{f3(crb[('XGB', 'orderedcal')][0])}–{f3(crb[('XGB', 'orderedcal')][1])} (calibrated) and {f3(crb[('XGB', 'orderedhalf')][0])}–{f3(crb[('XGB', 'orderedhalf')][1])} (first versus second half) for XGBoost "
    f"and {f3(crb[('RF', 'orderedcal')][0])}–{f3(crb[('RF', 'orderedcal')][1])} and {f3(crb[('RF', 'orderedhalf')][0])}–{f3(crb[('RF', 'orderedhalf')][1])} for Random Forest, "
    f"positive in {min(crb_pos)}–{max(crb_pos)} of the 11 repeats; for Random Forest this is {100*min(rf_rb_red):.0f}–{100*max(rf_rb_red):.0f}% below the gaps on each context's own records. "
    "For clusters, the two versions of the null point in the directions expected from their construction. "
    f"In the flow-level version, cross-context agreement exceeds within-context agreement ({f3(flow_cross)} against {f3(flow_within)} on average), so Δ is negative ({min(cl_flow):.3f} to {max(cl_flow):.3f}). "
    f"In the group-level version, the pseudo-contexts agree less with each other than the real contexts do in {len(grp_lower)} of the 6 cluster settings (for Random Forest with clusters of about 10 flows, cross-context ρ {f3(exg['null']['rho_cross'])} against {f3(exg['real_cross'])}), "
    f"so Δ is positive ({f3(min(cl_grp))} to {f3(max(cl_grp))}), with a spread across repeats of up to {f3(sd_grp)}. "
    f"The real cluster gaps ({f3(min(cl_real))} to {f3(max(cl_real))}) lie between the two versions for every model and cluster setting, in some close to the group-level null ({nm[closest[0]]} with {dn[closest[1]]}: {f3(cg['real_gap'])} against {f3(cg['null']['rho_gap'])}). "
    "No neutral null could be built for feature clusters, so the cluster results do not show whether an excess remains once related flows are kept together.")

# ---- VI-C "Taken together": what the two new checks add
lo_pct, hi_pct = 100 * share_lo, 100 * share_hi
rep("These are descriptive comparisons, not a test of that excess: the permutation null was run only for random halves, where it already gives +0.028 for Random Forest, no grouped design keeps the test records apart from the training sessions, and within-context pairs are explained on shared test records while cross-context pairs are not. Part of the random-split gap therefore reflects variation between groups of related flows, and part of it is retained under grouping by row order.",
    "The grouped null sharpens the row-order end of this range: at the two row-block settings tested, the grouping produces almost no gap by itself, and the real gaps stay positive when both contexts are explained on the same records, whereas for clusters the null itself is uncertain. "
    f"Explaining both contexts on the same records leaves the XGBoost gap almost unchanged and lowers the Random Forest gap by about {lo_pct:.0f}–{hi_pct:.0f}% under random halves and {100*min(rf_rb_red):.0f}–{100*max(rf_rb_red):.0f}% under row blocks. "
    "None of this tests the excess at the level of captures: no grouped design keeps the test records apart from the training sessions, and both proxies only approximate capture boundaries. "
    "Part of the random-split gap therefore reflects variation between groups of related flows, and part of it survives the two row-block settings tested, on each context's own records and on common ones, although under the first-versus-second-half split the XGBoost gap is small and the RBO and Top-5 gaps are negative in some repeats.",
    allow=NEW)

# ---- X-A
rep("Together they suggest that part of the random-split gap reflects variation between groups of related flows, from about 20% to nearly all of it depending on the proxy and model, and that some excess probably remains. That excess is described, not tested: the permutation null covers only random halves, the outer split is random in every design, and within-context and cross-context pairs are explained on different kinds of test records. Without capture identifiers or an independent second capture, the size of that excess cannot be quantified, and because the contexts are different sets of captures it cannot be attributed to tunnelling.",
    "Together they suggest that part of the random-split gap reflects variation between groups of related flows, from about 20% to nearly all of it depending on the proxy and model. "
    f"Two further checks narrow this. Explaining both contexts on the same records leaves the XGBoost gap almost unchanged and lowers the Random Forest gap by about {lo_pct:.0f}–{hi_pct:.0f}% under random halves and {100*min(rf_rb_red):.0f}–{100*max(rf_rb_red):.0f}% under row blocks, so the choice of explained records is not the main source of the disagreement. "
    "A grouped permutation null, run for two row-block and three cluster settings, shows that row-block grouping produces almost no gap by itself, so the positive row-block gaps are excess over the null; for clusters, the two versions of the null fall on either side of the observed gap, and the cluster results leave the question open. "
    "Some excess therefore remains under the grouping least tied to the features, but its size depends on how closely row blocks follow captures, and without capture identifiers or an independent second capture it cannot be pinned down. Because the contexts are different sets of captures, it cannot be attributed to tunnelling.",
    allow=NEW)

# ---- X-E: one practical consequence of the common-records check
rep("When explanations drive a decision, such as which features to keep, that decision should be tested on held-out data from the target context.",
    f"When rankings from two contexts are compared, it is worth checking how much of the difference comes from explaining the models on different records: under random halves this accounted for about {lo_pct:.0f}–{hi_pct:.0f}% of the Random Forest gap here and for almost none of the XGBoost gap. Explaining a model on another context's records evaluates it away from its own data, so this is a diagnostic, not a better ranking. When explanations drive a decision, such as which features to keep, that decision should be tested on held-out data from the target context.",
    allow=NEW)

# ---- XI
rep("Only the division of training data into halves is grouped; the outer 80/20 split is random, and the permutation null was run only for random halves. Within-context pairs are explained on shared test records and cross-context pairs are not; a comparison on a common set of explained records is left for future work.",
    "Only the division of training data into halves is grouped; the outer 80/20 split is random. The grouped null has no neutral form for clusters: reassigning single flows puts flows of one capture in both pseudo-contexts, and reassigning whole clusters makes the pseudo-contexts differ in feature space. For row blocks both versions agree, but they inherit the uncertainty about how well row blocks follow captures. "
    f"The null was run for two of the five row-block settings and three of the four cluster settings. Within-context pairs are explained on shared test records and cross-context pairs, in the designs themselves, are not. Explaining both contexts on common records lowers the Random Forest gaps by about {lo_pct:.0f}–{hi_pct:.0f}% under random halves and {100*min(rf_rb_red):.0f}–{100*max(rf_rb_red):.0f}% under the two row-block settings, so Random Forest gaps measured on each context's own records overstate the difference between the models by roughly that much; the cluster designs and the other row-block settings were not recomputed on common records.",
    allow=NEW)

# ---- abstract: last sentence (kept within 250 words; checked below)
rep("Part of the random-split gap is explained by related flows and part probably remains, although no design tests that remainder directly, and since the two contexts are different sets of captures, none of the differences can be attributed to tunnelling.",
    "Part of the random-split gap reflects related flows; under row-block grouping a smaller gap remains, also on common explained records and above a grouped permutation null, but the contexts are different capture sets, so none of it can be attributed to tunnelling.",
    allow=NEW)
ab = [p for p in body.iter(q(W, "p")) if ptext(p).startswith("Abstract")][0]
assert len(ptext(ab).split()) <= 250, len(ptext(ab).split())

# ---- XII: conclusion rewritten (answers the research question; new checks included; no new findings)
C = ["This study examined whether the features", "A key finding is that the measured difference", "The experiments using consecutive blocks",
     "The feature-level analysis also reveals", "Taken together, these findings suggest"]
CP = [find_c(c) for c in C]
rew(C[0], "This study asked whether the features a classifier relies on to separate seven application categories stay the same when traffic is observed with and without a VPN, and how any difference compares with the variation inside one context. On the ISCX VPN-nonVPN benchmark with its 23 time-related flow features, both models lose accuracy under VPN (macro F1 falls from 0.9340 to 0.8836 for XGBoost and from 0.9171 to 0.8533 for Random Forest), and a model trained in one context reaches only 0.43–0.61 macro F1 in the other. The SHAP rankings of the two contexts are correlated but not interchangeable: Spearman's ρ is 0.83 for Random Forest and 0.79 for XGBoost, RBO is 0.58 and 0.75, and each model keeps three of its five leading features.",
    allow=NEW)
rew(C[1], "How large this disagreement is depends on the reference. Against random halves of one context, cross-context Spearman agreement is lower by 0.13 for XGBoost and 0.19 for Random Forest, and a permutation null that keeps both class mixes produces little or no gap. "
    f"Explaining the random-half models on the same records changes little for XGBoost and removes about {lo_pct:.0f}–{hi_pct:.0f}% of the Random Forest gap. "
    "Keeping related flows in the same half also reduces the gap, measured on each context's own records: to 0.08–0.10 with clusters of about 10 flows, to 0.01–0.04 with clusters of about 35, and to about zero on average at the capture-calibrated setting, where the half-models are weak and opposite gaps in the two contexts cancel. "
    "Blocks of consecutive rows in the original files leave 0.07 and 0.15 at the calibrated setting and 0.03 and 0.09 when each class is split into its first and second halves. At these two settings the gaps remain when both contexts are explained on the same records, and a grouped permutation null shows that row-block grouping produces almost no gap by itself, although the RBO and Top-5 gaps are negative in some repeats of the first-versus-second-half split. For feature clusters no neutral null could be built, so the cluster results leave open whether an excess remains, and the top-weighted RBO gap and the XGBoost family-level gap shrink less under clusters than the Spearman gap does.",
    allow=NEW)
rew(C[2], "Among single features, the clearest change is the rise of the three minimum inter-arrival times. For XGBoost it is mostly a redistribution of Shapley values within the block. For Random Forest the block's combined SHAP share rises, but permuting or removing the block costs less macro F1 under VPN than without it, so the two kinds of measure disagree on whether the model relies on it more. "
    "In practice, the useful distinction is between a trained model and a selected feature list: the model does not transfer, whereas for feature sets of 10 or 15 features a SHAP-selected list carries over at an average macro F1 loss of at most about 0.013 across selection seeds. Because the seeds produced few distinct lists, this test has limited power.",
    allow=NEW)
rew(C[3], "On this benchmark, then, part of the random-split gap between SHAP rankings reflects variation between groups of related flows within a context, and part of it survives the row-order groupings tested, on each context's own records and on common ones. The two contexts are different sets of captures, so none of the remaining difference can be attributed to tunnelling, and the proxies used here cannot measure a true between-session effect. Captures that keep session identifiers, or an independent second capture of both contexts, would settle the question, and newer traffic and class-specific explanations would show whether the pattern holds beyond this benchmark.",
    allow=NEW)
CP[4].getparent().remove(CP[4])

# ---- split two paragraphs that became too long to follow (text unchanged; only a paragraph break)
def split_at(marker):
    p = find_c(marker)
    assert not list(p.iter(q(MN, "oMath"))) or marker in ptext(p)
    # split by runs: everything from the run that starts the marker moves to a new paragraph with the same properties
    full = ptext(p); i = full.index(marker)
    np_ = copy.deepcopy(p); p.addnext(np_)
    def keep(par, start, end):           # keep characters [start, end) of par's text, across runs and equations
        pos = 0
        for el in list(par.iter(q(W, "t"), q(MN, "t"))):
            s_ = el.text or ""; a, b = pos, pos + len(s_); pos = b
            lo, hi = max(a, start), min(b, end)
            el.text = s_[lo - a:hi - a] if lo < hi else ""
            el.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        for om in list(par.iter(q(MN, "oMath"))):
            if not "".join(t.text or "" for t in om.iter(q(MN, "t"))): om.getparent().remove(om)
    keep(p, 0, i); keep(np_, i, len(full))
    for el in p.iter(q(W, "t")):
        pass
    ts = [el for el in p.iter(q(W, "t")) if el.text]
    if ts: ts[-1].text = ts[-1].text.rstrip()
split_at("The grouped null sharpens the row-order end of this range")
split_at("For five grouped designs (clusters of about 10 and 35 flows")


# ---- Table XV: XGBoost permutation-null value rounded once (v13 fixed the same double rounding in Table VII and VI-C)
T15 = table_after("TABLE XV")
def t15_row(label):
    hit = [tr for tr in T15.iter(q(W, "tr")) if ptext(list(tr.iter(q(W, "tc")))[0]).strip().startswith(label)]
    assert len(hit) == 1, label; return hit[0]
T7N = json.load(open(RES / "p8_v13.json"))["table7"]["XGB_null"]
c = list(t15_row("Δρ, permutation null").iter(q(W, "tc")))[2]
pp = [x for x in c.iter(q(W, "p")) if ptext(x).strip()]; assert len(pp) == 1 and ptext(pp[0]).strip() == "-0.002 (0.009)", ptext(pp[0])
set_par(pp[0], f"{T7N['gap']:.3f} ({f3(T7N['sd'])})")
def t15_add(after_label, cells):
    base = t15_row(after_label); new = copy.deepcopy(base)
    for i_, (tc, v) in enumerate(zip(new.iter(q(W, "tc")), cells)):
        pp = [x for x in tc.iter(q(W, "p")) if ptext(x).strip()]
        if i_ == 0 and list(pp[0].iter(q(MN, "oMath"))):      # label cell: keep the Δρ equation, replace the text after it
            assert v.startswith("Δρ"); rs = [r for r in pp[0] if r.tag == q(W, "r")]
            for r in rs[:-1]:
                for t_ in r.iter(q(W, "t")): t_.text = ""
            t_ = rs[-1].find(q(W, "t")); t_.text = v[2:]; t_.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        else:
            set_par(pp[0], v)
    base.addnext(new); return new

NEWROWS = [("Δρ, random halves on common records (range)", f"{f3(rg[0])}–{f3(rg[1])}", f"{f3(xg[0])}–{f3(xg[1])}", "VI-C"),
           ("Δρ, row blocks calibrated / first vs second half, on common records (both test sets)",
            f"{sg(CRB['RF_orderedcal']['gap']['union'])} / {sg(CRB['RF_orderedhalf']['gap']['union'])}",
            f"{sg(CRB['XGB_orderedcal']['gap']['union'])} / {sg(CRB['XGB_orderedhalf']['gap']['union'])}", "VI-C"),
           ("Δρ minus group-level null, row blocks calibrated / first vs second half",
            f"{sg(exc['RF'][0])} / {sg(exc['RF'][1])}", f"{sg(exc['XGB'][0])} / {sg(exc['XGB'][1])}", "VI-C")]
prev_label = "Δρ, row blocks first vs second half"
for row in NEWROWS:
    r_ = t15_add(prev_label, row); prev_label = row[0]
