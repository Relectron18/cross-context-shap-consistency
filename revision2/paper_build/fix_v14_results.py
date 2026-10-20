# Part B of fix_v14.py (executed inside it): the two new analyses. Every number is read from
# results/p11_v14.json (src/p11_v14_summary.py). Uses rep / rew / guard / find_c / body from fix_v14.py.
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
rep("Under this null, Δ shows what the half-split procedure and the class-mix difference produce when nothing else separates the two contexts.",
    "Under this null, Δ shows what the half-split procedure and the class-mix difference produce when nothing else separates the two contexts. For five grouped designs (clusters of about 10 and 35 flows, calibrated clusters, calibrated row blocks and the first-versus-second-half split) we build the null in two ways, because neither is neutral on its own. In the flow-level version, flows are reassigned one by one as above and the grouped design is then applied within each pseudo-context. Flows of one capture then sit in both pseudo-contexts, so the pseudo cross pairs share related flows that the grouped within pairs do not, which pushes Δ down. In the group-level version, whole capture-sized groups move instead: within each class and source file, the pooled flows are divided into the calibrated clusters (for the cluster designs) or the calibrated row blocks (for the row-block designs), and each group is assigned at random to one pseudo-context so that each receives about its original share of the class; because whole groups move, class counts are matched only approximately. Calibrated clusters are regions of feature space, so for the cluster designs this version also gives the two pseudo-contexts different feature distributions, which pushes Δ up; row blocks are not built from the features. In both versions the grouped design runs separately on the flows of each source file, so that both halves hold the same mix of Non-VPN and VPN rows.",
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
    "The null design pools the flows of each class across contexts and reassigns them at random, without replacement, to pseudo-contexts with the original class counts. Rows marked Null (flows) apply a grouped design to those pseudo-contexts; rows marked Null (groups) build the pseudo-contexts from whole calibrated clusters or row blocks first (Section IV-D).")

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
rep("Allocating flows at random between two pseudo-contexts with the same Non-VPN and VPN class mixes gives a Δ of -0.001 for XGBoost (range -0.019 to 0.009) and +0.028 for Random Forest (range -0.010 to 0.052).",
    "Allocating flows at random between two pseudo-contexts with the same Non-VPN and VPN class mixes gives a Δ of -0.001 for XGBoost (range -0.019 to 0.009) and +0.028 for Random Forest (range -0.010 to 0.052)."
    f" Nor does the gap come from explaining the two contexts on different records. When every half-model is explained on the same records, Δ is {f3(xg[0])}–{f3(xg[1])} for XGBoost and {f3(rg[0])}–{f3(rg[1])} for Random Forest across the three common record sets (the Non-VPN records, the VPN records and both together), against {f3(own['XGB'])} and {f3(own['RF'])} when each model is explained on its own context's records, and it is positive in all 11 repeats for both models."
    f" For XGBoost the explained records make no material difference. For Random Forest they account for about {100*share_lo:.0f}–{100*share_hi:.0f}% of the gap, and the RBO gap falls from {f3(CR['RF']['rbo_gap']['own'])} to {f3(rb[0])}–{f3(rb[1])}."
    f" The primary models agree: on common records their cross-context ρ is {f3(prim['RF'][0])}–{f3(prim['RF'][2])} for Random Forest and {f3(prim['XGB'][0])}–{f3(prim['XGB'][2])} for XGBoost, against 0.830 and 0.786 on their own test partitions.",
    allow=NEW)

# ---- helpers for new paragraphs
def new_par_after(anchor_text, text):
    a = find_c(anchor_text); p = copy.deepcopy(a); a.addnext(p); rewrite(p, [text]); return p

# ---- VI-C: grouped null (result); claims are asserted against the numbers before they are written
NG, NS = GN["nullgroup"], GN["nullstrat"]
ROWD, CLUD = ("orderedcal", "orderedhalf"), ("grouped10", "grouped35", "calibrated")
row_nulls = [V[f"{m}_{d}"]["null"]["rho_gap"] for V in (NG, NS) for m in ("XGB", "RF") for d in ROWD]
assert max(abs(x) for x in row_nulls) < 0.03, row_nulls                     # "close to zero"
exc = {m: [NG[f"{m}_{d}"]["real_minus_null"] for d in ROWD] for m in ("XGB", "RF")}
assert all(x > 0 for v in exc.values() for x in v)
above = [NG[f"{m}_{d}"]["real_repeats_above_null_max"] for m in ("XGB", "RF") for d in ROWD]
cl_flow = [NS[f"{m}_{d}"]["null"]["rho_gap"] for m in ("XGB", "RF") for d in CLUD]
cl_grp = [NG[f"{m}_{d}"]["null"]["rho_gap"] for m in ("XGB", "RF") for d in CLUD]
cl_real = [NG[f"{m}_{d}"]["real_gap"] for m in ("XGB", "RF") for d in CLUD]
between = all(NS[f"{m}_{d}"]["null"]["rho_gap"] < NG[f"{m}_{d}"]["real_gap"] < NG[f"{m}_{d}"]["null"]["rho_gap"] for m in ("XGB", "RF") for d in CLUD)
assert max(cl_flow) < 0 < min(cl_grp), (cl_flow, cl_grp)                     # the two versions lie on either side of zero
sd_grp = max(NG[f"{m}_{d}"]["null_gap_sd"] for m in ("XGB", "RF") for d in CLUD)
real_within_grp = all(NG[f"{m}_{d}"]["null_gap_min"] < NG[f"{m}_{d}"]["real_gap"] < NG[f"{m}_{d}"]["null_gap_max"] for m in ("XGB", "RF") for d in CLUD)
assert real_within_grp
new_par_after("(RBO in 1 of 11 for XGBoost and 2 of 11 for Random Forest, Top-5 in 3 and 2).",
    "The grouped null separates the two proxies (Table VII). For row blocks, both versions of the null give a mean Δ close to zero "
    f"({min(row_nulls):.3f} to {max(row_nulls):.3f}), so this grouping produces almost no gap of its own, and the real row-block gaps lie above it: "
    f"against the group-level null by {f3(exc['XGB'][0])} and {f3(exc['XGB'][1])} for XGBoost and by {f3(exc['RF'][0])} and {f3(exc['RF'][1])} for Random Forest "
    f"at the calibrated and first-versus-second-half settings, with {min(above)}–{max(above)} of the 11 real repeats above the largest null repeat. "
    f"For clusters, the two versions point in the directions expected from their construction: the flow-level null gives {min(cl_flow):.3f} to {max(cl_flow):.3f} "
    f"and the group-level null {f3(min(cl_grp))} to {f3(max(cl_grp))}, with a spread across repeats of up to {f3(sd_grp)}"
    + (f", and the real cluster gaps ({f3(min(cl_real))} to {f3(max(cl_real))}) lie between them. " if between else
       f". The real cluster gaps ({f3(min(cl_real))} to {f3(max(cl_real))}) lie inside the range of the group-level null. ")
    + "No neutral null could be built for feature clusters, so the cluster results do not show whether an excess remains once related flows are kept together.")

# ---- VI-C "Taken together": what the two new checks add
lo_pct, hi_pct = 100 * share_lo, 100 * share_hi
rep("These are descriptive comparisons, not a test of that excess: the permutation null was run only for random halves, where it already gives +0.028 for Random Forest, no grouped design keeps the test records apart from the training sessions, and within-context pairs are explained on shared test records while cross-context pairs are not. Part of the random-split gap therefore reflects variation between groups of related flows, and part of it is retained under grouping by row order.",
    "The grouped null sharpens the row-order end of this range: row-block grouping produces almost no gap by itself, so the positive row-block gaps are excess over the null, whereas for clusters the null itself is uncertain. "
    f"Explaining both contexts on the same records leaves the XGBoost gap in place and removes about {lo_pct:.0f}–{hi_pct:.0f}% of the Random Forest gap under random halves, so part of the Random Forest excess may reflect the explained records. "
    "None of this tests the excess at the level of captures: no grouped design keeps the test records apart from the training sessions, and both proxies only approximate capture boundaries. "
    "Part of the random-split gap therefore reflects variation between groups of related flows, and part of it survives every row-order grouping.",
    allow=NEW)

# ---- X-A
rep("Together they suggest that part of the random-split gap reflects variation between groups of related flows, from about 20% to nearly all of it depending on the proxy and model, and that some excess probably remains. That excess is described, not tested: the permutation null covers only random halves, the outer split is random in every design, and within-context and cross-context pairs are explained on different kinds of test records. Without capture identifiers or an independent second capture, the size of that excess cannot be quantified, and because the contexts are different sets of captures it cannot be attributed to tunnelling.",
    "Together they suggest that part of the random-split gap reflects variation between groups of related flows, from about 20% to nearly all of it depending on the proxy and model. "
    f"Two further checks narrow this. Explaining both contexts on the same records leaves the XGBoost gap unchanged and lowers the Random Forest gap by about {lo_pct:.0f}–{hi_pct:.0f}%, so the choice of explained records is not the main source of the disagreement. "
    "A grouped permutation null shows that row-block grouping produces almost no gap by itself, so the positive row-block gaps are excess over the null; for clusters, the two versions of the null fall on either side of the observed gap, and the cluster results leave the question open. "
    "Some excess therefore remains under the grouping least tied to the features, but its size depends on how closely row blocks follow captures, and without capture identifiers or an independent second capture it cannot be pinned down. Because the contexts are different sets of captures, it cannot be attributed to tunnelling.",
    allow=NEW)

# ---- X-E: one practical consequence of the common-records check
rep("When explanations drive a decision, such as which features to keep, that decision should be tested on held-out data from the target context.",
    f"Rankings from two contexts should be computed on the same records, or the effect of the records measured, because here the explained records alone accounted for about {lo_pct:.0f}–{hi_pct:.0f}% of the Random Forest gap. When explanations drive a decision, such as which features to keep, that decision should be tested on held-out data from the target context.",
    allow=NEW)

# ---- XI
rep("Only the division of training data into halves is grouped; the outer 80/20 split is random, and the permutation null was run only for random halves. Within-context pairs are explained on shared test records and cross-context pairs are not; a comparison on a common set of explained records is left for future work.",
    "Only the division of training data into halves is grouped; the outer 80/20 split is random. The grouped null has no neutral form for clusters: reassigning single flows puts flows of one capture in both pseudo-contexts, and reassigning whole clusters makes the pseudo-contexts differ in feature space. For row blocks both versions agree, but they inherit the uncertainty about how well row blocks follow captures. "
    "Within-context pairs are explained on shared test records and cross-context pairs, in the designs themselves, are not; explaining both contexts on common records showed that this accounts for part of the Random Forest gap under random halves, so the Random Forest gaps of the grouped designs, which were not recomputed on common records, may slightly overstate the difference between the models.",
    allow=NEW)

# ---- abstract: last sentence (kept within 250 words; checked below)
rep("Part of the random-split gap is explained by related flows and part probably remains, although no design tests that remainder directly, and since the two contexts are different sets of captures, none of the differences can be attributed to tunnelling.",
    "Part of the random-split gap reflects related flows; the rest persists when both contexts are explained on the same records and exceeds a grouped permutation null under row-block grouping, but since the contexts are different sets of captures, none of it can be attributed to tunnelling.",
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
    f"Explaining every model on the same records changes little for XGBoost and removes about {lo_pct:.0f}–{hi_pct:.0f}% of the Random Forest gap. "
    "Keeping related flows in the same half reduces the gap further: to 0.08–0.10 with clusters of about 10 flows, to 0.01–0.04 with clusters of about 35, and to about zero on average at the capture-calibrated setting, where the half-models are weak and opposite gaps in the two contexts cancel. "
    "Blocks of consecutive rows in the original files leave 0.07 and 0.15 at the calibrated setting and 0.03 and 0.09 when each class is split into its first and second halves, and a grouped permutation null shows that row-block grouping produces almost no gap by itself. For feature clusters no neutral null could be built, so the cluster results leave open whether an excess remains.",
    allow=NEW)
rew(C[2], "Among single features, the clearest change is the rise of the three minimum inter-arrival times. For XGBoost it is mostly a redistribution of Shapley values within the block. For Random Forest the block's combined SHAP share rises, but permuting or removing the block costs less macro F1 under VPN than without it, so the two kinds of measure disagree on whether the model relies on it more. "
    "In practice, the useful distinction is between a trained model and a selected feature list: the model does not transfer, whereas for feature sets of 10 or 15 features a SHAP-selected list carries over at an average macro F1 loss of at most about 0.013 across selection seeds. Because the seeds produced few distinct lists, this test has limited power.",
    allow=NEW)
rew(C[3], "On this benchmark, then, part of the random-split gap between SHAP rankings reflects variation between groups of related flows within a context, and part of it survives every row-order grouping. The two contexts are different sets of captures, so none of the remaining difference can be attributed to tunnelling, and the proxies used here cannot measure a true between-session effect. Captures that keep session identifiers, or an independent second capture of both contexts, would settle the question, and newer traffic and class-specific explanations would show whether the pattern holds beyond this benchmark.",
    allow=NEW)
CP[4].getparent().remove(CP[4])
