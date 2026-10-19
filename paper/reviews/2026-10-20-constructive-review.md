---
title: "Constructive Review — SHAP_VPN_revised_v11"
role: constructive
date: 2026-10-20
reporting_standard: REFORMS + reproducibility audit
---

An in-project reading by a separate AI session. It is not an outside review.

Sources read: `paper/reviews/v11_text.txt` (all 1,313 lines), `Concerns2.txt`, `Concerns4.txt`, `revision2/results/*` (ref_summary.json, p1–p5 JSON, p2_topk.csv, p4_matched1330.csv, arff_order_locality.csv, arff_class_counts.csv), `revision2/src/designs.py`, `summarize_refs.py`, `revision2/README.md`. I modified nothing in the repository except to add this file. I ran read-only Python checks in the same environment the paper reports. Python 3.13.16, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0, NumPy 2.5.3 and pandas 3.0.5 are installed, which matches Table III and `p1_results.json["env"]`.

## Summary

The paper asks whether TreeSHAP rankings of RF and XGBoost application classifiers agree between the VPN and Non-VPN parts of ISCX VPN-nonVPN, measured against within-context references of increasing strictness. This revision is honest and careful. Every number I checked matches the results files. The concerns from rounds 2 and 4 are almost all addressed in the text, and the overclaiming of earlier versions ("effect of the VPN", "10 SD", "measurable cost") is gone. The remaining weakness is the one the authors already name as their most serious limitation: whether a cross-context excess survives grouping by session. That question now rests on two unvalidated proxies that disagree. Several cheap analyses using the existing code and outputs would make this part much sharper. The paper also leaves out a few results it already has, and one or two summary sentences hold for one model only.

### Evidence audit (main claims, one sentence each)

| # | Claim | Grade | Reason |
|---|---|---|---|
| C1 | On the primary split, cross-context rankings correlate at ρ = 0.83 (RF) / 0.79 (XGB), with only 3 of the top 5 features shared (Sec. VI-B). | **Strong** | Descriptive and verified exactly (`p1_results.json` cross_RF/cross_XGB). It is stable across TreeSHAP formulation, seeds, depth and sentinel encoding (Table XIV, `p3_results.json`). |
| C2 | Compared with random halves of one context, cross-context agreement is lower by 0.129 (XGB) / 0.186 (RF), and a flow-level permutation null does not reproduce this (Sec. VI-C). | **Strong** for the comparison as stated | All 11 repeats are positive (min 0.072 / 0.143) and the null sits near zero (-0.002 / +0.028). The paper itself shows that this reference is inflated. |
| C3 | Class mix does not explain the gap (Sec. VI-D). | **Moderate–strong** | Four convergent checks: swap, null, class weighting, matched training. The swap uses 3 RF draws, and the matched 1,330-flow check is XGBoost only (see `p4_classmix_miniat.py` line 50), which the text does not say. |
| C4 | Keeping related flows together removes part of the gap, and an excess of roughly 0–0.07 (XGB) / 0–0.15 (RF) probably remains (Sec. VI-C, X-A, abstract). | **Weak–moderate** | Both proxies are unvalidated and give opposite answers. There is no null at the group level. The outer 80/20 split stays random. The "every repeat" evidence comes largely from repeats that reuse the same halves (Major 1–2). |
| C5 | None of the differences can be attributed to tunnelling. | **Strong** | This follows from the design (different capture sets), and the wording is now consistent throughout. |
| C6 | A model trained in one context reaches only 0.43–0.61 macro F1 in the other (Sec. VIII-A). | **Strong as a measurement** | Verified in `p2_results.json`. There is one model per context and no interval. The comparison with cluster-CV is now hedged. |
| C7 | A SHAP-selected Top-10 list transfers with little loss, and the context-match term is small (Sec. VIII-B). | **Moderate** | Numbers verified, and the 2×2 decomposition is a real improvement. The five seeds vary less than they appear to (several lists are identical across seeds), and at k = 15 the lists are nearly identical by construction (Minor 6). |
| C8 | For XGB the minimum-IAT shift is mostly credit moving within the block. For RF the block itself gains about 6 pp (Sec. VI-E). | **Moderate** | Combined shares and drop-one results are verified. The permutation-importance comparison is reported selectively (Major 3). |
| C9 | Family aggregation lowers the gap, but the result depends on the grouping rule (Sec. VII). | **Moderate** | The numbers are verified, but two comparative sentences are wrong for one model each (Minor 1). |
| C10 | The sentinel encoding moves rankings by about as much as random-half retraining does (Sec. IX). | **Moderate–strong** | Verified (S1 vs S2 ρ 0.887–0.943 against within-random 0.928–0.976). The reframing asked for in Concerns2 #5 is done correctly. |

### Where the argument is weakest but most important

The paper's new headline is the bracket "between about zero and 0.07 (XGB) / 0.15 (RF)" and the claim that "some excess probably remains" (VI-C last paragraph, X-A, abstract, XII). This is the claim a reader will take away, and it is the least supported one. It depends on three things:

1. **Row order tracks captures.** This is inferred from neighbour similarity alone. The locality data in `arff_order_locality.csv` also show that the signal is much weaker in VPN than in Non-VPN. The median ratio of lag-1 distance to random distance is 0.77 for VPN against 0.49 for Non-VPN. The win20 ratio is 0.38–0.65 for VPN against 0.30–0.54 for Non-VPN. If row-order blocks are a weaker proxy for VPN captures, block designs probably undercorrect more on the VPN side. The paper does not say this.
2. **A positive Δ in every repeat is evidence of an excess.** No null distribution exists for Δ under grouping (Major 1).
3. **The bracket logic.** If blocks only undercorrect, the tightest upper bound is the strictest block design (0.03 / 0.09), not the calibrated one (0.07 / 0.15). The paper uses the calibrated value as the upper end, presumably because first-vs-second may overcorrect by separating sub-applications. That reasoning needs to be stated.

What would strengthen it, roughly in order of cost:
- per-context one-sided gaps, which need no new runs (Major 1a)
- a permutation null that preserves the grouping (Major 1b)
- a grouped outer split for the row-block designs (Major 1c)
- a locality check that compares how far the row-order similarity extends in each context with the capture counts. For example, does the lag at which the distance ratio approaches 1 scale with the 25–83 against 95–324 flows per capture?

A reader also needs a code and data availability statement (Major 4) and one sentence saying which analyses were added during review rather than planned (see Falsification Test).

## Major Issues

**1. The grouped-reference results can be read much more sharply with analyses the authors can already run.**

*Why it matters.* The averaged gap Δ (Eq. 6) hides what happens under coarse clustering. From `ref_summary.json` means, the one-sided gaps (within minus cross) per context are:

| Design | RF: NV − cross | RF: VPN − cross | XGB: NV − cross | XGB: VPN − cross |
|---|---|---|---|---|
| Random halves | +0.162 | +0.210 | +0.118 | +0.140 |
| Clusters ~140 | +0.093 | **−0.088** | +0.016 | −0.047 |
| Clusters, calibrated | **+0.228** | **−0.193** | +0.067 | −0.035 |
| Row blocks, calibrated | +0.142 | +0.164 | +0.063 | +0.086 |
| Row blocks, first vs second | +0.061 | +0.126 | +0.025 | +0.042 |

The near-zero RF Δ under calibrated clusters (+0.018) averages a large positive gap against the Non-VPN reference and a large negative one against a collapsed VPN reference (within-VPN ρ = 0.296). This supports the authors' own suspicion that coarse clusters overcorrect, and it shows that the "≈ 0" end of the bracket comes mainly from VPN half-models becoming unreliable. It is also more informative than the sentence "this is weak evidence of no gap".

*Where.* Sec. VI-C paragraphs 3–4, Table VII, Sec. X-A, Sec. XII paragraph 2 ("At the coarser resolution… the average gap approaches zero").

*Suggested resolution.*
- (a) Add the two one-sided gaps to Table VII, or as a compact extra table. Read the bracket off the context whose within-reference is still informative.
- (b) Run the permutation null with the same grouping. Pool whole row blocks (or clusters) within class across contexts, reassign whole groups to pseudo-contexts, then apply the same grouped half-split. Right now the null is flow-level only (`designs.py`, `contexts_for_repeat`), so it says nothing about how large Δ gets from grouping noise alone. Under coarse grouping that noise is large (RF SD 0.084–0.111).
- (c) For the row-block designs, also hold out whole blocks for the outer test partition, so that the explained records do not come from captures in either training half. The text already admits that "only the division of the training partition into halves is grouped". This would close that gap with a few lines added to `designs.py`.

**2. "Non-negative in every repeat" is presented as more evidence than it is, and the abstract hides the lowest row-block values.**

*Why it matters.* The abstract ("the gap is non-negative in every repeat, even with first- and second-half splits"), VI-C ("Under every row-order setting the gap is at or above zero in all 11 repeats"), X-A and XII paragraph 3 all use the count of positive repeats as support for a remaining excess. In `designs.py`, `orderedhalf` sets `kc = 2`, so each class is always split into its first and second halves. Across the 11 repeats only the outer 80/20 draw, the subsampling and the seeds change, so the halves are essentially the same every time. The calibrated block design has 4–10 blocks per VPN class (`CAPTURE_FILES["vpn"]`), so it allows few distinct assignments. Eleven positive repeats here are not eleven confirmations. The XGB minimum for the first-vs-second split is 0.0002 (`XGB_orderedhalf.min.rho_gap`), so "non-negative" holds by the thinnest margin. Separately, the abstract reports only the calibrated row-block values ("only to 0.07 and 0.15"). Readers will take those as the floor, but the strictest row-block split gives 0.033 and 0.093.

*Where.* Abstract. Sec. VI-C paragraph 5 and the "Taken together" paragraph. Sec. X-A. Sec. XII paragraph 3.

*Suggested resolution.* State how many distinct half-assignments each block design allows, and describe the repeat count as robustness to resampling rather than independent replication. In the abstract, give the range "0.03–0.07 (XGBoost) and 0.09–0.15 (Random Forest) with blocks of consecutive flows". Spell out the bracket logic in VI-C: which design is taken as the upper reference and why it is not the strictest one.

**3. The permutation-importance comparison leaves out results that change how the minimum-IAT shift and the general framing should be read.**

*Why it matters.* VI-E closes with "Permutation-based measures show smaller shifts than SHAP for both models", and X-B says the minimum IATs "move much less under marginal or conditional permutation importance." Both statements are true for min_fiat and min_biat. The results files, however, also show:
- **Whole-ranking cross-context agreement is lower under permutation importance than under SHAP.** Marginal: ρ = 0.611 (RF), 0.719 (XGB) (`p4_results.json` pi_cross_*). Conditional: 0.657 and 0.575 (`p5_results.json` cpi_cross_*). SHAP gives 0.830 and 0.786. Top-5 overlap is 4/5 for RF under both permutation measures and 3/5 for XGB marginal.
- **Within a single model and context, SHAP and marginal permutation importance disagree strongly.** `rho_vs_shap` is 0.307 for RF Non-VPN, and 0.739, 0.818 and 0.923 for the other three. For RF Non-VPN, switching the attribution method moves the ranking far more (1 − ρ ≈ 0.69) than switching the context (≈ 0.17).

A large share of this probably comes from near-zero and negative permutation values in the tail. The reader should still see it, because it bears directly on the paper's recommendation to add "a secondary importance measure" (X-B). Without it, the sentence in VI-E reads as if permutation measures were more stable overall.

*Where.* Sec. VI-E "Permutation importance" paragraph and closing paragraph. Table IX. Sec. X-B.

*Suggested resolution.* Add a row or a short sentence giving cross-context ρ / RBO / Top-5 for both permutation measures and the SHAP-vs-permutation ρ within each context. Limit the closing sentence of VI-E to the minimum IATs. Note that tail ranks of permutation importance are noisy, and consider reporting top-weighted agreement (RBO), which is less sensitive to that noise.

**4. No code or data availability statement (REFORMS reproducibility items).**

*Why it matters.* The pipeline is in good shape. Every number traces to a script in `revision2/src`, the environment is pinned and recorded in `p1_results.json`, seeds are fixed, and `paper_build/` regenerates the text from the results. None of this is visible to a reader of the manuscript. Two inputs are also not described well enough to reproduce: the capture-file prefix list (attributed to "the Deep Packet code [7]", but [7] is the paper, not the repository) and the ARFF row mapping (`p6_arff_order.py` needs the original Scenario A2 files).

*Where.* There is no availability section. Sec. IV-D "Calibrating the grouping".

*Suggested resolution.* Add a "Code and data availability" statement with a repository or archive link, the exact ARFF files used, and the prefix list with its source URL and commit, or include the list itself as a supplementary table. This is the cheapest high-value change in this review.

## Minor Issues

1. **Family-level comparisons hold for one model only (factual).**
   - Sec. VII: "The eight data-driven clusters produce gaps of 0.096 for Random Forest and 0.099 for XGBoost, larger than the semantic-family gaps." For RF, 0.096 is *smaller* than the six-family summed gap (0.112) and the per-member gap (0.149).
   - Sec. X-C: "per-member means and correlation-based clusters give larger gaps." Per-member is larger for RF (0.149 > 0.112) but smaller for XGB (0.064 < 0.075). Clusters are larger for XGB (0.099 > 0.075) but smaller for RF.
   - Rewrite both sentences per model.
2. **Sec. VII last paragraph and XII paragraph 3 overstate the persistence of the XGB family gap.** It persists at ~10 flows per cluster (0.077) but falls to 0.014 at ~35 and 0.016 under calibrated clusters, which VII itself reports. "Part of the XGBoost difference persists at the family level, even though it shrinks at the level of individual features" should be limited to fine grouping. Likewise, "the top-weighted RBO gap … decrease[s] less under feature-based clustering than the Spearman gap" (XII) holds for XGB (−27% against −39% at ~10). For RF it is about the same (−43% against −47%), and at ~140 the RBO gap drops further (−0.054).
3. **Sec. IX ranges do not match `p3_results.json`.**
   - "Within one model and context, path-dependent and interventional rankings agree at ρ = 0.969–0.984." Extra Trees VPN is 0.960, so the range is 0.960–0.984 if ET is included.
   - "Extra Trees agrees less well with the path-dependent Random Forest (0.709–0.801)." The path-dependent comparisons are 0.730 and 0.801. The 0.709 value is ET-interventional against RF-interventional.
4. **Class-matched check (VI-D, Sec. IV-E).** It is run for XGBoost only. Say so. "Compared to 0.77–0.82 for full-size models" is ambiguous: the primary values are 0.786 (XGB) and 0.830 (RF), and the half-split values are 0.825 and 0.766. Give the XGB-specific comparison.
5. **RF 800-record validation for half-models (IV-D).** `rf800_halfmodels` covers 8 half-models from repeats 0–1 of the *random* design only (ρ ≥ 0.998). The weak half-models of the coarse-cluster designs (macro F1 ≈ 0.53) are where sampling 800 records is most likely to matter. Either extend the check or limit the sentence to the random design.
6. **The selection seeds of the Top-k transfer vary less than "five seeds" suggests (VIII-B, Table XII).** In `p2_topk.csv` the RF VPN-derived Top-10 list gives the same macro F1 (0.866318) in all five seeds, and only three distinct Non-VPN lists occur. At k = 5 the RF Non-VPN list is identical across seeds. At k = 15 the lists share 14–15 of 15 features, so "within 0.002 of zero at k = 15" and the k = 15 part of X-D ("At k = 10 and k = 15 … at most about 0.013") are close to true by construction. Report the number of distinct lists per k, and weight the conclusion on k = 10.
7. **Primary RF cross-context ρ (0.830) lies above all 11 half-split values (0.695–0.804, `RF_random.min/max.rho_cross`)**, while the XGB primary value (0.786) lies below all of its half-split values (0.810–0.861). The model order reverses between the primary fits and the half-splits. One sentence on how agreement depends on training size would prevent confusion, because the conclusion quotes primary values next to half-split gaps.
8. **Capture counts.** `designs.py` assigns one Non-VPN P2P capture as well as one VPN P2P capture (`CAPTURE_FILES["novpn"][4] = 1`). The text says "plus one VPN P2P capture" and, later, "P2P has at most one capture per context". Make these consistent.
9. **Half sizes for row-block designs** are not reported. They fall to 3,264 at ~140 per block (`XGB_ordered140.mean.n`), below the 3,368–3,476 stated for the cluster designs. Smaller halves lower within-context agreement a little, so state the sizes in the Table VII note.
10. **X-A "from about 20% to nearly all of it".** The 18% figure is the calibrated row-block setting for RF. Row blocks of ~10 remove only 10% of the RF gap (0.168 against 0.186). Name the setting, or give the full range.
11. **XGB "redistribution" (VI-E closing paragraph, XII).** A stable combined share is *consistent with* redistribution, and drop-one supports substitution, but a stable sum does not prove it. "Is consistent with a redistribution" is the safer wording.
12. **Conditional permutation [36].** Strobl et al. condition on a partition derived from the forest's splits. Quintile cells of the two most correlated features is a simpler variant, so describe it as "in the spirit of [36]".
13. **Kendall τb [44].** The tie-corrected τb is from Kendall (1945), "The treatment of ties in ranking problems", Biometrika 33(3). The 1938 paper defines τa.

## Editorial

- Abstract (230 words, within typical limits): "achieves at best 0.43–0.61 macro F1" reads awkwardly. Use "reaches only 0.43–0.61". See Major 2 for the row-block range.
- Sec. XII is five paragraphs that largely repeat Sec. X and Table XV. Two paragraphs would do: the findings, then future work. The "Future Work" in the heading currently gets one sentence.
- The "Taken together" paragraph in VI-C and Sec. X-A make the same argument in nearly the same words. Keep one full version and cross-reference it.
- VI-B: "Spearman correlation is 0.60 (Non-VPN) and 0.60 (VPN)" can be shortened to "0.60 in both contexts".
- Table IX: in the text extraction the "Permutation ranks marginal / conditional" column holds two triplets per row. Check that the Word table shows two labelled sub-columns.
- 15 tables and 8 figures. Table XIV (sensitivity) and the full Table XII could move to a supplement, since Table XV already summarises them.
- Table XII header: "Δ k = 15 mean" has no [min, max] while the k = 5 and k = 10 columns do. Either add it or note the omission in the table note.

## Methodological Assessment

**REFORMS checklist (condensed)**

| Module | Status | Notes |
|---|---|---|
| Study goals and population | Met | The question, both contexts, the 7-class target and that the context is not predicted are all stated (Sec. I, III-A). |
| Data provenance and preprocessing | Met | Exact files, deduplication counts, label conflicts, the BROWSING/P2P caps (verified against `arff_class_counts.csv`) and sentinel handling are all given. |
| Data splitting and leakage | Mostly met | The leakage audit is thorough. Feature selection happens inside the training partition. The main residual problem is session dependence, which is acknowledged (Sec. XI) and is the topic of Major 1. |
| Modelling | Met | Hyperparameters, seeds, untuned status and a single pinned environment are reported. The environment was verified. |
| Metrics and uncertainty | Mostly met | Effect sizes with SD and range, an explicit non-independence caveat, and no t-like ratios. Interval scope is stated (explained records only). Missing: a null distribution for Δ under grouping (Major 1b). |
| Generalisability and limitations | Met | Section XI is candid and specific. |
| Reproducibility | Partially met | The pipeline is fully scripted and regenerates the text, but the manuscript has no availability statement and the prefix-list source is not cited properly (Major 4). |

**Computational reproducibility audit.** Installed versions match Table III exactly. The results files carry an `env` record. `revision2/README.md` maps every script to a paper section and explains how the docx is rebuilt from the results, so numbers are not hand-copied. Model artefacts (`ref_*.pkl`) store per-record |SHAP| for every half-model, which allows re-aggregation without refitting. I did not refit any model. The note "XGBoost was run single-threaded; refitting with two threads gave identical predictions" is reassuring, and the README states it too.

**Cited-evidence spot-check (five references; checked against my knowledge of the sources, nothing fetched in this session):**

| Ref | Attributed claim | Verdict |
|---|---|---|
| [34] Hooker, Mentch & Zhou 2021 | Unrestricted permutation evaluates the model off the data distribution | Accurate |
| [45] Webber et al. 2010 | With p = 0.9 the first 10 ranks carry about 86% of RBO weight (Eq. 21) | Accurate |
| [36] Strobl et al. 2008 | Conditional permutation importance | Accurate in spirit. The paper's quintile-cell variant differs from Strobl's partition (Minor 12). |
| [44] Kendall 1938 | Kendall's τb with tie correction | Partially accurate. τb is from Kendall 1945 (Minor 13). |
| [7] Lotfollahi et al. 2020 | Source of the public capture-file prefix list | Partially accurate. The list is in the associated code repository, not the paper (Major 4). |

## Number spot-checks

| Claim (location) | Manuscript value | Results field | Match? |
|---|---|---|---|
| Primary cross-context ρ / τb RF (VI-B, Table VI) | 0.8300 / 0.6443 | `p1_results.json` cross_RF.rho = 0.83004, tau = 0.64427 | Yes |
| Primary RBO / L1 XGB (Table VI) | 0.755 / 0.360 | cross_XGB.rbo = 0.75485, l1 = 0.36049 | Yes |
| κ10 RF (Table VI) | 0.82 | cross_RF.kappa10 = 0.8231 | Yes |
| Cluster-grouped CV macro F1 RF Non-VPN (III-C) | 0.901 → 0.475 | cv_RF_novpn_strat = 0.9013, cv_RF_novpn_grouped = 0.4748 | Yes |
| Random-halves Δρ XGB (VI-C, abstract) | +0.129 (SD 0.028), range 0.072–0.163 | `ref_summary.json` XGB_random mean.rho_gap = 0.1290, sd = 0.0281, min 0.0721, max 0.1633 | Yes |
| Random-halves Δρ RF | +0.186 (SD 0.036) | RF_random mean.rho_gap = 0.1864, sd = 0.0358 | Yes |
| Permutation null Δρ RF | +0.028, range −0.010 to 0.052 | RF_null mean.rho_gap = 0.0276, min −0.0104, max 0.0516 | Yes |
| Calibrated clusters within-VPN ρ RF (VI-C) | 0.296 | RF_calibrated mean.rho_wvpn = 0.2965 | Yes |
| Row blocks calibrated Δρ, % removed (VI-C) | 0.074 / 0.153; 43% / 18% | XGB_orderedcal 0.0741, RF_orderedcal 0.1528; 1 − 0.0741/0.1290 = 42.6%, 1 − 0.1528/0.1864 = 18.0% | Yes |
| First-vs-second Δρ XGB, range (VI-C) | 0.033 (0.000–0.082) | XGB_orderedhalf mean 0.0329, min 0.0002, max 0.0818 | Yes |
| Summed-family gap, row blocks (VII) | +0.016 to +0.116 | min RF_orderedhalf fam_gap 0.0156, max RF_ordered35 0.1156 | Yes |
| Class-mix swap change, RF Non-VPN (VI-D) | −0.035 | `p4_results.json` classmix_summary RF/novpn rho_other: 0.7648 − 0.7994 = −0.0346 | Yes |
| Matched 1,330 within/cross (VI-D) | 0.946 / 0.945; 0.853 / 0.857 | matched1330.mean within_nat_novpn 0.9458, within_nat_vpn 0.9448, cross_nat 0.8532, cross_mat 0.8573 | Yes (XGB only, not stated) |
| RF minimum-IAT block share (VI-E, Table IX) | 17.0 / 23.3; half-split 16.3 / 24.5 | `p5_results.json` block_RF_novpn.block = 17.04, block_RF_vpn = 23.33; block_half_RF_* 16.28 / 24.55 | Yes |
| VPN RF drop min_fiat share gains (VI-E) | +2.2 (min_flowiat), +1.5 (min_biat) pp | drop_RF_vpn_min_fiat: 9.083 − 6.842 = 2.24; 9.520 − 7.974 = 1.55 | Yes |
| Model transfer F1 (VIII-A) | XGB 0.430 / 0.613; RF 0.438 / 0.606 | `p2_results.json` transfer_XGB_novpn_to_vpn 0.4301, vpn_to_novpn 0.6126; RF 0.4379 / 0.6060 | Yes |
| Top-10 own − transferred, VPN RF (VIII-B, Table XII) | +0.013 [+0.007, +0.017]; CI [0.006, 0.028] | `p2_topk.csv` RF/vpn/k = 10 diff mean 0.0128, min 0.0072, max 0.0167; seed 0 ci_diff [0.0057, 0.0283] | Yes |
| Context-match term k = 10 (Table XIII) | RF +0.008, XGB +0.010 | `p5_results.json` topk_2x2 match RF/10 0.0084, XGB/10 0.0095 | Yes |
| Beat-random shares (VIII-B) | own 80–100%, transferred 87–100% | own_pct_random means 0.80 / 1.00 / 0.92 / 1.00; tr_pct_random 0.90 / 0.89 / 0.87 / 1.00 | Yes |
| S1 vs S2 within-context ρ (Table XV) | RF 0.914 / 0.887, XGB 0.942 / 0.943 | `p3_results.json` sent_RF_*_S1vsS2 0.9140 / 0.8874; sent_XGB_* 0.9417 / 0.9427 | Yes |
| Path vs interventional agreement (IX) | 0.969–0.984 | rob_*_path_vs_interv: RF 0.975 / 0.969, XGB 0.984 / 0.982, **ET 0.970 / 0.960** | Partial (ET VPN 0.960 falls outside the range) |
| ET vs path-dependent RF (IX) | 0.709–0.801 | rob_ET_vs_RF_novpn_path 0.730, vpn_path 0.801 (0.709 is interv vs interv) | Partial (mislabelled) |
| Capture-count calibration (IV-D) | 25–83 (median 44) NV; 95–324 (median 246) VPN | `designs.py` CAPTURE_FILES and ARFF_COUNTS: NV 25.4–83.1, median 44.5; VPN 95.0–324.4, median 245.5 | Yes |
| Row-order locality ratio (IV-D) | 0.30–0.65 | `arff_order_locality.csv` win20 ratio min 0.299, max 0.647 | Yes |

Twenty-four checks: 22 match exactly and 2 are partial (both in Sec. IX). I found no number that contradicts a results file.

## Concerns coverage

**Concerns4**

| Concern | Addressed? | Where |
|---|---|---|
| 1. Calibrate the session proxy (flows per capture) | Yes | IV-D "Calibrating the grouping", calibrated cluster and block designs, Table VII |
| 1a. Coarsest setting is weak evidence | Yes | VI-C ("this is weak evidence of no gap … SD 0.111") |
| 1b. Remove overcorrecting wording; use "cannot be attributed to tunnelling" | Yes | Abstract, VI-C, VI-E, X-A, XII. "Should not be read as an effect of the VPN" no longer appears. |
| 2. Block-summed min-IAT shares; temper RF | Yes | Table IX column, VI-E "Combined share" and closing paragraph |
| 2b. Permutation importance not neutral; grouped or conditional check | Yes, but reported selectively | VI-E with [34] and a conditional variant; see Major 3 |
| 3. Read list transfer as a 2×2; report interaction; soften "measurable cost" | Yes | IV-G, VIII-B, Table XIII. "Measurable cost" removed. |
| 4. XGB family gap does not shrink; limit the abstract | Yes | VII last two paragraphs, X-C. The abstract no longer mentions families. Minor 2 notes it does shrink at ~35. |
| 5. Hedge the model-transfer comparison | Yes | VIII-A ("only indicative … protocols differ"), XI |
| Minor: revision history in text | Yes | None of the flagged phrases ("previous version", "earlier version", "earlier reading") remain |
| Minor: abstract length | Yes | 230 words, about 8 numbers |
| Minor: class-mix SDs and range mismatch | Yes | Table VIII now shows SDs. The text range 0.924–0.977 (last column) and 0.936–0.986 (own-primary column) both match. |
| Minor: RF 800 records for half-models | Partly | IV-D gives ρ ≥ 0.998 for 8 half-models, random design only (Minor 5) |
| Minor: Table XI bottom-10 range | Yes | Text 0.685–0.735 matches Table XII |
| Minor: family-share table | Yes | Table X |

**Concerns2 (main points)**

| Concern | Addressed? | Where |
|---|---|---|
| 1. Within-context reference inflated; grouped halves; permutation null; PCAP rebuild | Mostly | Grouped and row-block references plus null (IV-D, VI-C). The PCAP rebuild is declined with a reason (III-C, XI). A null at the group level is still missing (Major 1b). |
| 2. Remove "10 SD"; RF n = 3 caveat | Yes | IV-I ("ratios … not test statistics, and we do not report them"). Both models now have 11 repeats. |
| 3. Consequence test (k choice, random controls, beat-random table, abstract wording, model transfer) | Yes | k = 5/10/15 for both models, 20 shared random lists, beat-random column in Table XII, direct transfer in VIII-A |
| 4. Single software environment | Yes | Table III note. Verified against the installed versions and `p1_results.json`. |
| 5. Sentinel conclusion overstated; check the 0.8281 duplicate | Yes | IX reframed against random-half variation. The identical ρ is explained with differing τb and Top-5. |
| 6. Class-mix: direct swap, same-size reference, BROWSING cap | Yes | VI-D, Table VIII, III-B (cap verified in `arff_class_counts.csv`) |
| 7. Min-IAT: 0.60 bound, entries below −1, drop-one or permutation | Yes | VI-B (0.60 reported), VI-E (four checks) |
| Minors (naming, label conflicts, κk, family design, VPN steadiness, LR baseline, log(x+2), citation order, length) | Yes, length only partly | "Primary partitions", III-A label conflicts, κk in Table VI, alternative grouping, VI-F, LR dropped, log(x+2), citations in order of first appearance. Length is still 15 tables and 8 figures (Editorial). |

## Falsification Test

The paper states refutation criteria for its secondary claims and engages them honestly:
- **Class mix.** "If class mix drove the cross-context disagreement, the swapped models should move toward the other context's ranking" (IV-E, Table VIII note). The swap does not do this, and RF Non-VPN moves away (−0.035).
- **Half-split artefact.** The permutation null is set up to reproduce the gap if the procedure plus class mix were enough. It does not for XGB (−0.002), and only 0.028 of RF's 0.186 is reproduced.
- **Credit redistribution.** The combined-share test is framed as a test ("If the shift were only a redistribution … the summed share would stay about the same"), and the paper accepts the negative result for RF.

For the **headline claim** that an excess probably remains after grouping, no criterion is set in advance. The paper names what would settle the question (a capture split or an independent capture, Sec. XI), but it does not say what result from its own designs would have refuted the claim. In practice, a gap near zero under the row-block designs would have been read as "blocks undercorrect" and a positive gap as "excess remains", so the claim cannot fail on the current evidence. The eleven reference designs were also added over successive revisions in response to review, which a reader cannot tell from the standalone text.

Suggested fix: add one sentence of the form "We would conclude that no excess remains if Δ under the row-block designs fell within the spread of a group-level permutation null (Major 1b)". Then run that null, and add a sentence in IV-D saying which designs were planned and which were added later. If the group-level null's spread covers 0.03–0.09, the "excess probably remains" sentences should become "cannot be distinguished from grouping noise".
