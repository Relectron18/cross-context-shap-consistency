---
title: "Adversarial Review — SHAP_VPN_revised_v11"
role: adversarial
date: 2026-10-20
reporting_standard: REFORMS + reproducibility audit
---
An in-project reading by a separate AI session. It is not an outside review.

Sources read: `paper/reviews/v11_text.txt` (all 1,314 lines), `Concerns2.txt`, `Concerns4.txt`, `revision2/README.md`, every script in `revision2/src/` plus `revision/src/common.py`, and the result files listed in the spot-checks. I ran read-only checks in a scratchpad outside the repository: (a) recomputed the row-order locality on the two original ARFF files, with and without exact duplicates, and correlated row position with each feature; (b) refit the primary XGBoost and Random Forest models and explained each model on both contexts' test partitions. No repository file was changed. Section and line references below use the line numbers of `v11_text.txt`.

## Summary

The paper compares global TreeSHAP rankings of RF and XGBoost models trained separately on ISCX VPN and Non-VPN flows. It uses eleven within-context reference designs, a class-mix permutation null, and two consequence tests. Almost every number checks out against the result files, and the descriptive core holds: cross-context Spearman agreement is about 0.13 (XGB) and 0.19 (RF) lower than agreement between random halves of one context, and the class-mix null does not reproduce that gap. The interpretive claims built on top of this are weaker than the abstract and conclusion suggest:

- The claim that "part [of the gap] is likely to remain" rests on the row-order proxy, and that proxy is validated only at a range of a few rows.
- Within-context and cross-context pairs are explained on different kinds of record sets.
- The 11 repeats share every capture, so they say nothing about session-level variation.
- The "Top-10 list transfers with little loss" result comes from a test that could hardly have found a loss.

## Major Issues

**1. Neither session proxy is validated at the scale that matters, and the row-order inference is not supported by the locality diagnostic.**
*Why it matters.* Under the row-block proxy the gap stays positive, and that is the only basis for the abstract's "part is likely to remain" (l. 6) and for the bounds of "about zero to 0.07 / 0.15" (l. 455, 1230).
*Location.* IV-D, "Row-order blocks" (l. 258): "Neighbouring rows are much more alike than random rows… the similarity fades over tens to a few hundred rows. This suggests that the files keep flows from the same capture close together."
*Problems found.*
- (a) **The evidence is short-range only.** In `arff_order_locality.csv` the window statistic (0.30–0.65) is a minimum over 40 neighbours. The lag-based ratios tell a different story. For VPN FT, MAIL, P2P and VOIP the distance between *adjacent* rows (lag 1) is 0.77–0.99 of a random pair's distance, so adjacent rows are hardly more alike than random ones. For Non-VPN STREAMING and BROWSING the similarity is gone by lag 5–10 (0.86–0.99). VPN VOIP is 0.99 at lag 1 but 0.08 at lag 10, which looks like round-robin interleaving of concurrent flows, not capture contiguity. Captures hold 25–324 flows (l. 257). At lags of 50–200 rows the ratios are 0.45–1.21, mostly 0.8–1.0, so the diagnostic says nothing about contiguity at capture scale. "Fades over tens to a few hundred rows" fits Non-VPN VOIP and P2P but not most VPN classes.
- (b) **Blocks do depend on the features.** I computed |Spearman| between row position and each feature, within class. It reaches 0.62 (Non-VPN FT, std_flowiat), 0.65 (Non-VPN P2P, total_biat), 0.63 (VPN BROWSING, min_flowiat), 0.63 (VPN P2P, flowPktsPerSecond) and 0.55–0.56 (VPN MAIL, STREAMING). Features drift steadily through the file. A first-versus-second-half split therefore also separates regions of feature space, which is the same objection the paper raises against clusters (l. 455: "Blocks… divide flows by their position in the files, not by their features").
- (c) **The diagnostic includes duplicates the analysis removed.** p6 computes it on the raw ARFF rows, including the 271/413 exact duplicates. Most duplicates sit next to each other: by my count 187 of the Non-VPN and 276 of the VPN duplicated rows are adjacent to an identical row. After deduplication, my recomputation gives a VPN P2P window ratio of 0.73 (raw: 0.47), outside the stated 0.30–0.65 range. This is one random draw; the statistic moved by about 0.08 between draws for VPN MAIL.
- (d) **No within-class grouping can separate captures for BROWSING or P2P.** By the paper's own account, BROWSING "has no dedicated captures" and P2P has "at most one capture per context" (l. 257). Together they make up 39.6% of Non-VPN and 33.8% of VPN training flows (Table II). Even the "strictest" first-versus-second-half split cuts the single P2P capture in two, and BROWSING flows may share captures with other classes (unverified: the dataset does not document where BROWSING flows came from). The calibrated design forces at least four groups, so VPN MAIL (2 captures) and both P2P classes are split within captures by construction.

*Resolution.* Report the lag profiles and the position–feature correlations. Recompute the diagnostic on deduplicated rows. Run the row-block designs without BROWSING and P2P as a sensitivity check. Stop calling the two proxies "bounds" (l. 455); they are two uncalibrated proxies with opposite failure modes. Make the abstract conditional: "under a row-order proxy whose relation to captures is unverified, a gap of 0.03–0.15 remains".

**2. No design holds out sessions from the explained records, and the grouped designs have no matching null.**
*Location.* l. 256 ("the outer 80/20 partition remains a random stratified split"), l. 455 and l. 1242, against the abstract (l. 6) and conclusion (l. 1253–1255), which read the grouped Δ as the excess that remains.
*Problems found.*
- **The test set shares captures with both halves.** In every design the explained test records come from the same captures as both training halves (`designs.py`: `train_test_split(..., random_state=1000 + r)` before grouping). So the grouped Δ is not an out-of-session quantity.
- **The null was run only under random halves.** There is no grouped or row-block null. The proper zero point for a grouped Δ is the same grouped procedure applied to pseudo-contexts, and that is unknown.
- **The null baseline is never subtracted.** The RF null Δ is +0.028 (range −0.010 to 0.052), about 15% of the RF random gap. The excess figures quoted ("0.03 and 0.09", "about zero to 0.15") do not subtract it. RF orderedhalf minus the null is about 0.065. RF calibrated clusters minus the null is negative.

*Resolution.* With row blocks this is cheap to fix. Assign whole blocks to three parts (half A, half B, test) so the explained records are block-disjoint from both halves. Run the null with groups: pool whole blocks across contexts within class and reassign them. Report Δ_design − Δ_null.

**3. Cross-context pairs are explained on different data; within-context pairs on the same data.**
*Location.* IV-B (l. 237: "Explanations are always computed on the held-out test partition of the context in question"), IV-D (within compares two halves on one test set; cross pairs each model with its own context's test set), and the research question in l. 13 ("whether models lean on similar flow features when making the same predictions").
*Problem.* Global mean |SHAP| depends on the distribution of the explained records as well as on the model. The two contexts' test sets differ in class mix and in feature distribution. Δ therefore contains a pure data-shift component that has nothing to do with what the model relies on. The class-weighted variant handles only class mix.
*Reviewer check.* I refit the primary models and crossed model with explained data (single fits; RF explained on 600 random test rows). My refit reproduced the paper's XGB primary cross values exactly: ρ 0.786, τ 0.581, RBO 0.755, L1 0.360.

| Model | Same model, other explained data | Same explained data, other model |
|---|---|---|
| XGB | ρ = 0.997 (Non-VPN model), 0.977 (VPN model) | ρ = 0.787 (VPN test set), 0.771 (Non-VPN test set) |
| RF | ρ = 0.974, 0.978 | ρ = 0.891 (VPN test set), 0.863 (Non-VPN test set) |

For XGB the model difference dominates. For RF, changing only the explained data costs 1 − ρ ≈ 0.02–0.03, the same size as the RF null Δ and about 10–15% of the RF random-half gap. The comparison is also asymmetric: within pairs never carry this component and cross pairs always do.
*Resolution.* Report this model × explained-data 2×2 for the primary fits and for the half-models. Explain every half-model on a common record set: each context's test set, or a pooled set. Recompute Δ on that common set.

**4. The repeats are not replications, and "non-negative in every repeat" is used as evidence in the abstract.**
*Location.* Abstract (l. 6: "the gap is non-negative in every repeat, even with first- and second-half splits"), l. 454, l. 1230, l. 1253.
*Problems found.*
- **The repeats share everything that matters.** All 11 repeats of every design draw from the same captures (30 VPN capture files in the paper's count). Any two training partitions share about 64% of flows, and the outer split seeds (`1000 + r`) are the same in every design. Variation over repeats is partition and seed noise. The session-level variance that a claim about "captures" needs is zero by construction.
- **The count is not evidence.** "11 of 11 positive" is therefore not 11 confirmations. Section IV-I says the right thing, but the abstract and conclusion use the count rhetorically anyway.
- **The claim holds for Spearman only.** In orderedhalf the RBO gap is negative in 1 XGB and 2 RF repeats, and the Top-5 gap reaches −0.5 (XGB) and −0.25 (RF) (`ref_summary_*_orderedhalf.csv`).
- **The analysis path forked across revisions.** Eleven designs and several metrics were added in response to reviewers: cluster sizes, then calibration, then row blocks. No rule was fixed in advance for which proxy decides the question.

*Resolution.* Remove "in every repeat" from the abstract, or say "for Spearman ρ, across 11 dependent repeats". Estimate uncertainty with a block bootstrap over row blocks or capture-sized groups. State which result would count against the hypothesis (see Falsification Test).

**5. The feature-list transfer test could hardly have found a cost, so "transfers with little loss" (abstract) is weakly supported.**
*Location.* VIII-B (l. 950–952), Tables XII–XIII, X-D, abstract, conclusion (l. 1254). Code: `p2_consequence.py`.
*Problems found.*
- (a) **The five selection seeds collapse to one to three distinct lists.** In `p2_topk.csv` the RF VPN own Top-10 list gives the same macro F1 (0.8663) for all five seeds. The RF Non-VPN transferred Top-10 gives 0.9171 for all five. Many `p5_topk_2x2.csv` rows repeat across seeds. "Positive for all five seeds" (l. 950) counts the same list several times.
- (b) **Refit noise is never measured.** Each reduced model is refit once with seed 42 and cached by feature set. Yet several Top-10 models beat the full 23-feature model by as much as the reported transfer costs: RF VPN own Top-10 scores 0.866 against 0.853 for the full model, and XGB Top-15 scores 0.937 against 0.934. The bootstrap interval covers only test-record noise (acknowledged).
- (c) **Almost any feature set works.** Twenty random 10-feature lists average 0.818–0.899 macro F1. The best random lists reach 0.863–0.938, above the own Top-10 for XGB Non-VPN (max 0.938 vs 0.935). The two lists being compared share 7–9 of 10 features.
- (d) **Nothing shows SHAP matters.** There is no non-SHAP selector, such as impurity, permutation or mutual information, so "SHAP-based selection is doing real work" (l. 952) shows only that ranking beats random and bottom-10.
- (e) **At k = 5 the own SHAP list is worse in its own context.** For RF on the Non-VPN target the own list loses by 0.026, a negative result about SHAP as a selection guide that the 2×2 reframes as "the VPN list is better".
- (f) **No margin is defined.** "Little loss" has no equivalence margin (acknowledged in XI).

*Resolution.* Refit the reduced models over several training seeds. Report the number of distinct lists per k. Add a non-SHAP selector baseline. Pre-specify a margin, for example 0.01 macro F1, and test against it. Otherwise the abstract should say "lists that share 7–9 of 10 features perform similarly".

**6. The permutation-importance comparison is reported selectively.**
*Location.* VI-E (l. 769, 771: "Permutation-based measures show smaller shifts than SHAP for both models") and X-B (l. 1232).
*Problem.* The paper reports only that the minimum-IAT ranks move less under permutation importance. It leaves out that whole-ranking cross-context agreement under permutation importance is *lower* than under SHAP. Marginal PI gives ρ 0.611 (RF) and 0.719 (XGB) (`p4_results.json` `pi_cross_*`); conditional PI gives 0.657 and 0.575 (`p5_results.json` `cpi_cross_*`), against SHAP's 0.830 and 0.786. Top-5 overlap is 4/5 for RF under both and for XGB under conditional PI, against 3/5 for SHAP. The same files show that conditional PI gives SHAP's top RF Non-VPN feature, total_biat, a negative importance (−0.002). Caveat: the code clips negative PI to about zero and breaks ties by feature order, which lowers ρ. Even so, the direction contradicts the impression X-B leaves.
*Resolution.* Report whole-ranking and top-weighted cross-context agreement for both PI variants, with tie handling stated, and rewrite X-B.

**7. The family-level statements contain factual errors and contradict their own numbers.**
*Location and problems.*
- **l. 875.** "The eight data-driven clusters produce gaps of 0.096 for Random Forest and 0.099 for XGBoost, larger than the semantic-family gaps." For RF, 0.096 is *smaller* than the summed semantic gap of 0.112 (`ref_summary.json` `RF_random` `clust_gap` vs `fam_gap`).
- **l. 1234.** "per-member means and correlation-based clusters give larger gaps." This is false for XGB per-member means (0.064 < 0.075) and for RF clusters (0.096 < 0.112).
- **l. 877.** "part of the XGBoost difference between contexts persists at the family level… redundancy among related flows does not fully explain the observed differences." The paragraph just before it (l. 876) reports that the XGB family gap falls to 0.014 at about 35 flows per cluster and to 0.016 under calibration, the settings the paper treats as session-like. The family gap behaves like the feature gap there. The conclusion repeats the claim (l. 1253).

*Resolution.* Correct the sentences. Limit the "persists" claim to the ~10-flow setting, or drop it.

## Minor Issues

- **Kendall's τb is miscited.** It is attributed to Kendall (1938) [44] (l. 245). τb, the tie-corrected form, comes from Kendall (1945), "The treatment of ties in ranking problems", *Biometrika* 33. The 1938 paper defines τa.
- **Tie handling is described inaccurately.** The paper says Spearman uses average ranks for ties (l. 246). The code ranks with `method="first"` (`common.ranks`), so ties are broken by feature order and τb reduces to τa. This has no practical effect on continuous importances, but the description is wrong. Family-level ρ does use average ranks.
- **The capture list is inconsistent and cannot be verified.** The text says "109 Non-VPN and 30 VPN capture files… plus one VPN P2P capture" (l. 257). `designs.py` also counts one Non-VPN P2P capture; a second copy of the list in `paper_build/resnums.py` omits P2P. The Deep Packet prefix list itself is not in the repository, so the 109 and 30 cannot be checked. Median flows per capture of 44.5 and 245.5 are printed as "44" and "246" (round-half-even); this is harmless.
- **Matched-1,330 training was run for XGBoost only** (`p4_classmix_miniat.py`, `fit_imp("XGB", ...)`). IV-E and VI-D do not say so, and XI lists it as a general class-mix control.
- **The 800-record check covers the easy case only.** It was validated on 8 random-design RF half-models (`p5_results.json` `rf800_halfmodels`, ρ ≥ 0.998). It was not checked on the weak grouped half-models, where it matters more. The 800 records are also a subset of the full test set, which raises the agreement.
- **Row-block half sizes are not stated.** The common half size is given as 3,368–3,476 for clusters (l. 256). For row blocks it is 3,264–3,476, and ordered140 uses 3,264, 6% smaller than the random halves.
- **The lower bound on the explained share is wrong.** "from about 20% to nearly all of it" (l. 1230) uses the calibrated row blocks. The ~10-row blocks remove only 12% (XGB) and 10% (RF).
- **The "bounds" are chosen arbitrarily.** "between about zero and 0.07 / 0.15" (l. 455) uses the calibrated block setting as the upper end, although the ~10 and ~35 block settings give 0.11/0.17 and 0.11/0.16.
- **Abstract wording.** "achieves at best 0.43–0.61 macro F1": these are single-fit values, not maxima.
- **The conditional permutation scheme is not Strobl et al.'s method.** Quintile cells of the two most correlated features are cited to [36] (l. 769). Strobl et al. condition on the forest's split grid, so "in the spirit of [36]" is more accurate.
- **Extra Trees comparison wording (l. 1057).** "Extra Trees agrees less well with the path-dependent Random Forest (0.709–0.801)": 0.709 and 0.776 are the interventional-versus-interventional comparisons (`rob_ET_vs_RF_*_interv`).
- **Unsupported mechanism claim.** "hidden traffic direction" (l. 10): a VPN does not hide packet direction on the wire, and [4] does not establish this mechanism. Rephrase.
- **Oversimplified SHAP characterisation.** "credit is apportioned according to the fitted model, not the data distribution [18–20]" (l. 12) blurs two problems: off-manifold evaluation in interventional SHAP and credit to correlated features in observational SHAP.
- **The primary partition cannot be regenerated from the repository.** No script in the repository writes `data/novpn/X_train.csv` or `data/vpn/X_train.csv`. I searched all `*.py` files; `notebooks/` is empty. The seed of the primary 80/20 split is not stated in the paper.
- **External inputs.** The original ARFF files needed by `p0`/`p6` are outside the repository; this is acceptable if the download is documented.
- **[49] (Mali et al., 2025, Ann. Math. Comput. Sci.)** is unverified: I could not check this venue or the attributed content.
- **Missing related work an expert would expect:**
  - Explanation shift under distribution shift (Mougan et al., 2022–23), which is the closest conceptual neighbour of a cross-context SHAP comparison.
  - Blocked and grouped cross-validation for dependent data (Roberts et al., 2017, *Ecography*).
  - Variance correction for overlapping resamples (Nadeau & Bengio, 2003, *Machine Learning*).
  - Flow-meter artefacts in CIC datasets (Engelen, Rimmer & Joosen, 2021, IEEE SPW), relevant to the negative min_flowiat values and to how ISCXFlowMeter writes rows.
  - Pitfalls of model-agnostic importance under dependence (Molnar et al., 2022).
  - The REFORMS checklist itself (Kapoor et al., 2024, *Science Advances*).

## Editorial

- **Tone differs between abstract and conclusion.** The abstract says each model "keeps only three of its five" leading features; the conclusion says three of five shared features suggest "some important features remain consistent" (l. 1251). Use one framing.
- **The conclusion (l. 1251–1255) restates results without their caveats.** "The gaps remain non-negative across the tested repetitions" omits that this holds for Spearman only and over dependent repeats.
- **The abstract's sentence on grouping (l. 6) carries about six clauses.** Split it.
- **Table XII.** The "Δ k = 15" column has no [min, max], unlike the other Δ columns.
- **Awkward phrasing.** "Encryption does not eliminate all information about the traffic flows, such as…" (l. 17) and "The selection process in the feature list transfer involves selection from…" (l. 268).
- **Revision-history phrases.** None found ("earlier version", "previous version" and similar do not occur).

## Methodological Assessment

REFORMS modules:

- **Study goals and population.** The goals are clearly stated, and so is the non-causal framing. Partly undercut by the "same predictions" wording (Issue 3).
- **Computational reproducibility.** Good overall. Every reported number traces to a result file, versions are recorded (`p1_results.json` env matches Table III), and XGBoost is single-threaded. My independent refit reproduced the primary XGB cross-context metrics exactly. Gaps:
  - the creation script for the primary split is missing;
  - the capture list is hard-coded twice, inconsistently;
  - the source ARFF files and the Deep Packet list are not archived.
- **Data quality.** Strong. Deduplication, label conflicts, sentinels, negative IATs, the BROWSING and P2P caps and collisions are all audited and match the code.
- **Preprocessing and modelling.** Adequate. Hyperparameters are fixed, not tuned, as stated.
- **Leakage and dependence (REFORMS 6).** This is the central weakness. Session dependence is acknowledged and probed only through proxies. The test records are never session-disjoint (Issue 2). The row-order proxy is weakly validated (Issue 1).
- **Metrics and uncertainty (REFORMS 7).** The effect sizes are well chosen: ρ, RBO, κ5, L1, a class-weighted variant and family level. Uncertainty is limited to dependent repeats with no session-level component (Issue 4). The consequence test lacks training-seed variance and an equivalence margin (Issue 5).
- **Generalisability.** Limited to one dataset and one timeout. This is properly disclosed in XI.
- **Interpretation.** Cross-context explanations are computed on different record sets (Issue 3). The permutation comparison is selective (Issue 6).

GRADE-style grading of the main claims:

| Claim | Design | Risk of bias | Precision | Consistency | Directness | Grade |
|---|---|---|---|---|---|---|
| C1. Cross-context ρ is 0.13 (XGB) / 0.19 (RF) below random within-context halves | Observational, 1 dataset, 11 dependent repeats | Serious: session leakage inflates the within value; explained-data asymmetry | Adequate for the descriptive value (SD 0.03–0.04) | Consistent across ρ, RBO, Top-5 and class weighting | Indirect as a "context" effect | Moderate (descriptive) |
| C2. Class mix and the half-split procedure do not produce the gap (null) | Same | Low for this narrow claim | Adequate | Consistent with the swap and matched tests | Direct for class mix | Moderate–High |
| C3. Related flows explain part of the gap; the remaining excess is "between ~0 and 0.07/0.15" | Two unvalidated proxies, outer split random | Serious (Issues 1, 2) | Wide (RF calibrated SD 0.084) | Inconsistent: the proxies disagree | Indirect | Very low |
| C4. "Part is likely to remain"; gap non-negative in every repeat | Row-order proxy only | Serious | Repeats not independent | Holds for ρ, not for RBO or Top-5 in orderedhalf | Indirect | Very low |
| C5. Direct model transfer fails (0.43–0.61 macro F1) | Single fit per direction | Low for the descriptive claim | Single fit, no interval | Both models agree | Direct for the deployment question, not attributable to tunnelling | Moderate |
| C6. A SHAP Top-10 list transfers with little loss | 5 seeds collapsing to 1–3 lists, single refit | Moderate | Effects (0.004–0.013) ≈ refit-level differences | Fails at k = 5 | Weak link to explanation consistency | Low |
| C7. Min-IAT shift is redistribution for XGB and a real gain for RF | Retraining and permutation checks, single refits | Moderate (PI reported selectively) | Single refits | Mixed | Moderate | Low–Moderate |
| C8. Family-level gaps smaller; XGB family gap "persists" | Repeats | Text errors (Issue 7) | — | Contradicted at ~35 and calibrated settings | — | Low |

## Code vs paper

| Statement in paper | What the code does | Consistent? |
|---|---|---|
| Each repeat redraws a stratified 80/20 split with a seed that changes per repeat (IV-D) | `train_test_split(..., random_state=1000 + r)`, the same seed for both contexts and every design (designs are paired) | Yes. Pairing across designs is not stated. |
| Random halves stratified, subsampled to at most 3,477 (3,476–3,477) | Per-class halves with `h = len // 2`, then `strat_sub` to `N_H = 3477`; `ref_summary` n = 3,476 | Yes |
| Clusters: k-means on log(x+2)-standardised features within class, ~10/35/140 per cluster, ≥4 clusters | `KMeans(n_clusters=max(4, round(n/CL_SIZE)), n_init=3)` on `StandardScaler(logx2)` within class | Yes |
| Whole clusters assigned at random, larger first, each to the smaller half; class subsampled to its smaller half; common size 3,368–3,476 | Random permutation, stable sort by size, greedy assignment, `rng.choice(n)`, then the common minimum across contexts | Yes for clusters. Row blocks reach 3,264 (unstated). |
| Calibrated: clusters per class = number of capture files (≥4); BROWSING = flows / median flows per capture | `CAPTURE_FILES` includes P2P = 1 for both contexts; BROWSING uses the 2,500 ARFF count / median | Mostly. Text mentions only a VPN P2P capture. |
| Row blocks: contiguous blocks of about B *training* flows within class; first vs second half | `np.array_split(ids[np.argsort(pos[ids])], kc)` on training indices only; `orderedhalf`: kc = 2 | Yes. Blocks are contiguous among training rows; test rows interleave. |
| Every deduplicated flow matches exactly one ARFF row | p6 asserts one-to-one on a key of features rounded to 3 dp plus class | Yes (rounding key is a weak point, but the assert passed) |
| Locality: nearest of 40 rows within ±20 vs nearest of 40 random rows, 0.30–0.65 | Ratio of medians computed on *raw* ARFF rows, duplicates included | Partly. Duplicates are included; lag statistics are not reported. |
| Null: pool flows within class, reassign with original counts, then the random design | `contexts_for_repeat` with `rng(31000 + r)`, as described | Yes |
| XGB explained on the full test set; RF on an 800-record class-stratified sample | `ex_ids = strat_sub(..., 800, ...)` for RF | Yes |
| Every half-model has its own seed | `seed = 9000 + 100r + 10ci + h` (shared across designs) | Yes |
| Spearman on average ranks for ties; Kendall τb | `ranks(method="first")`, so no ties exist | No (immaterial) |
| Matched 1,330 class-matched training (no model named) | XGB only | Not stated |
| Selection repeated with five inner-split seeds | Five seeds, but reduced models refit once with seed 42 and cached by feature set; many seeds produce identical lists | Yes literally, but the seed spread is degenerate |
| Conditional PI permutes within quintile cells of the two most correlated features | As described | Yes |

## Number spot-checks

| Claim | Manuscript value | Results field | Match? |
|---|---|---|---|
| Class-mix TVD, χ², Cramér's V | 0.142; 447.4; 0.176 | `p1.classmix` tvd 0.1423, chi2 447.37, cramer_v 0.1759 | Yes |
| Grouped CV macro F1, XGB Non-VPN | 0.924 → 0.642 | `p1.cv_XGB_novpn_strat` 0.9235, `_grouped` 0.6420 | Yes |
| Near-duplicate audit (log), Non-VPN | 0.108 vs 0.101 | `p1.nn_novpn.log` 0.1085 / 0.1005 | Yes |
| Primary RF cross ρ, τb, RBO, L1, displacement | 0.8300, 0.6443, 0.580, 0.333, 3.04 | `p1.cross_RF` 0.83004, 0.64427, 0.57999, 0.33326, 3.0435 | Yes |
| Primary XGB κ10 (8/10) | 0.65 | `p1.cross_XGB.kappa10` 0.6462 | Yes |
| XGB random-halves Δρ (SD, range) | +0.129 (0.028), 0.072–0.163 | `ref_summary.XGB_random` rho_gap 0.1290, sd 0.0281, min 0.0721, max 0.1633 | Yes |
| RF random-halves Δρ | +0.186 (0.036), 0.143–0.252 | `RF_random` 0.1864, 0.0358, 0.1433, 0.2522 | Yes |
| Permutation null Δρ | −0.002 / +0.028 | `XGB_null` −0.0015; `RF_null` 0.0276 | Yes (rounding) |
| Row blocks calibrated Δρ | 0.074 / 0.153 | `XGB_orderedcal` 0.0741; `RF_orderedcal` 0.1528 | Yes |
| First vs second half: Δρ, minimum, "≥ 0 in all repeats" | 0.033 (0.000 to 0.082); 0.093 | `XGB_orderedhalf` mean 0.0329, min 0.000247, max 0.0818; `RF_orderedhalf` 0.0933 | Yes for ρ. RBO gap < 0 in 1 / 2 repeats. |
| Gap reduction percentages | 39%, 47%; 43%, 18% | Computed from `ref_summary`: 39.1%, 46.6%, 42.6%, 18.0% | Yes |
| RF calibrated cluster within-VPN ρ | 0.296 | `RF_calibrated.rho_wvpn` 0.296 | Yes |
| Model transfer macro F1 | 0.430 / 0.613; RF 0.438 / 0.606 | `p2.transfer_XGB_novpn_to_vpn` 0.4301; `vpn_to_novpn` 0.6126; RF 0.4379 / 0.6060 | Yes |
| Class-mix swap Δ(ρ with other) | +0.008, +0.017, −0.035, −0.007 | `p4.classmix_summary` rho_other means: 0.8306 − 0.8225, 0.7972 − 0.7806, 0.7648 − 0.7994, 0.8300 − 0.8366 | Yes |
| Matched 1,330 | 0.946 / 0.945; 0.853; 0.857; 0.971 | `p4.matched1330.mean` | Yes |
| Combined min-IAT share | 17.0 / 23.3; 26.3 / 24.7; half means 16.3 / 24.5 | `p5.block_*` 17.04, 23.33, 26.31, 24.71; `block_half_RF` 16.28 / 24.55 | Yes |
| Top-k context-match term, k = 10 | +0.008 [0.004, 0.013] / +0.010 [−0.001, 0.018] | `p5.topk_2x2` RF 0.0084 [0.0044, 0.0128]; XGB 0.0095 [−0.0013, 0.0182] | Yes |
| VPN RF k = 10 bootstrap interval | [0.006, 0.028] | `p2_topk.csv` row 46 ci_diff [0.00568, 0.02830] | Yes |
| Bottom-10 range | 0.685–0.735 | `p2_topk.csv` seed means: RF VPN 0.685, XGB VPN 0.735 | Yes |
| Family gaps, correlation clusters vs semantic | "0.096 / 0.099, larger than the semantic-family gaps" | `RF_random` clust_gap 0.096 < fam_gap 0.112; `XGB_random` 0.099 > 0.075 | Numbers match; **the comparison is false for RF** |
| Sentinel S1 vs S2 within ρ | 0.914 / 0.887; 0.942 / 0.943 | `p3.sent_*_S1vsS2.rho` 0.9140, 0.8874, 0.9417, 0.9427 | Yes |
| Row-locality window ratio range | 0.30–0.65 | `arff_order_locality.csv` win20 min 0.299, max 0.647 | Yes (raw rows; see Issue 1c) |

I found no number in the text that disagrees with its results field beyond rounding. The errors are in the comparisons drawn from the numbers (Issue 7) and in what the numbers are taken to show (Issues 1, 4, 5, 6).

## Falsification Test

There was no pre-registration. Reconstructed hypothesis H1: "VPN and Non-VPN models rely on features differently by more than within-context retraining on session-independent data does." A fair falsification criterion: H1 is refuted if a session-respecting reference brings Δ to ≤ 0, or to within the null, with half-models that are still competent.

How the paper engages with this:
- **The cluster proxy meets the criterion, but only with weak half-models.** At the session-calibrated settings Δ is about 0 (XGB −0.015 to +0.016, RF +0.002 to +0.018), and the half-model macro F1 is 0.53–0.68. Read strictly, that meets the refutation criterion, except that the "competent half-models" condition fails.
- **The row-order proxy does not meet it, but the proxy is not validated.** Row blocks keep Δρ > 0, yet Issue 1 shows the blocks are neither validated at capture scale nor feature-independent.
- **Neither design is clean.** Both share the random outer split (Issue 2) and the explained-data asymmetry (Issue 3), and the RF null baseline of 0.028 is not subtracted.

The honest verdict is that H1 is neither corroborated nor refuted by these data. The paper's final paragraph (l. 1255) states roughly this. The abstract's "part is likely to remain" and the discussion's "some excess probably remains" go beyond it and should be weakened to "is not ruled out".

**What would settle it.** A three-way block split, or better a capture split rebuilt from the PCAPs, with a grouped null computed on the same groups, every model explained on a common record set, and a decision rule fixed before the run: H1 is supported only if Δ minus the grouped-null Δ exceeds a stated margin and is positive under a capture-level block bootstrap.
