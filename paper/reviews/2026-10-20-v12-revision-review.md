An in-project reading by a separate AI session. It is not an outside review.

Scope: a full line-level diff of v11_original.txt against v12_revised.txt (difflib, 19 changed blocks: abstract, IV-C, IV-D x3, IV-G, VI-C x3, VI-E x2, VII x2, VIII-B, IX, X-A, X-C, X-D, XI x2, XII x3, a new Data and Code Availability section, and the reference list). I checked every changed or added number against revision2/results and the code. I also recomputed independently the row-order locality, the per-repeat gap signs, the one-sided gaps, the distinct-list counts and the half-split overlap, using read-only Python in the scratchpad. No project file was modified.

## Issues (most severe first)

### 1. The new row-order caveats are not carried into the conclusions that rest on row-order blocks (VI-F, X-A, abstract)
- **Category:** Inconsistency between sections; important limitation not propagated
- **Location:** IV-D and XI (new text) against VI-F (unchanged), the row-order part of VI-C, X-A and the abstract
- **Original statement:** IV-D: "Neighbouring rows are much more alike than random rows of the same class: in every class and context ... This suggests that the files keep flows from the same capture close together". VI-F (unchanged in v12): "With blocks of consecutive rows, however, within-VPN rankings are again steadier (calibrated blocks: 0.942 against 0.919 for XGBoost, 0.906 against 0.884 for Random Forest), so that the reversal in trend depends on the nature of the grouping."
- **Revised statement:** IV-D: "Adjacent rows, however, are almost as different as random pairs for VPN FT, MAIL, P2P and VOIP (median distance ratio 0.81–0.99 ...) ... the evidence is weak for those four VPN classes". XI: "For VPN FT, MAIL, P2P and VOIP, adjacent rows are nearly as different as random rows, and BROWSING and P2P cannot be split by capture at all." VI-F is unchanged.
- **Discrepancy:** v12 now admits that row order is a weak capture proxy in the VPN context. It does not draw the consequence.
  - Four weak-locality classes (FT 1,883, MAIL 432, P2P 666, VOIP 2,266) plus BROWSING (2,500) make up 7,747 of 9,380 VPN flows, or 83% (class counts from p7_audit.json locality_dedup "n").
  - The residual row-order gap is driven mainly by the VPN side, where the proxy is weakest. One-sided gaps (rho_wvpn − rho_cross against rho_wnv − rho_cross, ref_summary.json means; also p7_audit.json onesided_*):
    - orderedcal: RF 0.164 against 0.142; XGB 0.086 against 0.063
    - orderedhalf: RF 0.126 against 0.061; XGB 0.041 against 0.025
  - Two readings follow:
    - VI-F's "within-VPN rankings are again steadier" under row blocks can be explained by row blocks failing to keep VPN captures together. It therefore cannot show that "the reversal in trend depends on the nature of the grouping".
    - The "part probably remains" reading (abstract, VI-C, X-A, XII) leans most on the context where the proxy is least supported.
- **Recommended correction:**
  - In VI-F, add that row blocks are a weak proxy for most VPN classes, so the within-VPN steadiness under blocks may come from related VPN flows still crossing the halves.
  - In VI-C or X-A, report the one-sided row-order gaps and state that the Non-VPN side, where locality is better supported, gives 0.025 (XGB) and 0.061 (RF) under the first-versus-second-half split.

### 2. New whole-ranking permutation-importance comparison is driven by ties at zero, and top-weighted measures point the other way
- **Category:** Unsupported or under-qualified scientific claim
- **Location:** VI-E, "Permutation importance" paragraph and the summary paragraph
- **Original statement:** "Permutation-based measures show smaller shifts than SHAP for both models".
- **Revised statement:** "Over the whole ranking, permutation importance agrees less across contexts than SHAP does: cross-context ρ is 0.611 (Random Forest) and 0.719 (XGBoost) for marginal and 0.657 and 0.575 for conditional permutation importance, against 0.830 and 0.786 for SHAP." Also: "... although their whole rankings differ more between contexts".
- **Discrepancy:** The digits are correct (p4_results.json pi_cross_RF/XGB.rho; p5_results.json cpi_cross_RF/XGB.rho; p1_results.json cross_RF/XGB.rho). The interpretation is not supported.
  - p4_classmix_miniat.py line 125 and p5_concerns4.py line 51 clip PI at 0 before ranking.
  - For Random Forest, 10 of 23 marginal PI values are ≤ 0 in each context (p4_results.json pi_RF_novpn/vpn.values). Six are ≤ 0 for XGB Non-VPN, and 11 of 23 are below 0.002.
  - A large block of tied or near-zero features therefore dominates whole-ranking ρ for PI. That reflects estimator noise around zero, not a context difference.
  - Top-weighted agreement, which the paper itself argues matters more (IV-C, X-B), shows the opposite for Random Forest. PI has Top-5 overlap 4 and RBO 0.765 (conditional: 4 and 0.805), against SHAP's 3 and 0.580. For XGBoost, PI RBO is 0.752 and 0.791 against SHAP's 0.755.
  - The general claim that PI "whole rankings differ more between contexts" is therefore one-sided.
- **Recommended correction:**
  - Either drop the whole-ranking sentence, or qualify it: about half of the PI values are indistinguishable from zero and are tied after clipping.
  - Report RBO and Top-5 for PI next to ρ.
  - Rephrase the summary as "PI moves the minimum IATs less than SHAP; its whole-ranking ρ is lower, mainly because many PI values are near zero, while its top-weighted agreement is similar or higher".

### 3. "None of the designs controls for" the different explained records misdescribes the permutation null
- **Category:** Misrepresentation of methodology
- **Location:** IV-D (new sentences); repeated in VI-C, X-A and XI
- **Original statement:** none – added
- **Revised statement:** "Within-context pairs are thus explained on the same test records, whereas cross-context pairs are explained on the two contexts' different test partitions. Part of the cross-context disagreement may therefore come from the difference in explained records rather than from the models, and none of the designs controls for this." VI-C and X-A list this as a separate reason the excess is untested.
- **Discrepancy:** The first sentence is correct. In designs.py, ex_ids is drawn once per context and repeat and shared by h = 0 and h = 1, and cross pairs use each context's own Xte. The second sentence is wrong as stated.
  - In the null design (designs.py contexts_for_repeat), each pseudo-context gets its own outer 80/20 split and its own explained test records. Cross-pseudo-context pairs are therefore also explained on different records, while within pairs share records.
  - The null Δ thus already contains the effect of explaining different record samples from one distribution: +0.028 for RF and −0.002 for XGB (ref_summary.json RF_null and XGB_null rho_gap).
  - What no design separates is the difference in the distribution of the explained records between the real contexts, from the difference in the models. That is a narrower statement.
- **Recommended correction:** Rewrite along these lines: "The permutation null, whose pseudo-contexts are also explained on different test records, shows that explaining different record samples adds at most about 0.03 (Random Forest). No design separates the change in the explained record distribution between contexts from the change in the models." Adjust VI-C, X-A and XI to match.

### 4. The family-level inference is now a non sequitur, and XII still says the family gap "decreases less"
- **Category:** Altered technical meaning; inconsistency between sections
- **Location:** VII, last paragraph; XII, first changed paragraph (unchanged sentence)
- **Original statement:** "These results indicate that part of the XGBoost difference between contexts persists at the family level, even though it shrinks at the level of individual features. Therefore, redundancy among related flows does not fully explain the observed differences."
- **Revised statement:** "... under fine grouping, part of the XGBoost difference ... persists at the family level ...; with clusters of about 35 flows or the calibrated grouping, the XGBoost family gap also falls (0.014 and +0.016). Therefore, redundancy among related flows does not fully explain the observed differences."
- **Discrepancy:** The numbers are correct (ref_summary.json XGB_grouped35 fam_gap 0.0143 and XGB_calibrated fam_gap 0.0156; these happen to equal the Spearman gaps 0.0140 and 0.0163).
  - v12 now concedes that the family gap falls as far as the feature gap under the groupings the paper treats as closest to captures. Persistence only under the 10-flow grouping, which the paper calls one that "keeps only very close neighbours together", cannot support "Therefore ... does not fully explain".
  - XII (unchanged) still says "the XGBoost family-level gap decrease[s] less under feature-based clustering than the Spearman gap". That holds at 10 flows (0.077 against 0.079) and 140 flows (0.033 against −0.015), but not at 35 or calibrated (0.014 against 0.014; 0.016 against 0.016).
- **Recommended correction:**
  - Restrict the inference: "under fine grouping the family-level gap persists; under coarser grouping it falls with the feature-level gap, so the family level gives no separate evidence on residual differences".
  - Qualify the XII sentence with "at fine and coarse cluster sizes".

### 5. The new evidence that row position tracks features is not reflected in the bounds framing
- **Category:** Inconsistency between sections; interpretation
- **Location:** IV-D (new) against VI-C, X-A and XI ("blocks may undercorrect")
- **Original statement:** none – added
- **Revised statement:** IV-D: "within a class the row position is itself correlated with some features (|ρ| up to 0.68)". VI-C: "Clusters may overcorrect, and blocks may undercorrect wherever a capture crosses a block boundary. Reading them as rough bounds ..."
- **Discrepancy:** The 0.68 is correct (p7_audit.json max max_abs_rho_position_feature = 0.677, novpn_FT). However, if position tracks features, contiguous blocks, especially the first-versus-second-half split, also separate regions of feature space. Blocks can then overcorrect as well, which weakens their use as an upper bound.
  - The orderedhalf half-models are less accurate than the finer row-block designs: per-repeat macro F1 0.717–0.813, against 0.824–0.915 for ordered10 (ref_summary_*_orderedhalf.csv and ref_summary_*_ordered10.csv f1_nv/f1_vpn). That is consistent with some feature-space separation.
  - X-A (unchanged) still says blocks "do not exploit the features".
- **Recommended correction:** Add to VI-C and X-A that position is partly correlated with features, so the first-versus-second-half split may also overcorrect. Read the row-order values as neither a clean upper nor a clean lower bound.

### 6. "Only the outer partition and the seeds change" leaves out how half-models are assembled
- **Category:** Methodology description (minor)
- **Location:** VI-C (new sentence)
- **Original statement:** none – added
- **Revised statement:** "in the first-versus-second-half split every repeat uses nearly the same two halves of each class, so only the outer partition and the seeds change."
- **Discrepancy:** The "nearly the same two halves of each class" part is correct. Recomputed from designs.py logic and arff_order_positions.json: per class, 98.3–99.7% of flows present in two repeats fall on the same side.
  - The second clause is incomplete. In grouped_halves, the two class halves are ordered by length after a random permutation. For classes with an even number of training flows, which class half goes to half-model A is re-randomised in each repeat. For example, Non-VPN class 0 flips in 5 of 11 repeats, and VPN classes 0, 1, 2, 3 and 5 flip as well.
  - The two half-models' training sets therefore combine class halves differently across repeats, and the subsampling to the common size also changes.
- **Recommended correction:** "... so the outer partition, the subsampling, the seeds and the pairing of class halves into the two half-models change, but the split point within each class does not."

### 7. The 0.37–0.76 range comes from a single random draw
- **Category:** Numerical result (precision)
- **Location:** IV-D
- **Original statement:** "0.30–0.65 times the distance" (computed on all ARFF rows)
- **Revised statement:** "Over the deduplicated flows, the nearest of the 40 rows within 20 positions lies at 0.37–0.76 times the distance of the nearest of 40 random rows"
- **Discrepancy:** The reported values match p7_audit.json locality_dedup win20_dedup (min 0.3688 for novpn_FT, max 0.7563 for vpn_P2P). The change from v11 is a legitimate correction to deduplicated flows.
  - The statistic depends on the random 40-row draw. Over six other seeds I obtained a lower end of 0.338–0.379 and an upper end of 0.705–0.763. Non-VPN MAIL ranged 0.540–0.585 against 0.508 in the audit run.
  - The qualitative claim holds. The two-decimal endpoints are not stable.
- **Recommended correction:** Average over several random draws, or write "about 0.35–0.75".

### 8. P2P capture count and the "BROWSING cannot be split by capture" claim (partly unverified)
- **Category:** Unsupported claim (minor); citation content
- **Location:** IV-D and XI (new sentences)
- **Original statement:** "P2P has at most one capture per context"
- **Revised statement:** "BROWSING has no capture files of its own and P2P has a single capture per context, so for these two classes ... no within-class grouping can separate captures." XI: "BROWSING and P2P cannot be split by capture at all."
- **Discrepancy:**
  - The cited list ([56], munhouiani/Deep-Packet utils.py PREFIX_TO_TRAFFIC_ID, read through a fetch summary) has exactly one P2P entry, vpn_bittorrent, and no Non-VPN P2P prefix. The same paragraph says "plus one VPN P2P capture". designs.py CAPTURE_FILES still sets Non-VPN P2P = 1, which appears to be an assumption.
  - "Single capture per context" is therefore not supported by the cited source for Non-VPN. It may be true of the raw dataset; unverified.
  - BROWSING flows have no dedicated captures but presumably come from several other captures. Capture separation is impossible to calibrate, not impossible in principle; unverified.
  - The 34–40% share is computed on all deduplicated flows (p7_audit.json share_browsing_p2p_*: 0.3375 and 0.3958), not on training partitions. With a stratified split the difference is negligible.
- **Recommended correction:** "P2P has at most one capture per context, and BROWSING has no dedicated captures, so for these classes the capture structure is unknown and cannot be used for grouping."

### 9. Repository documentation still attributes the prefix list to the Deep Packet authors (not in the manuscript)
- **Category:** Citation and attribution (code-side only)
- **Location:** revision2/src/designs.py lines 40–42; revision2/README.md line 9
- **Original statement:** designs.py: "prefix list of the Deep Packet code, Lotfollahi et al."
- **Revised statement:** Manuscript v12: "a public, independent implementation of Deep Packet [7, 56]"
- **Discrepancy:** The manuscript change is correct. The repository page shows a third-party PyTorch reimplementation owned by "munhouiani", read through a fetch summary. Its PREFIX_TO_TRAFFIC_ID gives 109 Non-VPN prefixes (CHAT 20, EMAIL 4, FT 32, STREAMING 19, VOIP 34) and 30 VPN prefixes plus vpn_bittorrent. This matches CAPTURE_FILES in designs.py and the "109 ... 30 ... plus one VPN P2P" sentence.
  - The code comment and README line 9 still credit "Lotfollahi et al.", contradicting the manuscript and README line 71.
- **Recommended correction:** Update the comment and README line 9 to name [56].

## Categories with no material issue found
- **Changed numerical results:** every changed or added number matches its result field (see below). The two v11 numbers that v12 changed (0.30–0.65 → 0.37–0.76, and the Extra Trees range) are corrections, subject to issue 7.
- **Causal claims:** none were added. v12 adds non-causal caveats only, and the tunnelling disclaimers are kept everywhere.
- **Limitations weakened or removed:** none. Each v11 limitation paragraph is kept or extended. The v11 phrase "a positive gap in every repeat" in XI became "a positive mean gap"; that removes an overstatement, not a limitation. The deleted IV-D phrases "much more alike ... in every class" and "the similarity fades over tens to a few hundred rows" were overstatements given issue 1.
- **Reproducibility information removed:** none. A Data and Code Availability section was added. The analysis code is private and available on request, which is new information rather than a removal.
- **Citation mapping:** no errors (see below).

## Checked and found consistent
- **Kendall τb citation [44] → [55]:** Kendall (1945), "The treatment of ties in ranking problems", Biometrika 33(3):239–251 is the correct source for τb. [44] (Kendall 1938) is still cited in II-E for τ in general. The bibliographic details are from memory and were not fetched.
- **Reference renumbering:** old [55–57] became [57–59] (IV-G "[57–59]" → Ambroise, Kaufman, Kapoor). All 59 references are cited, and first-citation order is monotone. Checked with a regex over the v12 body.
- **[7, 56] attribution:** the new [56] is the munhouiani/Deep-Packet repository. Its prefix counts reproduce 109 Non-VPN and 30 VPN, plus vpn_bittorrent.
- **Abstract "0.03 and 0.09 with first- and second-half splits":** ref_summary.json XGB_orderedhalf rho_gap 0.0329 and RF_orderedhalf 0.0933.
- **Abstract "opposite gaps in the two contexts cancel":** one-sided gaps are of opposite sign for calibrated and for grouped (140) in both models (ref_summary.json rho_wnv/rho_wvpn/rho_cross).
- **VI-C calibrated one-sided gaps:**
  - RF +0.228 and −0.193: ref_summary_RF_calibrated.csv means 0.2277 and −0.1926.
  - XGB +0.067 and −0.035: 0.0671 and −0.0344.
  - X-A "+0.23 / −0.19" matches.
- **VI-C "Spearman gap at or above zero in all 11 repeats" under every row-order setting:** per-repeat minima from ref_summary_{RF,XGB}_{ordered10,35,140,orderedcal,orderedhalf}.csv rho_gap are all ≥ 0.0002.
- **VI-C "smallest XGBoost value 0.0002":** XGB_orderedhalf min rho_gap 0.000247.
- **VI-C negative counts in orderedhalf:**
  - RBO 1 (XGB) and 2 (RF); Top-5 3 and 2. Recounted from the CSV rbo_gap and top5_gap columns.
  - XGB ordered140 also has one negative Top-5 repeat. The text does not mention it, and nothing in the text contradicts it.
- **VI-C, X-A "permutation null ... +0.028 for Random Forest":** RF_null rho_gap 0.0276.
- **IV-D lag-1 ratios "0.81–0.99, against 0.11–0.77":** p7_audit.json lag1_dedup is 0.812–0.994 for VPN FT, MAIL, P2P and VOIP and 0.111–0.769 for the others. My recomputation gave 0.815–0.991.
- **IV-D "|ρ| up to 0.68":** p7_audit.json max_abs_rho_position_feature 0.677.
- **IV-D "Every deduplicated flow matches exactly one ARFF row":** arff_order_positions.json has 8,694 and 9,380 unique positions, equal to the deduplicated counts.
- **IV-D "34–40% of the training flows":** 33.8% and 39.6% (see the caveat in issue 8).
- **IV-D "Within-context pairs ... same test records; cross-context pairs ... different":** in designs.py, ex_ids is shared by both halves of a context in each repeat (subject to issue 3).
- **VI-E PI and CPI cross-context ρ 0.611, 0.719, 0.657, 0.575 and SHAP 0.830, 0.786:** p4 pi_cross_*.rho, p5 cpi_cross_*.rho and p1 cross_*.rho (subject to issue 2).
- **VII family gaps:**
  - XGB semantic 0.064–0.088 against clusters 0.099.
  - RF semantic 0.096–0.149 against clusters 0.096.
  - XGB at 35 flows and calibrated: 0.014 and +0.016.
  - Source: ref_summary.json fam_gap, fam_mean_gap, fam_alt_gap and clust_gap for the random, grouped35 and calibrated designs.
- **X-C "per-member means larger for RF not XGB; clusters larger for XGB not RF":** RF 0.149 > 0.112 and XGB 0.064 < 0.075; clusters XGB 0.099 > 0.075 and RF 0.096 < 0.112.
- **IX path-dependent vs interventional, and Extra Trees vs Random Forest (corrects v11):**
  - Path vs interventional: 0.969–0.984 for RF and XGB, 0.960–0.970 for ET.
  - ET vs RF: path 0.730–0.801, interventional 0.709–0.776.
  - Source: p3_results.json rob_*_path_vs_interv and rob_ET_vs_RF_*. v11's "path-dependent ... 0.709–0.801" had mixed the two formulations.
- **VIII-B, X-D, XI "one to three distinct lists ... each distinct list is refitted once with a fixed seed":**
  - p7_distinct_lists.json gives 1–3 lists, recomputed from p2_results.json rank_lists.
  - p2_consequence.py score() caches by (model, context, sorted features) and fits with seed 42.
  - The RF VPN k = 10 contrasts take only 3 distinct values (+0.0167, +0.0161, +0.0072; p2_topk.csv), which is consistent with the caveat.
- **XI and XII additions:** consistent with VI-C, apart from issues 1, 3 and 5.
- **Table VII, Table XV and Fig. 7:** unchanged, and every changed text number agrees with them (Δρ values, null −0.002 and +0.028, calibrated +0.018 and +0.016).

---
## Verification by the drafting session (2026-10-20)
- Issue 1: one-sided row-block gaps confirmed from results/p7_audit.json (RF orderedhalf 0.061 NV / 0.126 VPN; orderedcal 0.142 / 0.164; XGB 0.025 / 0.041 and 0.063 / 0.086); VPN BROWSING+FT+MAIL+P2P+VOIP = 82.6% of VPN flows.
- Issue 2: clipping at 0 confirmed (p4_classmix_miniat.py pi_cross, p5_concerns4.py cpi_cross); RF has 10 features with PI <= 0 in each context; top-weighted agreement of PI exceeds SHAP for RF (RBO 0.765 / 0.805 vs 0.580).
- Issue 3: confirmed (designs.py: null pseudo-contexts are explained on their own test partitions).
- Issue 7: confirmed; seeds 1-3 give 0.36-0.74, 0.36-0.75, 0.36-0.70.
- Issue 4 (XII part) not confirmed: the XGB family gap falls less than the Spearman gap at every grouped setting in relative and absolute terms; only the VII "Therefore" needs scoping.
Decision: pending the author.
