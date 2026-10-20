An in-project reading by a separate AI session. It is not an outside review.

# Problems (most severe first)

1. **The abstract's last sentence claims more than was tested.**
   - Quote: "Part of the random-split gap reflects related flows; the rest survives explanation on common records and exceeds a grouped permutation null under row-block grouping".
   - Evidence: the common-records check (p9_common_records.py) refits only the flow-level random-halves design and the primary models. Nothing about "the rest" was re-explained on common records, meaning the part left after grouping. Even under random halves, the Random Forest gap does not "survive" whole. It falls from 0.186 to 0.134–0.147, so 21–28% goes (p11_v14.json common_records.RF.gap). The abstract never mentions this reduction. "The rest" also suggests a single remainder. Under the cluster proxy the abstract itself reports about zero, and VI-C says the cluster question stays open.
   - Fix: say what was shown, for example: "Under random halves the gap persists when both contexts are explained on the same records (fully for XGBoost; for Random Forest it falls by about a quarter). Under row-block grouping it exceeds a grouped permutation null, while for feature clusters no neutral null exists. Because the contexts are different capture sets, none of it can be attributed to tunnelling."

2. **XI understates the records effect, and it carries straight into the row-block claims.**
   - Quote (XI): "the Random Forest gaps of the grouped designs, which were not recomputed on common records, may slightly overstate the difference between the models."
   - Evidence: under random halves the records effect for Random Forest is 0.039–0.052 in absolute terms (0.186 against 0.147/0.134). The Random Forest RBO gap falls by 37–47% (0.302 to 0.161–0.189). Suppose the absolute effect carries over to row blocks; the direction is untested. It would then be roughly 25–55% of the first-versus-second-half gap (0.093) and 25–35% of the calibrated row-block gap (0.153). The null pseudo-contexts have exchangeable records, so for Random Forest the "real minus null" excess (0.121, 0.111) includes this component. The excess probably stays positive (0.093 − 0.052 > −0.017), but "slightly" has no support.
   - Fix: drop "slightly". State that the size of the effect under grouping is unknown, with 21–28% of the gap under random halves (about 0.04–0.05 in ρ) as the only estimate. Better still, rerun p9 for orderedcal and orderedhalf. The script needs only the design's half-forming code swapped in.

3. **"Part of it survives every row-order grouping" (VI-C last paragraph; XII paragraph 4) is unqualified, and the qualifications v13 had are gone from the conclusion.**
   - Evidence:
     - (a) Nulls exist only for orderedcal and orderedhalf, not for blocks of about 10, 35 or 140.
     - (b) Random Forest row-block gaps were not recomputed on common records (item 2).
     - (c) For XGBoost first-versus-second-half, only 5 of 11 real repeats exceed the largest group-level null repeat. The smallest real value is 0.0002. The RBO and Top-5 gaps are negative in some repeats, and the repeats reuse nearly the same halves.
     - (d) Row blocks follow captures poorly in four VPN classes, and BROWSING and P2P (34–40% of the training flows) cannot be split by capture.
   - The v13 conclusion kept the RBO and Top-5 caveat and "Neither result is a direct test of the remaining excess". v14 drops both. It also drops "Not all measures ... respond equally: the top-weighted RBO gap and the XGBoost family-level gap decrease less under feature-based clustering", which is a substantive finding and no longer appears in XII.
   - Fix: "a positive mean Spearman gap remains under every row-order grouping and, at the two settings tested, exceeds the grouped null; for Random Forest part of it may reflect the explained records, and the top-weighted measures are less consistent across repeats". Restore the RBO and family-level sentence in XII.

4. **VI-C, second paragraph, contradicts itself.**
   - Quote: "Nor does the gap come from explaining the two contexts on different records." Two sentences later: "For Random Forest they account for about 21–28% of the gap".
   - Fix: "Nor does the gap come mainly from …".

5. **XII paragraph 2 overstates the scope of the records check and implies effects add up.**
   - Quote: "Explaining every model on the same records changes little for XGBoost and removes about 21–28% of the Random Forest gap. Keeping related flows in the same half reduces the gap further: to 0.08–0.10 …".
   - Evidence: only the random-half models and the primary models were explained on common records, not "every model". The grouped gaps that follow (0.08–0.10, 0.01–0.04, about zero, 0.07/0.15, 0.03/0.09) are all on each context's own records. They are not reductions applied after the records effect, so "further" is wrong.
   - Fix: "Explaining the random-half models on the same records …". Replace "reduces the gap further" with "Separately, keeping related flows in the same half reduces the (own-record) gap …".

6. **The nulls for the calibrated designs do not use the grouping resolution of the real design, and IV-D does not say so.**
   - Code: in grouped_halves with origin, the cluster or block count per class and origin is `calibrated_k(CTX[o], k)`. That is the full count of the origin context, but it is applied to only the pseudo-context's share of that origin's flows (f_nv ≈ 0.31–0.59 by class). The groups are therefore about 0.3–0.7 times the size of the real calibrated groups. For example, Non-VPN CHAT has 6.7 flows per group in pseudo-NV against 21.4 in the real design, and VPN FT has 52.5 against 92.6. The cluster-size designs (about 10 and 35) keep their sizes and are not affected.
   - In the group-level version this also means each capture-sized unit is re-cut into about 1/f pieces, which go to both halves. Within halves therefore share units, while pseudo cross pairs share none.
   - Consequences:
     - (a) The group-level calibrated nulls are less strictly grouped than the real calibrated designs. Random Forest within-NV ρ is 0.926 in the null against 0.884 in the real design for row blocks, and 0.633/0.671 against 0.717/0.297 for clusters. A finer grouping raises Δ (compare real grouped10 > grouped35 > calibrated). Part of the "pushes Δ up" at the calibrated setting may therefore be resolution rather than feature distribution.
     - (b) The group-level row-block null is not neutral either. Unit splitting, plus the within-class correlation of row position with features (|ρ| up to 0.68, stated in IV-D), both push it up. This direction is conservative for the row-block excess, but it contradicts the implication of "row blocks are not built from the features".
     - (c) Half sizes in the group-level nulls range from 2,775 to 3,589, against 3,457–3,476 in the real designs. They are not reported.
   - Fix: either scale the group count by the pseudo-context share and rerun, or disclose all three points in IV-D. Then rephrase the row-block argument: "the flow-level version is biased down, and the group-level version, if anything, up; both are near zero."

7. **The stated bias directions are asserted rather than shown.**
   - Quote (VI-C): "the two versions point in the directions expected from their construction".
   - Evidence: the run itself contains support that the text does not use.
     - Flow-level cluster nulls: pseudo cross ρ exceeds within ρ (Random Forest ~35: cross 0.835 against within 0.731/0.810).
     - Group-level cluster nulls: pseudo-contexts disagree more than the real contexts do under the same design (Random Forest ~10: null cross 0.600 against real cross 0.730; ~35: 0.608 against 0.682; XGBoost ~10/~35: 0.803/0.794 against 0.832/0.837). At the calibrated setting the evidence goes the other way for Random Forest (null 0.582 against real 0.489), which ties in with item 6.
   - In the flow-level argument, "the pseudo cross pairs share related flows that the grouped within pairs do not" holds only if groups match captures. With clusters of about 10, within halves also share captures.
   - "Lie between them" holds for the means of all 6 model × cluster settings (checked). For XGBoost ~10, however, the real gap (0.079) is close to the group-level null (0.097, SD 0.079), and the ranges overlap.
   - Fix: cite the cross-agreement evidence and say where it fails (calibrated). Change "do not" to "share more … than". Say "at the mean, for every model and cluster setting".

8. **Lost and inflated wording in the editorial pass (non-new text).**
   - X-D: "SHAP-selected feature lists, however, transfer more reliably" became "a SHAP-selected feature list transfers well". This is stronger than a test the same paragraph calls limited in power, with no equivalence margin, and with single comparisons reaching 0.05 at k = 5. Suggest "transfers with little loss at k = 10 and 15".
   - II-A: "Encryption does not eliminate all information …" became "inter-arrival times, packet sizes, byte and packet rates and flow duration all survive encryption". This is a new and stronger claim. Packet sizes can be padded, and a VPN tunnel aggregates and re-times flows, which is the paper's own premise. Suggest "remain observable after encryption, though tunnelling can alter them".
   - II-A: "deep learning models that classify packets and bytes [5, 7]" is imprecise. Those models classify traffic from packets or raw bytes.
   - X-C: v13's "family-level conclusions should always be interpreted alongside the grouping method and family sizes" is replaced by "(Section VII)". This is acceptable because VII keeps the caveat, but the Discussion loses it.

9. **X-E's new recommendation is broader than its evidence.**
   - Quote: "Rankings from two contexts should be computed on the same records, or the effect of the records measured, because here the explained records alone accounted for about 21–28% of the Random Forest gap."
   - Evidence: the 21–28% holds for random halves only, and XGBoost shows about 0%. Explaining one context's model on the other context's records also evaluates it off its own data distribution. That answers a different question from "what the model relies on in its context", so the common-record ranking is not simply the correct one.
   - Fix: add "under random halves (none for XGBoost)". Prefer "the effect of the explained records should be measured" over prescribing common records.

10. **"For XGBoost the explained records make no material difference" (VI-C) holds for ρ but not quite for RBO.**
   - Evidence: the XGBoost RBO gap falls from 0.152 to 0.121–0.149 (up to about 20% on the union and VPN records). X-A's "leaves the XGBoost gap unchanged" is likewise only about ρ.
   - Fix: "make no material difference to Spearman agreement (the RBO gap falls by up to a fifth)".

11. **IV-D's assignment description does not match the code.**
   - Quote: "each group is assigned at random to one pseudo-context so that each receives about its original share".
   - Code: groups are taken in random order, and each goes deterministically to the pseudo-context further below its target (contexts_for_group_null).
   - Fix: "taken in random order, each group goes to the pseudo-context further below its target share".

12. **The Table VII note is now partly wrong.**
   - Quote: "The null design … reassigns them … to pseudo-contexts with the original class counts."
   - Evidence: for Null (groups) rows, class counts are matched only approximately (IV-D says so).
   - Fix: add "(approximately, for Null (groups))".

13. **Transparency of the superseded version.**
   - The first grouped null (null_*, with halves differing in Non-VPN/VPN origin mix, documented in p12_null_origin_check.json) is not mentioned. Its gaps were −0.057 to +0.005.
   - The replacement is principled, but one sentence in IV-D or the supplement would pre-empt a "choice after seeing results" reading.
   - The p12 origin check covers null_* and nullstrat_* but not nullgroup_*. Nullgroup is balanced by construction (STRAT applies), so this is a note, not an error.

14. **Table XV.**
   - Pre-existing: "Δρ, permutation null" for XGBoost reads −0.002 (0.009), while VI-C says −0.001. The data give −0.00146, which rounds to −0.001.
   - New: Table XV "collects the main quantities", but it has no row for either new analysis, even though both now appear in the abstract and conclusion.
   - Fix: correct to −0.001. Add rows for the common-record Δρ and for the group-level row-block null.

# Checked

**Editorial pass**
- Read all 34 diff blocks in v13_v14_diff.txt, plus a line-level diff of v13.txt against v14.txt to confirm nothing else changed. Apart from item 8 and the conclusion losses in item 3, the edits keep meaning, numbers and citations. Citation lists are unchanged in every edited sentence, and I found no grammar errors.
- Edits that are clearly better: II-E (Fisher et al.), III-A (deduplication), III-B, the drop-one paragraph in VI-E (substitution logic now explicit), VII (family-summary paragraphs), and X-B ("A single summary statistic would hide both points").
- XII's "redistribution … within the block" (v13: "among features") is supported by X-C and Table XV (XGBoost minimum-IAT block share 26.3%/24.7%).

**Code against the method text**
- designs.py:
  - contexts_for_group_null groups per class and origin on the pooled flows, using calibrated row blocks or calibrated k-means units.
  - unit_groups behaves as described.
  - grouped_halves with origin groups within class and origin, so both halves hold the same origin mix.
  - nullstrat uses the same pseudo-contexts and seeds as null.
- p9_common_records.py:
  - It uses the same seeds as the random design (1000+r, 5000+100r+ci, 7000…, 9000…, 61000…).
  - Records are both contexts' explained records (XGBoost: full test partitions; Random Forest: the 800-record samples), and the union is taken by stacking.
  - Reproduction is exact (max_abs_rec_diff = 0.0; own-gap difference about 1e-16).
  - The primary-model own-record ρ equals the stored 0.830 and 0.786.

**Numbers**
- All 20 new Table VII rows (9 cells each) match ref_summary_<M>_<nullstrat|nullgroup>_<design>.csv. The only differences are sign formatting of zeros ("0.000" against "−0.000", "0.00" against "+0.00").
- I recomputed four designs directly from the pkl files with scipy, outside summarise(): Random Forest nullgroup_orderedhalf −0.0172 (SD 0.0249), XGBoost nullgroup_orderedcal 0.0113, Random Forest nullstrat_grouped35 −0.0639, XGBoost nullgroup_grouped10 0.0971. All four match.
- VI-C common records:
  - XGBoost 0.123–0.130 and Random Forest 0.134–0.147: correct.
  - Positive in all 11 repeats on all three record sets: correct (minimum 0.049).
  - 21–28%: (0.186−0.147)/0.186 = 21.1% and (0.186−0.134)/0.186 = 28.1%.
  - RBO 0.302 → 0.161–0.189: correct.
  - Primary models: Random Forest 0.863–0.883, XGBoost 0.771–0.787: correct.
- VI-C grouped null:
  - Row-block null range −0.022 to 0.032: correct.
  - Excess over the group-level null 0.063/0.035 (XGBoost) and 0.121/0.111 (Random Forest): correct.
  - 5–11 of 11 real repeats above the null maximum: correct.
  - Flow-level cluster nulls −0.064 to −0.009, group-level 0.041 to 0.265, SD up to 0.184, real 0.014–0.100: correct.
  - "Between" holds at the mean for all 6 settings.
- The new numbers in the abstract, X-A, X-E, XI and XII are consistent with the above.

**Consistency**
- XI's old "null only for random halves" and "left for future work" were removed, as was the future-work item in XII. No remaining statement says the null covers only random halves.
- Section I contributions are updated.
- Fig. 2 (image2.png in the docx) is byte-identical to the regenerated fig2_pipeline.png and lists "null with grouped halves; same-record check".
- Fig. 7 still describes the eleven original designs, which is correct.
- The Table VII note has the issue in item 12.
