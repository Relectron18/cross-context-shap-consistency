An in-project reading by a separate AI session. It is not an outside review.

Text read: /tmp/claude-0/rev13/v14_final.txt (commit f594e59). Note that /tmp/claude-0/rev13/v14.txt has been overwritten with the same rebuilt text, so the round-1 text no longer exists on disk. I checked the rebuilt text against v13.txt and my round-1 notes.

# Status of the 14 round-1 items

1. **Abstract overclaim: fixed.**
   - New text: "Part of the random-split gap reflects related flows; under row-block grouping a smaller gap remains on common explained records and above a grouped permutation null, but the contexts are different capture sets, so none of it can be attributed to tunnelling."
   - "The rest" and the claim that it "survives explanation on common records" are gone.
   - Every part of the new sentence is now measured: row-block gaps on common records are positive on average (see Checked), and they exceed the group-level null.
   - Small residue (see new problem 4): only two of the five row-block settings were tested, and the abstract does not say that common records lower the Random Forest row-block gap by 22–35%.

2. **XI "slightly overstate": fixed.**
   - The row-block designs were rerun with EXPLAIN_OTHER=1. XI now reads: "lowers the Random Forest gaps by about 21–28% under random halves and 22–35% under the two row-block settings, so Random Forest gaps measured on each context's own records overstate the difference between the models by roughly that much; the cluster designs and the other row-block settings were not recomputed on common records."
   - I recomputed 22–35% from the pkls: orderedcal 22.4–29.7%, orderedhalf 24.1–35.1%.

3. **"Survives every row-order grouping": fixed.**
   - VI-C now says: "survives the two row-block settings tested, on each context's own records and on common ones, although under the first-versus-second-half split the XGBoost gap is small and the RBO and Top-5 gaps are negative in some repeats."
   - XII restores the RBO/Top-5 caveat ("although the RBO and Top-5 gaps are negative in some repeats of the first-versus-second-half split").
   - XII also restores the RBO/XGBoost family finding ("the top-weighted RBO gap and the XGBoost family-level gap shrink less under clusters than the Spearman gap does").
   - XI states the scope: "two of the five row-block settings and three of the four cluster settings".
   - One wording residue in the final paragraph of XII (new problem 3).

4. **VI-C self-contradiction: fixed.** The sentence now reads "Nor does the gap come mainly from explaining the two contexts on different records."

5. **XII scope and the implied adding of effects: fixed.**
   - XII now says "Explaining the random-half models on the same records …" and "Keeping related flows in the same half also reduces the gap, measured on each context's own records: …".
   - It then separately says the two row-block settings "remain when both contexts are explained on the same records".

6. **Group resolution in the calibrated nulls: fixed, with two small residues.**
   - Code: grouped_halves `cal()` sets the group count to `calibrated_k * len(ids) / (0.8 * N_origin,class)`, which is about the pseudo-context share times calibrated_k. Groups therefore stay about the size of a capture's training part. Real designs are unchanged, because `cal()` returns calibrated_k when origin is None.
   - The rescaled pkls show the effect. Half sizes fell, for example nullstrat_calibrated from 3,415–3,462 to 2,981–3,128. The nulls moved in the expected direction:
     - group-level orderedcal: XGBoost +0.011 → +0.006, Random Forest +0.032 → +0.023
     - flow-level calibrated, Random Forest: −0.048 → −0.095
   - IV-D now discloses both the scaling and the group-level half sizes (2,409–3,589, against 3,457–3,476).
   - Residues: (a) the flow-level null half sizes are not disclosed (new problem 1). (b) IV-D now mentions that row position is correlated with features, but still gives no bias direction for the group-level row-block null (new problem 2).

7. **Bias directions asserted rather than shown: fixed.**
   - The flow-level wording is now "share more related flows than the grouped within pairs do".
   - The evidence is now cited: flow-level cross ρ against within ρ is 0.863 against 0.821 on average. I recomputed this from p11 as 0.8627 and 0.8209, and cross exceeds within in each of the 6 settings.
   - Group-level pseudo-contexts agree less than the real contexts "in all six model and cluster settings". After rescaling this is true in all six, including both calibrated settings (0.766 < 0.786 and 0.472 < 0.489; the margins are small).
   - "Lie between the two versions for every model and cluster setting" is true: XGBoost calibrated −0.028 < 0.016 < 0.040 and Random Forest calibrated −0.095 < 0.018 < 0.080. The close case (0.079 against 0.097) is named.

8. **Editorial inflation: fixed.**
   - X-D is back to "transfers more reliably".
   - II-A now reads "Encryption does not remove all information about a flow: … remain observable".
   - "Deep learning models that take packet- or byte-level input [5, 7]".
   - The X-C family caveat is restored ("family-level conclusions depend on the grouping rule and family sizes (Section VII)").

9. **X-E recommendation: fixed.** It is now framed as a diagnostic: "under random halves this accounted for about 21–28% of the Random Forest gap here and for almost none of the XGBoost gap. Explaining a model on another context's records evaluates it away from its own data, so this is a diagnostic, not a better ranking."

10. **XGBoost RBO: fixed.** VI-C now reads "little difference to ρ, although the RBO gap falls from 0.152 to 0.121–0.149".

11. **Assignment rule: fixed.** IV-D now reads "the groups are taken in random order, each going to the pseudo-context that is further below its original share of the class", which matches contexts_for_group_null.

12. **Table VII note: fixed.** It now says "so their class counts match the original ones only approximately".

13. **Superseded version: fixed.** IV-D now says: "An earlier version without the per-file split was discarded because its two halves could hold very different shares of Non-VPN and VPN rows." This is accurate. The p12 origin check still covers only null_ and nullstrat_. Nullgroup is balanced by construction, so this is not an error.

14. **Table XV: fixed.**
   - The permutation null for XGBoost now reads −0.001 (0.009).
   - Three rows were added, and their values match p11_v14.json:
     - random halves on common records: 0.134–0.147 / 0.123–0.130
     - row blocks on common records (union): Random Forest 0.114/0.068, XGBoost 0.066/0.032
     - minus the group-level null: Random Forest 0.130/0.111, XGBoost 0.069/0.035

No item introduced a contradiction, as far as I found.

# New problems in the rebuilt text (most severe first; all minor)

1. **Flow-level null half sizes are not disclosed.**
   - IV-D gives half sizes only for the group-level version: "the halves hold 2,409–3,589 training flows, against 3,457–3,476 in the real grouped designs".
   - The rescaled flow-level nulls also have smaller halves: nullstrat_calibrated 2,981–3,128 and nullstrat_orderedcal 3,264–3,347 (p11_v14.json half_size_min/max), against 3,457–3,474 and 3,467 in the real designs.
   - Fix: "In both versions the halves hold 2,409–3,589 training flows …", or give both ranges.

2. **The bias direction of the group-level row-block null is still not stated.**
   - IV-D: "row blocks are not built from the features, although row position is correlated with some of them."
   - The clause implies a possible upward push, but the text never says so. VI-C and XI then read the two row-block nulls simply as "agree" and near zero.
   - A second, smaller source of upward push remains. In the group-level version, the moved units are calibrated blocks over all flows. The halves inside a pseudo-context are re-blocked after a random 80/20 split, so block boundaries need not line up with unit boundaries, and a unit can still reach both halves.
   - Both effects push the null up, which makes the reported excess conservative.
   - Fix: one clause, e.g. "…, so this version may push Δ up slightly, which makes the comparison conservative".

3. **The final paragraph of XII is loosely scoped.**
   - Quote: "part of it survives the row-order groupings tested, on each context's own records and on common ones."
   - Own-record gaps are positive at all five row-order settings, but only two settings were tested on common records or against a null. Paragraph 2 of XII states this correctly.
   - Fix: "survives the row-order groupings (at the two settings tested, also on common records and above a grouped null)".

4. **"Almost unchanged" for XGBoost under row blocks (VI-C last paragraph, X-A) is slightly generous.**
   - Quote: "Explaining both contexts on the same records leaves the XGBoost gap almost unchanged".
   - Under orderedcal, the XGBoost gap on common records is 0.063–0.072 against 0.074 on own records, a fall of up to 15% on the Non-VPN records. Under random halves and orderedhalf, it is indeed almost unchanged (within 5%).
   - Fix: "changes the XGBoost gap little (by at most 15%)".

5. **The abstract still does not say that Random Forest loses 22–35% of the row-block gap on common records,** and "under row-block grouping" does not say that only the two strictest settings were tested. This is acceptable for an abstract given the body text; mentioned for completeness.

No number errors and no grammar errors found in the new or changed text.

# Checked

- **Code.** designs.py: `cal()` rescaling, the EXPLAIN_OTHER branch, and the other-context explained records. Random Forest uses the same 800-record sample seed 61000+100r+oi as the other context's own run, and XGBoost uses the full test partition (rec_other shapes are 800 and 1,876 rows). p11_v14_summary.py: the common_records_rowblocks block and the half-size and real-cross additions.
- **Recomputed with scipy directly from ref_<M>_<d>_common.pkl:**
  - XGBoost orderedcal: own 0.0741; Non-VPN, VPN and union records 0.0632, 0.0718, 0.0659; positive in 11 of 11.
  - XGBoost orderedhalf: own 0.0329; common 0.0313, 0.0317, 0.0319; positive in 10, 11 and 10 repeats; minimum −0.011.
  - Random Forest orderedcal: own 0.1528; common 0.1073, 0.1186, 0.1141; positive in 11 of 11.
  - Random Forest orderedhalf: own 0.0933; common 0.0709, 0.0606, 0.0678; positive in 11, 10 and 11 repeats; minimum −0.001.
  - The stored own-record explanations are reproduced exactly (maximum difference 0.0).
  - These support every common-records number in VI-C, X-A, XI, XII and Table XV: 0.063–0.072, 0.031–0.032, 0.107–0.119, 0.061–0.071, 10–11 of 11, and 22–35%.
- **Rescaled nulls recomputed from the pkls:**
  - Random Forest nullgroup_calibrated: +0.0796 (SD 0.1317, −0.097 to 0.356)
  - XGBoost nullstrat_orderedcal: −0.0044 (SD 0.0089, −0.021 to 0.014)
  - Random Forest nullgroup_orderedcal: +0.0232 (SD 0.0307, −0.043 to 0.060)
  - All three match Table VII and p11_v14.json.
- **Table VII.** All 20 null rows (9 cells each) in v14_final.txt match the current ref_summary CSVs, including the rescaled calibrated and orderedcal rows.
- **VI-C numbers.**
  - Row-block null range −0.022 to 0.023: correct.
  - Excess over the group-level null 0.069/0.035 (XGBoost) and 0.130/0.111 (Random Forest): correct.
  - 5–11 real repeats above the largest null repeat: correct.
  - Flow-level cluster nulls −0.095 to −0.009, group-level 0.040 to 0.265, SD up to 0.184: correct.
  - Cross 0.863 against within 0.821: correct.
  - Group-level cross below the real cross in all six settings: correct.
- **Table XV.** All new rows and the corrected null value match p11_v14.json.
- **Other sections.** The hunk-level diff between v13 and v14_final shows only the sections reviewed. Pre-existing figures I spot-checked are still correct, for example the XGBoost family gaps 0.014 and +0.016 in VII.
