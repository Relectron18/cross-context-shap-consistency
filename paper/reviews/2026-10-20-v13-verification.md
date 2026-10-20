An in-project reading by a separate AI session. It is not an outside review.

Scope: v13 text /tmp/claude-0/rev13/v13.txt (MD5 1a2d9dc9889a8b904b498fe6b3dc099d, written 04:54:42 together with paper/SHAP_VPN_revised_v13.docx, MD5 dee035b076feb8a3cca71404497999fe), compared with v12.txt using a fresh word-level diff. Numbers were checked against revision2/results (p8_v13.json MD5 f1fe2fa2293c0ed71595c6bcc90bde32, plus the per-repeat CSVs, npz/npy and p1–p5/p7 JSON files) and against the code, using read-only Python. Note: v13.txt, p8_v13.json and p8_v13_audit.py were rewritten while this review was running (04:53–04:54). The only text change from that rewrite is the Table III thread sentence, which is new problem 1 below. No project file was modified.

## Verdicts on the 30 defects

| # | Defect (short) | Verdict | Evidence / remark |
|---|---|---|---|
| 1 | PI cross-context rho used first-ranks; IV-C misdescribed ties | Fixed correctly | Recomputed from p4/p5 values: average ranks after clipping give 0.523/0.695/0.618/0.557; unclipped raw values give 0.590/0.669/0.565/0.563. RF tie counts are 10/10 (marginal) and 9/12 (conditional). common.ranks() now uses method="average". The IV-C text is accurate. p8b reproduces the conditional PI. |
| 2 | RF min-IAT "change in what the models rely on" | Fixed correctly | VI-E, X-C and XII now call the rise SHAP-specific and give 0.168→0.155 and 0.028→0.012 (p4 values verified). The leftover phrase "relies on" is gone. |
| 3 | IX ET-interv vs RF-path range | Fixed correctly | From et_rec_*.npz and rf_interv_rec_*.npz: path/path 0.730/0.801, interv/interv 0.743/0.731, so the 0.731–0.743 range is correct. The RF_path arrays are identical in both files, and the RF-interv cross rho (0.802) reproduces p3. |
| 4 | Matched training XGB-only; wrong comparator | Fixed correctly | IV-E, VI-D, the Table II note and XI now say XGBoost. 0.786 (primary) and 0.825 (random-half cross) match p1 and ref_summary. |
| 5 | Table II matched Test columns | Fixed correctly | The Test cells are now "—", and the note says matched models are explained on the full test partition, which matches p4 fit_imp (D[c][1]). |
| 6 | Drop-one 2.2/1.5 include renormalisation | Fixed correctly | 1.634/0.839 renormalised and 2.24/1.55 raw against a min_fiat base of 8.15%, all matching the text. |
| 7 | min_flowiat 43%/31% denominator | Fixed correctly | Recomputed from data: 48.46%/33.51% defined (printed as 48%/34%) and 42.6%/31.0% overall. 0 exceedances. |
| 8 | Class weighting "same level" | Fixed correctly | rho 0.821/0.792; RF RBO 0.580→0.732, L1 0.333→0.270; Top-10 9→8 and 8→7. All verified. |
| 9 | ~10-flow clusters "barely changed" | Fixed correctly | XGB 0.832 vs 0.825, RF 0.730 vs 0.766. |
| 10 | VII "Therefore … does not fully explain" | Fixed correctly | The claim is now scoped to the finest grouping. The XGB family gap is 0.075→0.077 at ~10 flows, 0.014 at ~35 and 0.016 calibrated (CSV). |
| 11 | VI-F "again steadier" under row blocks | Fixed (minor leftover) | The caveat is added. The next sentence, "indicate the same direction", now has no clear antecedent (new problem 4). |
| 12 | Clusters overcorrect / blocks "do not exploit features" | Fixed correctly | VI-C, X-A and XII now mention the position–feature correlation and the 0.73–0.80 F1 (Table VII orderedhalf 0.800/0.752/0.779/0.732). See new problem 5 for a mild tension. |
| 13 | "none of the designs controls for this" | Fixed correctly | designs.py null keeps per-class counts and explains each pseudo-context on its own test partition, so the IV-D sentence is accurate. +0.028 and -0.001 match the CSVs. |
| 14 | VIII-A "same range" | Fixed correctly | 0.430/0.438 vs 0.47–0.64 (p2, p1). |
| 15 | Swap "does not bring closer" | Fixed, but the new supporting sentence has a wrong number | "materially" is added. The new sentence's SD range mixes RF and XGB values (new problem 2). |
| 16 | XII "at least 10" | Fixed correctly | Now "10 or 15". |
| 17 | IV-D row order stated as fact | Fixed correctly | Now "appear to be in the order". |
| 18 | P2P described three ways; BROWSING "at all" | Partly fixed | The P2P wording is now consistent (no Non-VPN P2P listed; counted as one; "at most one listed"). 34–40% is on training flows (p8 39.6/33.8). IV-D still says "cannot be divided by capture at all", while XI now says "with the information available" (new problem 3). |
| 19 | IV-B "always" test partition vs IV-G | Fixed correctly | |
| 20 | VI-C "only the outer partition and seeds" | Fixed correctly | |
| 21 | Clusters "assigned at random" | Fixed correctly | The text matches grouped_halves: random order, stable sort by size descending, each cluster to the half with fewer flows. k-means n_init=3 with random_state=seed+k, as stated. |
| 22 | Locality range seed-dependent | Fixed correctly | p7 gives 0.37–0.76; p8 seeds 0–3 give 0.362–0.747. The union matches "about 0.36–0.76". |
| 23 | 2.1 vs 2.15; medians | Fixed correctly | RF Non-VPN alt-family spread 25.147−23.001=2.146. Medians 44.5/245.5. |
| 24 | Missing reproducibility details | Mostly fixed | Now stated: ties, tree_method, CPI partners and quintiles, k-means settings for the grouped designs, row-block half sizes (3,264–3,476 verified), and that S2 applies to every feature. Code-on-request versus the mapping is now consistent. Still missing: the k-means settings of the 50-cluster grouped CV in III-C (p1 line 52: n_init=3, random_state=0). This is minor. |
| 25 | CPI presented as the method of [36] | Fixed correctly | "simplified conditional variant inspired by [36]". The partner and quintile description matches p5 cond_pi. |
| 26 | No record of thread check; clusters from earlier run | Records fixed; text now wrong | p8 records 1-vs-2-thread identity, and clusters_0.2 regenerated identical. The rewritten Table III note is false (new problem 1). |
| 27 | [49] and Webber "Eq. 21" | Not changed; unverified | Webber's prefix-weight formula for p=0.9, d=10 evaluates to 0.856, which matches "86%". From memory, that formula is Eq. (21) and RBO_ext is Eq. (32), so the code comment (eq. 32) and the paper (Eq. 21) cite different equations and do not conflict. [49] venue and DOI could not be confirmed with one web search. |
| 28 | Prefix list credited to Lotfollahi in code | Fixed correctly | designs.py, designs_meta.py and the README credit munhouiani [56]. Reference [56] is that repository. |
| 29 | Language items (IV-G, II-C, X-E, VI-B) | Fixed correctly | The IV-G rewrite matches p2_consequence.py: 80/20 stratified inner split; selector on inner-train; SHAP on inner-val (RF: 800 stratified); top-k; refit on the full training partition; scored on the test partition; transferred list uses the same seed index. "give them equal weight" is removed. |
| 30 | Apostrophes, macro F1, dataset, citation order, asterisk | Fixed correctly | One remaining ’ is inside the quoted title of a reference. Tables I–XV and Figs. 1–8 are first cited in order. "*Corresponding author." is added. |

## New problems in v13 (most severe first)

1. **Table III note: the thread statement is false, and the check that supports it is vacuous.**
   Quote: "XGBoost used one thread in the repeated designs and all available cores in the other fits; refitting the primary models with one thread and with two gave identical predictions."
   Evidence:
   - Every revision2 script imports `make_model` from c2.py. Its wrapper sets `kw.setdefault("n_jobs", 1)` for XGB, and I confirmed this in the environment: `make_model('XGB').get_params()['n_jobs'] == 1`.
   - p1_audit_primary.py fits the primary models with `make_model(mname)`, so they ran single-threaded. No revision2 XGB fit passes n_jobs=-1.
   - In p8_v13_audit.py, `pall = make_model("XGB")` (comment: "the primary models use n_jobs=-1") is therefore also one thread, so `all_cores_vs_one_max_abs_prob_diff = 0.0` compares one thread with one thread. fix_v13.py asserts on this value.
   - The v12 wording ("run single-threaded") matched the code; v13 made it wrong.
   - The real 1-vs-2-thread check is fine, and this machine has 2 cores (`n_cores = 2`).

   Fix: restore "XGBoost was run single-threaded (n_jobs = 1); refitting the primary models with two threads gave identical predictions". Change the p8 check to pass `n_jobs=-1` explicitly, or drop the all-cores field.

2. **VI-D: the SD range for the XGBoost moves includes a Random Forest SD.**
   Quote: "The two XGBoost changes are small moves toward the other context, no larger than the spread across draws (SD 0.009–0.024)."
   Evidence:
   - fix_v13.py takes the min and max over all four model/context entries of classmix_delta.
   - The XGB SDs (Table VIII, "ρ with other") are 0.012, 0.017, 0.024 and 0.015. The 0.009 belongs to RF/Non-VPN swapped.
   - The XGB VPN change (+0.017) also exceeds that row's swapped-mix SD (0.015). The SD of the paired swapped-minus-natural differences is 0.018 (Non-VPN) and 0.029 (VPN).

   Fix: "(SD 0.012–0.024 across draws)", or compare with the SD of the paired differences (0.018 and 0.029).

3. **IV-D keeps the overstatement that XI removed (residual of item 18).**
   Quote (IV-D): "BROWSING and P2P, which hold 34–40% of the training flows, cannot be divided by capture at all".
   XI now says: "cannot be split by capture with the information available".
   Fix: use the XI wording in IV-D.

4. **VI-F: "the same direction" has lost its antecedent.**
   The paragraph now ends "Which context gives the steadier rankings therefore depends on the grouping, and neither proxy settles it." It is followed by "The capture counts of Section IV-D indicate the same direction: …", and no direction has just been stated.
   Fix: e.g., "The capture counts of Section IV-D point to redundancy as the explanation: …".

5. **Mild tension about half-model accuracy under row blocks (low).**
   - VI-C says the block half-models "remain accurate (macro F1 0.73–0.90)", and X-A says the blocks "leave the half-models accurate".
   - The new VI-C sentence uses the same 0.73–0.80 for first-versus-second halves as a sign that the split "may also separate regions of feature space" (random halves: 0.84–0.91).

   This is not a contradiction, but a reader sees the same numbers called both accurate and degraded.
   Fix: "remain far more accurate than under clusters (0.73–0.90)".

No other new numeric errors were found. Every changed number in the text and tables matches the result files.

## Task 3: rank tie handling

- **SHAP results are unchanged.** p8 reports `ties_shap` = 0 for the primary fits, all 1,936 reference vectors and the ET/RF robustness vectors, and `ref_summary_max_abs_change_with_average_ranks` = 1.1e-16. That recomputation covered 5 of the 11 designs; the other 6 are covered by the no-tie check.
- **Independent check.** I found 0 exact ties in all 1,936 reference vectors (smallest relative gap between adjacent importances 7e-7) and in the 4 primary vectors.
- **Configurations p8 did not check.** I refitted XGB at depth 3 and depth 10, XGB under S1 and S2 (23 original features), and three 1,330-flow XGB models per context: none has tied importances. RF under S1/S2 was not rechecked.
- **Other quantities that depend on tie handling:**
  - Only the PI cross-context ρ/τ is reported, and it has been corrected.
  - p4 `rho_vs_shap` (PI versus SHAP within a context) also changes, but it is not reported in the paper.
  - PI ranks of the three minima are unaffected, because they are positive and distinct.
  - PI and SHAP top-k lists and RBO use sort_values, not ranks(), and are unaffected where values are positive.
  - Sentinel indicator features tie at zero (3–8 per XGB S1/S2 fit), but rankings compare only the 23 original features.
  - Family ρ (scipy spearmanr) and the drop-all-three ρ (pandas .rank()) already used average ranks.

## Checked

- **Diff.** Fresh word-level diff of v12 against v13; every change was reviewed.
- **Table VII.** All 176 cells recomputed from ref_summary_*.csv with single rounding: all match. That rounding is the source of the undisclosed changes 0.751→0.752, 0.296→0.297, 0.947→0.946, 0.944→0.943, −0.002→−0.001, −0.018→−0.019, (0.029)→(0.030), 0.743→0.742 and +0.046→+0.047. Fig. 7 labels (figs.py) use the same CSV means.
- **Table XI.** All values recomputed from the CSVs, including 0.755 and +0.055.
- **Values verified against result files or the data:**
  - PI ρ (both variants) and tie counts
  - ET versus RF, path and interventional
  - min_flowiat shares
  - drop-one shares
  - class-weighting values
  - matched comparators
  - grouped10 cross values
  - half sizes for every design
  - locality by seed
  - BROWSING+P2P share
  - transfer F1 against grouped CV
  - class-mix deltas and SDs
  - alternative-family spread
  - flows-per-capture medians
  - capture counts (109/30)
  - tree_method (auto, identical to hist for Non-VPN)
  - clusters_0.2 regenerated identical
- **Code against text:** grouped_halves, the k-means settings, null construction, cond_pi partners and quintiles, the p2 selection pipeline, and the c2 thread default.
- **Table II.** The matched rows' "Train %" column is training-only.
- **Citation order.** First citations are in order for Tables I–XV and Figs. 1–8.
- **Number checker.** check_v13_numbers.py passes on the v13 docx.
- **Not verified:**
  - [49] bibliographic details.
  - The content of the munhouiani prefix list (the p8 "no Non-VPN P2P prefix" entry is hard-coded).
  - The data-availability claim that "every table and figure that reports results was generated from these result files by the same code".
  - Figures as images.
