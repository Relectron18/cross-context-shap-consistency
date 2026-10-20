# Second and third revisions (responses to Concerns 2 and Concerns 4)

Every number in `SHAP_VPN_revised_v8.docx` (and the earlier `SHAP_VPN_revised_v7.docx`) comes from the scripts in `revision2/src`, run in one environment
(Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0, Linux). Run from `revision2/src`.

| Script | What it does | Paper |
|---|---|---|
| `p1_audit_primary.py` | Data audit (duplicates/label conflicts, entries below -1, class mix), near-duplicate audit with log(x+2), cluster-grouped CV, baselines, primary TreeSHAP, cross-context agreement incl. kappa_k | III, V, VI-A/B |
| `designs.py <RF/XGB> <random/grouped10/grouped35/grouped/calibrated/ordered10/ordered35/ordered140/orderedcal/orderedhalf/null> 11` | Within-context reference designs; 80/20 partition redrawn each repeat. `calibrated` sets the clusters per class to the number of capture files in the capture-prefix list of munhouiani's independent Deep-Packet implementation on GitHub (paper ref. [56]; counts in `designs_meta.py`; the Deep Packet paper itself, ref. [7], gives no such list) (BROWSING: flows / median flows per capture). `ordered*` keep whole blocks of consecutive rows of the original ARFF files together (needs `results/arff_order_positions.json`) | IV-D, VI-C, VII |
| `summarize_refs.py` | Gaps (rho, RBO, Top-5, kappa_5, class-weighted, four family rules) per design | Tables VII, X |
| `p2_consequence.py` | Direct model transfer; Top-5/10/15 list transfer with 5 selection seeds, bottom-10, 20 fixed random-10 lists | VIII |
| `p5_concerns4.py` | Block-summed SHAP shares of the three minimum IATs, conditional permutation importance, list-origin x target decomposition of the Top-k transfer, class-mix SDs, RF 800-record check on half-models -> `results/p5_results.json`, `results/p5_topk_2x2.csv` | VI-D, VI-E, VIII-B, IV-D |
| `p6_arff_order.py "<folder with the two 15 s ARFF files>"` | Maps every deduplicated flow to its row in the original Scenario A2 ARFF files and measures how alike neighbouring rows are -> `results/arff_order_positions.json`, `results/arff_order_locality.csv` | IV-D |
| `p4_classmix_miniat.py` | Class-mix swap, matched 1,330 with same-size reference, drop-one retraining, permutation importance, entries below -1 | VI-D, VI-E |
| `p3_sentinel_robust.py`, `p3c_et.py`, `p3d_finalize.py` | Sentinel S1/S2, RF/XGB/ET x path/interventional, XGB seeds and depth | IX |
| `figs.py` | Figures 2, 3, 4, 5, 7, 8 (Figs 1 and 6 unchanged) | |
| `p0_arff_counts.py "<Scenario A2-ARFF folder>"` | Class counts in the original ARFF files (BROWSING capped at 2,500 in every file; Non-VPN P2P at 1,000) -> `results/arff_class_counts.csv` | III-B |

XGBoost is run single-threaded (`c2.make_model`); thread count does not change its predictions.

## Building the paper

`paper_build/` regenerates `paper/SHAP_VPN_revised_v8.docx` and `paper/Response_to_Reviewer_v8.docx` from the result
files, so every number in the text comes straight from `revision2/results`. It uses `paper/SHAP_VPN_final_v6_merged.docx`
(the v6 manuscript with merged runs) as the template for styles, equations, Table I/II and Figs 1 and 6.

```bash
cd revision2/paper_build
python build.py       # -> paper/SHAP_VPN_revised_v8.docx (citations renumbered by first appearance)
python response4.py   # -> paper/Response_to_Reviewer_v12.docx
python response.py    # -> paper/Response_to_Reviewer_v7.docx (previous round)
```

`results/*.pkl` hold per-record |SHAP| for every half-model of the reference designs; `primary_*.npy` the primary
explanations; `logs/` the console output of each run.

`build.py` calls `mathfix.py`, which sets every mathematical symbol (Greek letters, relations, primes) as an inline Word
equation (OMML) and replaces Unicode minus signs, non-breaking spaces and arrows with ASCII. It also sets the document
language to English (US) in every script slot. Outside equations the running text then contains only ASCII, typographic
quotes and dashes, and accented letters in author names; this avoids the "unsupported language" and
"characters from other alphabets" flags that plain-text Greek letters and U+2212 minus signs raise in similarity checkers.
The v7 manuscript was built by the version of `paper_build/` at commit 1f56a71.

## Team-edited versions (v9, v10)

`paper/SHAP_VPN_revised_v14.docx` is the current version (v13 with an editorial pass and two new analyses, `paper_build/fix_v14.py`; v13 = v12 corrected after the standalone defect review, `paper_build/fix_v13.py`; v12 = v11 revised after the in-project review, `paper_build/fix_v12.py`; v11 = v10 with the team's own conclusion, `paper_build/fix_v11.py`). The team rephrased 38 passages of v8 by hand
(saved as `paper/SHAP_VPN_v8_team_edited.docx`); `paper_build/fix_v9.py` takes that file and corrects what the
rephrasing broke (factual slips, LaTeX left as text, plain-text Greek letters, citation brackets, split paragraphs,
spacing and typos) while keeping the team's wording elsewhere. No number in the text or tables differs from v8.

```bash
cd revision2/paper_build
python fix_v9.py "<team-edited docx>" ../../paper/SHAP_VPN_revised_v9.docx
```

The team then edited v9 again (saved as `paper/SHAP_VPN_v9_team_edited.docx`: twelve passages reworded, Section III-D
deleted, en dashes turned into hyphens). `paper_build/fix_v10.py` (helpers in `edits_lib.py`) keeps that wording,
restores Section III-D and the conclusion, re-adds lost equation symbols and fixes the remaining slips -> v10.

```bash
python fix_v10.py ../../paper/SHAP_VPN_v9_team_edited.docx ../../paper/SHAP_VPN_revised_v10.docx
```

## v12: revision after the in-project review

`paper/reviews/` holds a constructive and an adversarial reading of v11 by separate AI sessions (not an outside review)
and the collated issue list. `src/p7_audit_v12.py` reads existing results only (no model is refitted) and writes
`results/p7_audit.json` (one-sided gaps, sign of RBO/Top-5 gaps in the first-versus-second-half split, permutation-importance
cross-context agreement, family gaps, row-order locality on deduplicated flows); `results/p7_distinct_lists.json` counts the
distinct Top-k lists produced by the five selection seeds. `paper_build/fix_v12.py` applies the text changes, adds the data
and code availability section and two references (Kendall 1945; the public Deep-Packet implementation that supplied the
capture-file prefix list) and renumbers references by first appearance (59 references, checked in order).

```bash
cd revision2/src && python p7_audit_v12.py
cd ../paper_build && python fix_v12.py ../../paper/SHAP_VPN_revised_v11.docx ../../paper/SHAP_VPN_revised_v12.docx && python response4.py
```

## v13: corrections after the standalone defect review of v12

`paper/reviews/2026-10-20-v12-defect-review.md` lists 30 defects found by a separate AI session (not an outside review);
`paper/reviews/2026-10-20-v13-audit-ledger.md` records, for each one, the evidence, whether it was confirmed, the fix
and how it was checked.

Code change: `revision/src/common.py` `ranks()` now gives tied values their average rank (`method="average"`), as the
paper states. Up to v12 it used `method="first"`, which breaks ties by feature order. SHAP importances have no ties
(checked for the primary fits, all 1,936 reference-design vectors and the robustness runs), so every SHAP result is
unchanged; only the cross-context agreement of permutation importance, whose negative values are set to zero and so
tie, changes. `p4_results.json` and `p5_results.json` keep the old values for audit; the corrected values are in
`results/p8_v13.json`.

| Script | What it does |
|---|---|
| `src/p8a_rf_interv.py <novpn/vpn>` | Interventional TreeSHAP for the primary Random Forest on the same 500 records and 200-record background as the robustness check, so Extra Trees and Random Forest can be compared under the same formulation (`results/rf_interv_rec_*.npz`) |
| `src/p8b_cpi_full.py` | Re-runs the conditional permutation importance of `p5_concerns4.py` with the same seeds at full precision and checks it reproduces the stored rounded values (`results/p8b_cpi_full.json`) |
| `src/p8_v13_audit.py` | Recomputes every number the v13 corrections use from existing results (tie counts, permutation-importance agreement with average ranks, min_flowiat shares, drop-one shares, class weighting, half sizes, locality by seed, Table VII and XI means rounded once, Extra Trees vs Random Forest, XGBoost thread and tree-method checks) -> `results/p8_v13.json` |
| `src/check_v13_numbers.py <docx>` | Checks that the corrected numbers appear in the manuscript as derived, that retracted statements are gone, and every numeric cell of Table VII; exits 1 on v12 and on planted digit errors |
| `paper_build/fix_v13.py` | Applies the text and table corrections to v12 (numbers read from `p8_v13.json`), normalises formatting left by the team edits, and writes v13 |

```bash
cd revision2/src && python p8a_rf_interv.py novpn && python p8a_rf_interv.py vpn && python p8b_cpi_full.py && python p8_v13_audit.py
cd ../paper_build && python fix_v13.py ../../paper/SHAP_VPN_revised_v12.docx ../../paper/SHAP_VPN_revised_v13.docx
cd ../src && python check_v13_numbers.py ../../paper/SHAP_VPN_revised_v13.docx
```

## v14: editorial pass and two new analyses

Change log: `paper/reviews/2026-10-20-v14-change-log.md`.

| Script | What it does |
|---|---|
| `src/designs.py <RF/XGB> nullgroup_<design> 11` | Grouped permutation null, group-level: pseudo-contexts built from whole capture-sized groups (calibrated clusters or calibrated row blocks per class and source file), then the grouped design within each class and source file. Used in the paper. |
| `src/designs.py <RF/XGB> nullstrat_<design> 11` | Grouped permutation null, flow-level: flows reassigned one by one, grouped design within each class and source file. Used in the paper as the second version. |
| `src/designs.py <RF/XGB> null_<design> 11` | First version (superseded, kept for the record): the halves of a pseudo-context can differ in source-file mix. |
| `src/p12_null_origin_check.py` | Per-class difference in Non-VPN/VPN source mix between the two halves of each null version (`results/p12_null_origin_check.json`). |
| `src/p9_common_records.py <RF/XGB>` | Refits the random-halves design with its seeds and explains every half-model on both contexts' records; reproduces the stored design exactly (`results/p9_common_*.{csv,json}`). |
| `src/p10_leftover_checks.py` | RF sentinel fits: no tied importances, stored values reproduced; XGBoost `auto` = `hist` on both contexts. |
| `src/p11_v14_summary.py` | Summaries of both analyses -> `results/p11_v14.json` (the only source of v14's new numbers). |
| `src/check_v14_numbers.py <docx>` | Runs the v13 checks, checks the new numbers and the 20 new Table VII rows, and that retired statements are gone; fails on v13 and on planted errors. |
| `paper_build/fix_v14.py` (+ `fix_v14_results.py`) | Editorial pass with a guard that stops the build if a stylistic edit changes any number or citation; new-results text; Fig. 2 replaced by the version from `src/figs.py 2`. |

Grouped designs with `designs.py` are unchanged: the existing designs reproduce the stored repeat 0 exactly after the
changes (`REF_OUT=<path> python designs.py XGB <design> 1`, then compare with `results/ref_XGB_<design>.pkl`).

```bash
cd revision2/src
for d in grouped10 grouped35 calibrated orderedcal orderedhalf; do for v in nullgroup nullstrat; do
  python designs.py XGB ${v}_$d 11; python designs.py RF ${v}_$d 11; done; done
python p9_common_records.py XGB && python p9_common_records.py RF && python p10_leftover_checks.py && python p11_v14_summary.py
python figs.py 2
cd ../paper_build && python fix_v14.py ../../paper/SHAP_VPN_revised_v13.docx ../../paper/SHAP_VPN_revised_v14.docx
cd ../src && python check_v14_numbers.py ../../paper/SHAP_VPN_revised_v14.docx
```
