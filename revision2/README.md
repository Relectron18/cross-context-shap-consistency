# Second and third revisions (responses to Concerns 2 and Concerns 4)

Every number in `SHAP_VPN_revised_v8.docx` (and the earlier `SHAP_VPN_revised_v7.docx`) comes from the scripts in `revision2/src`, run in one environment
(Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0, Linux). Run from `revision2/src`.

| Script | What it does | Paper |
|---|---|---|
| `p1_audit_primary.py` | Data audit (duplicates/label conflicts, entries below -1, class mix), near-duplicate audit with log(x+2), cluster-grouped CV, baselines, primary TreeSHAP, cross-context agreement incl. kappa_k | III, V, VI-A/B |
| `designs.py <RF/XGB> <random/grouped10/grouped35/grouped/calibrated/ordered10/ordered35/ordered140/orderedcal/orderedhalf/null> 11` | Within-context reference designs; 80/20 partition redrawn each repeat. `calibrated` sets the clusters per class to the number of capture files in the Deep Packet prefix list (BROWSING: flows / median flows per capture). `ordered*` keep whole blocks of consecutive rows of the original ARFF files together (needs `results/arff_order_positions.json`) | IV-D, VI-C, VII |
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
python response4.py   # -> paper/Response_to_Reviewer_v10.docx
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

`paper/SHAP_VPN_revised_v10.docx` is the final version. The team rephrased 38 passages of v8 by hand
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
