# Second revision (response to Concerns 2)

Every number in `SHAP_VPN_revised_v7.docx` comes from the scripts in `revision2/src`, run in one environment
(Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0, Linux). Run from `revision2/src`.

| Script | What it does | Paper |
|---|---|---|
| `p1_audit_primary.py` | Data audit (duplicates/label conflicts, entries below -1, class mix), near-duplicate audit with log(x+2), cluster-grouped CV, baselines, primary TreeSHAP, cross-context agreement incl. kappa_k | III, V, VI-A/B |
| `designs.py <RF/XGB> <random/grouped10/grouped35/grouped/null> 11` | Within-context reference designs; 80/20 partition redrawn each repeat | IV-D, VI-C, VII |
| `summarize_refs.py` | Gaps (rho, RBO, Top-5, kappa_5, class-weighted, four family rules) per design | Tables VII, X |
| `p2_consequence.py` | Direct model transfer; Top-5/10/15 list transfer with 5 selection seeds, bottom-10, 20 fixed random-10 lists | VIII |
| `p4_classmix_miniat.py` | Class-mix swap, matched 1,330 with same-size reference, drop-one retraining, permutation importance, entries below -1 | VI-D, VI-E |
| `p3_sentinel_robust.py`, `p3c_et.py`, `p3d_finalize.py` | Sentinel S1/S2, RF/XGB/ET x path/interventional, XGB seeds and depth | IX |
| `figs.py` | Figures 2, 3, 4, 5, 7, 8 (Figs 1 and 6 unchanged) | |
| `p0_arff_counts.py "<Scenario A2-ARFF folder>"` | Class counts in the original ARFF files (BROWSING capped at 2,500 in every file; Non-VPN P2P at 1,000) -> `results/arff_class_counts.csv` | III-B |

XGBoost is run single-threaded (`c2.make_model`); thread count does not change its predictions.

## Building the paper

`paper_build/` regenerates `paper/SHAP_VPN_revised_v7.docx` and `paper/Response_to_Reviewer_v7.docx` from the result
files, so every number in the text comes straight from `revision2/results`. It uses `paper/SHAP_VPN_final_v6_merged.docx`
(the v6 manuscript with merged runs) as the template for styles, equations, Table I/II and Figs 1 and 6.

```bash
cd revision2/paper_build
python build.py      # -> paper/SHAP_VPN_revised_v7.docx (citations renumbered by first appearance)
python response.py   # -> paper/Response_to_Reviewer_v7.docx
```

`results/*.pkl` hold per-record |SHAP| for every half-model of the reference designs; `primary_*.npy` the primary
explanations; `logs/` the console output of each run.
