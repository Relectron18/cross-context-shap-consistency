# Revision analyses (response to reviewer concerns)

Scripts added for the revised paper (SHAP_VPN_final_v6). Everything added for the revision lives in this `revision/` folder:
`revision/src/` (scripts), `revision/results/` (summary outputs) and `revision/figures/` (new and updated figures).
Run scripts from the repository root, e.g. `python revision/src/e1_halfsplit.py XGB 11`. Outputs go to `revision/results/`.
Existing scripts and results were not changed.

| Script | Paper section | What it does |
|---|---|---|
| e0_primary.py, e0b_saved_xgb.py | check | Refits primary RF (reproduces exactly) and explains the saved XGBoost models |
| e0c_prov.py `<path to Scenario A2-ARFF>` | III-A | Confirms the processed files come from the 15 s Scenario A2 ARFF files after duplicate removal |
| e1_halfsplit.py `<RF/XGB> <repeats>` | IV-D, VII-B, Table IX | Disjoint, size-matched half-split reference with separate model seeds (run: XGB 11 repeats, RF 3) |
| e3_light.py | VI-B, VI-C, VII-A, VIII, Table XIII | Class-weighted importance, sentinel prevalence, RBO/L1 for primary and bootstrap runs, feature correlations, correlation clusters, per-member family means |
| e2_fast.py | III-C, V, VII-F, XI-B | Log/rank near-duplicate audit, LR convergence and log refit, class-matched models, matched-vs-natural draws, XGB seeds and depth settings, cluster-grouped CV |
| e7_consequence.py then e7b.py | IV-F, IX, Table XV | Top-k transfer with bottom-10 and random-10 controls (e7_consequence.py was run for the Non-VPN XGBoost target; e7b.py ran the remaining targets with fewer random subsets) |
| s4_tablexiv.py | XI-A, Table XVII | Re-runs the XGB/RF path bootstrap and reports percentile vs basic intervals |
| s5_sentinel_cross.py | X | Cross-context agreement under S1/S2 |
| make_figures.py | Fig. 2, Fig. 6 | Redraws the pipeline figure and the feature-correlation heatmaps |
| summarize.py | — | Builds halfsplit_summary.json and consequence_summary.json |

Environment for these runs: Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0 (Linux).
XGBoost refits can differ from the macOS results by up to about 0.005 macro F1.
Large intermediate files (per-record SHAP arrays, .pkl) are not included; rerun the scripts to regenerate them.
