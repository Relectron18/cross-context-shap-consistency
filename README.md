# Cross-Context Consistency of SHAP Feature Importance (VPN vs Non-VPN)

Code, data splits and results for the paper *"Cross-Context Consistency of SHAP-Based Feature Importance for
Application Classification Across VPN and Non-VPN Traffic"* (P. Gomase, G. Khekare, P. Kothari — VIT Vellore).

The study trains Random Forest and XGBoost models separately on Non-VPN and VPN flows of the ISCX VPN-nonVPN
benchmark (seven application classes, 23 time-related features), explains them with TreeSHAP, and asks whether the
feature rankings agree across the two contexts more or less than models trained on different data from the same
context.

## Final manuscript

**`paper/SHAP_VPN_revised_v20.docx` is the final version of the paper.** Every number in it is checked against the
result files by `revision2/src/check_v14_numbers.py` (passes on v20). The architecture diagram is in
`revision2/figures/architecture.{png,svg}` (Graphviz source `architecture.dot`). Earlier versions (v6–v19), the
reviewer concerns and the responses are kept in `paper/` for the record; `revision2/README.md` describes how each
version was built.

## Data

* Source: ISCX VPN-nonVPN (ISCXVPN2016), Canadian Institute for Cybersecurity, University of New Brunswick —
  https://www.unb.ca/cic/datasets/vpn.html (Draper-Gil et al., ICISSP 2016).
* Files used: Scenario A2, 15 s flow timeout (`TimeBasedFeatures-Dataset-15s-NO-VPN.arff`,
  `TimeBasedFeatures-Dataset-15s-VPN.arff`). Exact duplicate rows were removed within each context
  (8,965 → 8,694 Non-VPN; 9,793 → 9,380 VPN flows), and each context was split once into stratified 80/20
  train/test partitions.
* `data/novpn`, `data/vpn`: the train/test partitions used throughout (raw and standardised features).
  `data/controlled`: class-matched subsets (190 train / 48 test flows per class).
* Please cite the dataset authors if you reuse the data.

## Repository layout

| Path | Contents |
|---|---|
| `src/models/` | Baseline training (RF, XGBoost, Logistic Regression) |
| `SHAP/` | Primary TreeSHAP explanations, rankings and plots |
| `src/stability/` | 30 class-stratified training-bootstrap runs per context |
| `src/group_shap/` | Six semantic feature families |
| `src/consequence_test/` | Training-only Top-10 selection and transfer |
| `src/sentinel/` | Sentinel (−1) encodings S1/S2 |
| `src/robustness/` | Extra Trees and interventional TreeSHAP checks |
| `src/statistical_analysis/` | Bootstrap and test-record resampling summaries |
| `src/preprocessing/` | Leakage, near-duplicate, label-conflict and class-distribution audits |
| `results/` | Outputs of the scripts above |
| `models/` | Saved XGBoost and Logistic Regression models (Random Forests are not stored; they refit exactly with `random_state=42`) |
| `revision/` | **Analyses added in response to reviewer concerns** — disjoint half-split reference, class-matched training and class weighting, RBO/L1 measures, correlation clusters, random/bottom-10 controls, log/rank near-duplicate audit, cluster-grouped CV, seed and hyperparameter checks, revised figures. See `revision/README.md`. |
| `revision2/` | **Second revision (Concerns 2)** — every result recomputed in one environment; cluster-grouped half-splits, permutation null, partition redraws, direct model transfer, Top-5/10/15 transfer over five seeds, class-mix swap, drop-one and permutation importance, ARFF class counts; scripts that build the revised paper. See `revision2/README.md`. |
| `paper/` | **Final manuscript `SHAP_VPN_revised_v20.docx`**; earlier versions v6–v19 (including the team-edited uploads), reviewer concerns (Concerns2, Concerns4), responses to the reviewer, and `reviews/` (in-project readings by separate AI sessions, not outside reviews) |

## Running

```bash
pip install -r revision/requirements.txt
python src/models/train_baseline.py          # baselines
python SHAP/run_shap_task_c.py               # primary SHAP rankings (expects models in models/models_CN_project; see script)
python src/stability/run_task_d.py           # training-bootstrap runs
python revision/src/e1_halfsplit.py XGB 11   # revision: half-split reference (see revision/README.md for the rest)
```

Scripts are run from the repository root. Main results were produced with Python 3.14, scikit-learn 1.9.0,
XGBoost 3.4.1 and SHAP 0.52.0 (macOS); revision analyses with scikit-learn 1.9.1 and SHAP 0.53.0 (Linux).
XGBoost refits can differ across platforms by up to about 0.005 macro F1; the saved XGBoost models reproduce the
paper's numbers exactly.
