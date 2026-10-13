# Task I — Statistical analysis

Bootstrap replicates: 3000; fixed random seed: 20261009.

## Stability inference
Positive `stability_gap` means the rankings are more stable within a context than across contexts. For rank displacement, the sign is reversed (`cross_minus_within`) so positive retains this interpretation.
The bootstrap resamples the Non-VPN and VPN run-level ranking lists independently. It does not treat the 435/900 repeated pairwise comparisons as independent observations, and it does not assume matching seed numbers create matched experimental units.
Primary H1/H2 inference uses basic bootstrap intervals and an approximate one-sided null-centred nonparametric run bootstrap. The pairwise statistic's bootstrap errors are centered at their empirical mean to mitigate finite-sample self-comparison bias. This is not a permutation test; minimum attainable Monte Carlo p-value is 1/(3000+1).

### Primary Spearman stability-gap tests

| hypothesis_level | model | within_mean | cross_mean | stability_gap | ci_95_low | ci_95_high | p_one_sided_raw | p_bh_primary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| feature | RandomForest | 0.9755 | 0.7888 | 0.1867 | 0.1776 | 0.1954 | 0.0003 | 0.0003 |
| feature | XGBoost | 0.9739 | 0.8177 | 0.1562 | 0.1484 | 0.1639 | 0.0003 | 0.0003 |
| family | RandomForest | 0.9713 | 0.9295 | 0.0418 | 0.0178 | 0.0619 | 0.0003 | 0.0003 |
| family | XGBoost | 0.9816 | 0.8951 | 0.0865 | 0.0627 | 0.1086 | 0.0003 | 0.0003 |

Benjamini–Hochberg correction is applied separately to the two model-level primary bootstrap tests within feature-level H1 and family-level H2. Other metric intervals are secondary/descriptive.

Family importance is the sum of member-feature mean absolute SHAP values per run; it is not a fresh grouped-feature SHAP calculation.

## Task F reproduction check

| target_context | model | feature_set | saved_macro_f1 | reproduced_macro_f1 | absolute_difference | matches_within_1e-4 |
| --- | --- | --- | --- | --- | --- | --- |
| novpn | RandomForest | full | 0.9171 | 0.9171 | 0.0000 | 1.0000 |
| novpn | RandomForest | context_specific_top10 | 0.9143 | 0.9143 | 0.0000 | 1.0000 |
| novpn | RandomForest | transferred_top10 | 0.9171 | 0.9171 | 0.0000 | 1.0000 |
| novpn | XGBoost | full | 0.9347 | 0.9347 | 0.0000 | 1.0000 |
| novpn | XGBoost | context_specific_top10 | 0.9382 | 0.9382 | 0.0000 | 1.0000 |
| novpn | XGBoost | transferred_top10 | 0.9331 | 0.9331 | 0.0000 | 1.0000 |
| vpn | RandomForest | full | 0.8533 | 0.8533 | 0.0000 | 1.0000 |
| vpn | RandomForest | context_specific_top10 | 0.8663 | 0.8663 | 0.0000 | 1.0000 |
| vpn | RandomForest | transferred_top10 | 0.8571 | 0.8571 | 0.0000 | 1.0000 |
| vpn | XGBoost | full | 0.8779 | 0.8779 | 0.0000 | 1.0000 |
| vpn | XGBoost | context_specific_top10 | 0.8757 | 0.8757 | 0.0000 | 1.0000 |
| vpn | XGBoost | transferred_top10 | 0.8738 | 0.8738 | 0.0000 | 1.0000 |

## Macro-F1 intervals

Intervals are stratified bootstrap intervals over held-out test records, conditional on the fitted models and selected features.

| target_context | model | feature_set | macro_f1_point | macro_f1_ci_95_low | macro_f1_ci_95_high |
| --- | --- | --- | --- | --- | --- |
| novpn | RandomForest | full | 0.9171 | 0.8990 | 0.9348 |
| novpn | RandomForest | context_specific_top10 | 0.9143 | 0.8961 | 0.9318 |
| novpn | RandomForest | transferred_top10 | 0.9171 | 0.8994 | 0.9336 |
| novpn | XGBoost | full | 0.9347 | 0.9179 | 0.9500 |
| novpn | XGBoost | context_specific_top10 | 0.9382 | 0.9229 | 0.9534 |
| novpn | XGBoost | transferred_top10 | 0.9331 | 0.9163 | 0.9482 |
| vpn | RandomForest | full | 0.8533 | 0.8335 | 0.8723 |
| vpn | RandomForest | context_specific_top10 | 0.8663 | 0.8471 | 0.8853 |
| vpn | RandomForest | transferred_top10 | 0.8571 | 0.8376 | 0.8763 |
| vpn | XGBoost | full | 0.8779 | 0.8603 | 0.8951 |
| vpn | XGBoost | context_specific_top10 | 0.8757 | 0.8576 | 0.8932 |
| vpn | XGBoost | transferred_top10 | 0.8738 | 0.8548 | 0.8918 |

## Consequence contrasts

| target_context | model | contrast | macro_f1_difference | ci_95_low | ci_95_high | p_one_sided_raw_for_H3 | p_bh_H3_top10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| novpn | RandomForest | context_specific_minus_full | -0.0028 | -0.0125 | 0.0067 |  |  |
| novpn | RandomForest | transferred_minus_full | -0.0000 | -0.0105 | 0.0105 |  |  |
| novpn | RandomForest | context_specific_minus_transferred | -0.0028 | -0.0122 | 0.0060 | 0.7271 | 0.7271 |
| novpn | XGBoost | context_specific_minus_full | 0.0035 | -0.0069 | 0.0143 |  |  |
| novpn | XGBoost | transferred_minus_full | -0.0017 | -0.0114 | 0.0080 |  |  |
| novpn | XGBoost | context_specific_minus_transferred | 0.0051 | -0.0042 | 0.0148 | 0.1445 | 0.2890 |
| vpn | RandomForest | context_specific_minus_full | 0.0130 | 0.0020 | 0.0242 |  |  |
| vpn | RandomForest | transferred_minus_full | 0.0038 | -0.0070 | 0.0144 |  |  |
| vpn | RandomForest | context_specific_minus_transferred | 0.0092 | 0.0010 | 0.0176 | 0.0142 | 0.0568 |
| vpn | XGBoost | context_specific_minus_full | -0.0022 | -0.0119 | 0.0079 |  |  |
| vpn | XGBoost | transferred_minus_full | -0.0041 | -0.0159 | 0.0086 |  |  |
| vpn | XGBoost | context_specific_minus_transferred | 0.0019 | -0.0087 | 0.0121 | 0.3627 | 0.4836 |

The four H3-style p-values use one-sided paired prediction-swap permutation tests (20,000 permutations), with Benjamini–Hochberg correction across the four comparisons. Confidence intervals use stratified paired test-record bootstrap resampling.
Caution: Task F uses Top-10 feature subsets, so the four context/model tests of context-specific minus transferred Top-10 are exploratory adaptations. They do not establish the result of a planned Top-15 experiment. The bootstrap does not include training or feature-selection variability.

## Run-level permutation validation of primary stability gaps

A separate permutation validation was performed using the saved Task D feature rankings and the aggregated feature-family rankings. For each model and analysis level, 10,000 random allocations of the 60 saved run-level rankings into two groups of 30 were evaluated. The statistic was mean within-group Spearman similarity minus mean between-group similarity. The reconstructed observed gaps matched the existing Task I gaps within the validation tolerance.

| Analysis level | Model | Observed gap | Permutation p-value | BH-adjusted p-value |
| --- | --- | ---: | ---: | ---: |
| Feature | RandomForest | 0.1867 | 0.000100 | 0.000100 |
| Feature | XGBoost | 0.1562 | 0.000100 | 0.000100 |
| Feature family | RandomForest | 0.0418 | 0.000100 | 0.000100 |
| Feature family | XGBoost | 0.0865 | 0.000100 | 0.000100 |

No sampled permutation produced a gap at least as large as the observed gap. The reported Monte Carlo p-value uses the plus-one correction, so it is not zero. Benjamini–Hochberg correction was applied separately to the two model-level tests within each analysis level.

This validation reuses saved rankings and does not retrain models or recompute SHAP values. Its inference is conditional on the saved runs and depends on the exchangeability of run-level rankings under the null hypothesis. It supplements, rather than replaces, the run-level bootstrap intervals and approximate bootstrap tests reported above. Together, these analyses support the finding that feature and feature-family rankings are more similar across repeated runs within a context than across VPN and Non-VPN contexts.
