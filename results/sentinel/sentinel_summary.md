# Task G — Sentinel Analysis

## Experimental conditions
- S1: retain `-1` and add sentinel indicators.
- S2: convert `-1` to `NaN` and retain the same indicators.
- RF and XGBoost are retrained separately for Non-VPN and VPN.
- Evaluation uses the untouched original test labels and rows.
- SHAP ranking comparisons are restricted to the original 23 features.

## Performance summary
context model condition  accuracy  macro_f1  weighted_f1
  novpn    RF        S1  0.945946  0.919399     0.945235
  novpn    RF        S2  0.945946  0.919638     0.945292
  novpn   XGB        S1  0.958022  0.932917     0.957681
  novpn   XGB        S2  0.957447  0.932971     0.957166
    vpn    RF        S1  0.882196  0.854811     0.881744
    vpn    RF        S2  0.879531  0.853074     0.879165
    vpn   XGB        S1  0.904584  0.882938     0.904348
    vpn   XGB        S2  0.904584  0.880793     0.904328

## S1 vs S2 SHAP ranking stability
context model comparison  spearman  kendall  rank_displacement  top5_overlap  top10_overlap
  novpn    RF   S1_vs_S2  0.914032 0.770751           2.086957             5              8
  novpn   XGB   S1_vs_S2  0.961462 0.881423           1.217391             5             10
    vpn    RF   S1_vs_S2  0.887352 0.731225           2.347826             4              9
    vpn   XGB   S1_vs_S2  0.942688 0.841897           1.391304             5              9

## Sentinel-indicator contribution
context model condition  indicator_total_mean_abs_shap  original_23_total_mean_abs_shap  indicator_fraction_of_all_shap
  novpn    RF        S1                       0.023679                         0.253552                        0.085413
  novpn    RF        S2                       0.019812                         0.257021                        0.071567
  novpn   XGB        S1                       0.020071                         5.831833                        0.003430
  novpn   XGB        S2                       0.000659                         5.873602                        0.000112
    vpn    RF        S1                       0.010832                         0.249021                        0.041686
    vpn    RF        S2                       0.009249                         0.250798                        0.035565
    vpn   XGB        S1                       0.012724                         5.378779                        0.002360
    vpn   XGB        S2                       0.001691                         5.350439                        0.000316

Total runtime: 442.76 seconds