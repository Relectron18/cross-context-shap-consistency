---
title: "Standalone defect review — SHAP_VPN_revised_v12"
date: 2026-10-20
manuscript: paper/SHAP_VPN_revised_v12.docx (MD5 eca57b13dd070e30f18efa67ce68c6df)
sources: separate-session reading (2026-10-20-v12-defect-review-reviewer.md) + drafting-session checks
---
An in-project reading by a separate AI session, with findings re-checked by the drafting session. It is not an outside review.
No manuscript changes made. "verified" = re-computed or read from the named result file/code line.

## A. Results/method errors (fix before submission)
1. VI-E / IV-C: permutation-importance cross-context rho (0.611/0.719 marginal, 0.657/0.575 conditional) uses ranks(method="first") on clip(0)+1e-12 values; 10/10 RF features tied at zero are ordered by list position. Average ranks give 0.523/0.695/0.618/0.557; raw values 0.590/0.669/0.565/0.563. IV-C wrongly says ties get average rank. Conclusion (PI agrees less than SHAP) survives. [verified: common.py:78, p4:125, p5:51]
2. VI-E, X-C, XII: RF min-IAT block "gains importance under VPN ... change in what the models rely on" rests on SHAP share only; grouped permutation (0.168 -> 0.155) and drop-all-three (0.028 -> 0.012) point the other way. [verified: p4 pi_RF_*.grouped, drop_RF_*_all3]
3. IX: "0.709-0.776 for the interventional formulation" compares interventional ET with path-dependent RF. [verified: p3d_finalize.py:18]
4. IV-E / VI-D: class-matched training is XGBoost-only but not stated; comparator "0.77-0.82 for full-size models" is not an XGBoost quantity. [verified: p4 line 64]
5. Table II matched-subset Test columns (48/class, 336) describe a test set no analysis uses. [reviewer, verified]
6. VI-E: "2.2 and 1.5 percentage points" include mechanical renormalisation; renormalised gains 1.6 and 0.8. [reviewer, verified]
7. VI-B: 43% / 31% (min_flowiat equals the minimum) include flows where both inputs are -1; among defined flows 48.5% / 33.5%. [reviewer, verified]

## B. Overstated or unsupported statements
8. VI-B: equal class weighting "cross-context values stay at about the same level" - true for rho only; RF RBO 0.580 -> 0.732, L1 0.333 -> 0.270. [verified: p1 cross_RF_weighted]
9. VI-C: ~10-flow clusters leave cross agreement "barely changed" - RF cross rho 0.766 -> 0.730, ~40% of RF gap reduction. [reviewer]
10. VII: "Therefore, redundancy ... does not fully explain" follows evidence that XGB family gap falls to 0.014/0.016; needs "at fine grouping". [verified]
11. VI-F: row blocks used as evidence that within-VPN rankings are "again steadier", although adjacent VPN FT/MAIL/P2P/VOIP rows are nearly as different as random rows (lag-1 0.81-0.99). [verified: p7_audit]
12. VI-C / X-A: "Clusters may overcorrect, blocks may undercorrect"; blocks "do not exploit the features" - row position correlates with features (|rho| up to 0.68) and orderedhalf half-models are less accurate. [verified: p7_audit]
13. IV-D (limitations): "none of the designs controls for this" - the permutation null explains pseudo-contexts on different records. [verified: designs.py null]
14. VIII-A: transfer F1 0.43-0.61 "in the same range" as grouped CV 0.47-0.64; two values below 0.47. [reviewer]
15. VI-D: "The swap does not bring a context closer" - XGB moves slightly closer (+0.008, +0.017), within draw spread. [reviewer]
16. XII: "at least 10 features" generalises beyond k = 10, 15 tested (X-D is correct). [reviewer]
17. IV-D: rows "appear in the order in which the flow meter wrote them" stated as fact; same paragraph and XI call it an inference. [reviewer]

## C. Internal inconsistencies / factual description
18. IV-D/XI: P2P captures described three ways ("plus one VPN P2P capture", "at most one per context", "a single capture per context"); BROWSING "cannot be split by capture at all" is stronger than the data support; 34-40% share is computed on all flows. [verified text/code]
19. IV-B vs IV-G: explanations "always computed on the held-out test partition", but feature selection uses SHAP on an inner-validation split. [reviewer]
20. VI-C: "only the outer partition and the seeds change" - half pairing and subsampling change too. [verified]
21. IV-D: clusters "assigned at random" - code uses greedy size-balanced assignment. [reviewer]
22. IV-D: 0.37-0.76 locality range is seed-dependent (seeds 1-3: 0.36-0.74/0.75/0.70). [verified: p7_audit]
23. VII: "within 2.1 percentage points" - spread is 2.15. IV-D medians 44.5/245.5 printed 44/246. [reviewer]

## D. Reproducibility / citation
24. Not stated: k-means n_init and seeds, rank tie handling, XGBoost tree_method, conditional-PI partner selection, row-block half sizes (3,264-3,467 vs 3,476 random), S2 replaces every -1. Code "on request" yet ARFF mapping "regenerated ... with that code". [reviewer]
25. VI-E: quintile-cell conditional permutation presented as the method of [36]; it is a simplification. [reviewer]
26. No result file records the two-thread XGBoost check; Table XI clusters (clusters_0.2.json) come from the earlier run. [reviewer, unverified]
27. [49] details and "Webber et al. [45], Eq. 21" not checked offline (common.py comment says eq. 32). [unverified]
28. Code, not manuscript: designs.py lines 40-42 and README credit the prefix list to Lotfollahi et al.; it comes from the munhouiani implementation [56].

## E. Language / format (meaning or submission polish)
29. IV-G: "the importance is calculated by using the TreeSHAP measure of the latter"; "Every training split is separated into ... split" (split/partition clash); "give them equal weight" (no weighting applied across k). II-C: "Explanation capability is being required of security classifiers". X-E: "If there is an application of the explanations for decision-making tasks". VI-B: "mean absolute rank change is 3.04 and 3.39" (positions).
30. Mixed Kendall's/Kendall’s and Spearman's/Spearman’s; one "macro-F1" vs 33 "macro F1"; "data set" (3) vs "dataset" (22); Table VII first cited (III-C) before Table IV; Fig. 4 cited before Fig. 3; author asterisk with no "Corresponding author" legend.

## Checked and consistent
Abstract/conclusion numbers; Tables II-XV values (except items above); dedup counts; equations (2), (5), (6); RBO, L1; all section/table/figure/equation cross-references; references [1]-[59] numbered in first-appearance order, all cited; abstract 250 words; no non-Latin characters outside equations.
