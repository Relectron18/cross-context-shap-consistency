---
title: "Audit ledger — v12 defect review -> v13"
date: 2026-10-20
input: paper/SHAP_VPN_revised_v12.docx (MD5 eca57b13dd070e30f18efa67ce68c6df)
output: paper/SHAP_VPN_revised_v13.docx
checklist: paper/reviews/2026-10-20-v12-defect-review.md (items D1-D30, numbered as there)
---
Written by the drafting session. Every "confirmed" below was re-derived from code or result files by this session;
an independent re-check of v13 by a separate AI session is recorded at the end. Neither is an outside review.

Status key: **C** verified and corrected; **N** checked, not a defect, no change; **P** partly verifiable (what could
not be checked is named); **X** needs new data or experiments, stated as a limitation.

All numbers used by the corrections are read from `revision2/results/p8_v13.json` (script `src/p8_v13_audit.py`), and
`src/check_v13_numbers.py` confirms them in the manuscript (exit 0 on v13; exit 1 on v12 and on two planted digit errors).

| ID | Location | v12 claim or implementation | Evidence | Confirmed | Correction in v13 | Files | Validation | Status |
|---|---|---|---|---|---|---|---|---|
| D1 | IV-C, VI-E | "ranks with tied values given their average rank"; code `ranks()` used `method="first"`. PI cross-context ρ 0.611 / 0.719 / 0.657 / 0.575 | `common.py:78`; RF has 10 and 10 (marginal), 9 and 12 (conditional) features tied at 0 after clipping. Average ranks: 0.523 / 0.695 / 0.618 / 0.557; unclipped 0.590 / 0.669 / 0.565 / 0.563 (`p8_v13.json` pi_cross). SHAP vectors have no ties (0 of 1,936 reference vectors, 0 primary, 0 robustness); reference summaries change by at most 1.1e-16 | yes | `ranks()` now `method="average"`; VI-E reports corrected values, tie counts and unclipped values; IV-C states tie handling and that ties arise only for PI. Table IX PI ranks of the minimum IATs are identical under average ranks (checked) | common.py, p8_v13_audit.py, fix_v13.py | conditional PI re-run at full precision reproduces stored values exactly (`p8b_cpi_full.json`, max diff 0.0); check script | C |
| D2 | VI-E, X-C, XII | RF min-IAT block "gains importance under VPN, so part of the shift is a change in what the models rely on" | SHAP block share 17.0 -> 23.3%; joint permutation cost 0.168 -> 0.155; drop-all-three cost 0.028 -> 0.012 (`p4_results.json`) | yes | All three places now say the SHAP share rises while the performance-based measures do not show greater reliance, and that the measures disagree | fix_v13.py | check script ("RF reliance") | C |
| D3 | VI-E | dropping min_fiat raises min_flowiat / min_biat shares by 2.2 / 1.5 pp | raw differences include renormalisation after removing min_fiat's 8.2%; renormalised 1.6 / 0.8 (`p8_v13.json`, equals stored `top_gainers`) | yes | reports 1.6 and 0.8 pp beyond renormalisation, with 2.2 and 1.5 as the raw values | fix_v13.py | check script | C |
| D4 | IV-E, VI-D, XI | class-matched training not stated as XGBoost-only; comparator "0.77–0.82 for full-size models" | `p4_classmix_miniat.py` calls `fit_imp("XGB", ...)`; XGB primary cross ρ 0.786, XGB random-half cross 0.825; 0.77 matches no XGB value | yes | "run for XGBoost only" in IV-E, VI-D, XI and Table II note; comparators 0.786 and 0.825. No RF result invented | fix_v13.py | check script | C |
| D5 | Table II | matched-subset Test columns 48 / 336 | no analysis uses a matched test set; matched models are explained on the full test partition | yes | Test cells set to "—"; note says there is no matched test set. Training counts kept (needed to reproduce) | fix_v13.py | rendered table inspected | C |
| D6 | IX | ET vs RF "0.709–0.776 for the interventional formulation" | `p3d_finalize.py:18` compares ET interventional with RF path-dependent. RF interventional TreeSHAP computed on the same 500 records and 200 background records (`p8a_rf_interv.py`); its cross-context ρ reproduces the stored 0.802 exactly | yes | like-for-like: 0.730–0.801 (both path-dependent), 0.731–0.743 (both interventional) | p8a_rf_interv.py, results/rf_interv_rec_*.npz, fix_v13.py | check script; RF path arrays identical to stored ones | C |
| D7 | VI-B | class weighting: "cross-context values stay at about the same level" | ρ 0.830 -> 0.821, 0.786 -> 0.792; RF RBO 0.580 -> 0.732, L1 0.333 -> 0.270; Top-10 9 -> 8, 8 -> 7 | yes | ρ stated as stable, top-weighted changes reported | fix_v13.py | check script | C |
| D8 | VI-C | ~10-flow clusters: cross agreement "barely changed" | RF 0.766 -> 0.730 (41% of RF gap reduction); XGB 0.825 -> 0.832 | yes | reports both models' cross values | fix_v13.py | check script | C |
| D9 | VII | "Therefore, redundancy ... does not fully explain" after XGB family gap falls to 0.014 / 0.016 | family gap 0.075 random, 0.077 at ~10, 0.014 at ~35, 0.016 calibrated | yes | scoped to the finest grouping; "at coarser grouping it may" | fix_v13.py | diff reviewed | C |
| D10 | VI-B | min_flowiat equals the smaller minimum in 43% / 31% | denominator included flows where both minima are -1; among flows with at least one defined minimum 48% / 34% (48.5% / 33.5%); no violations of min_flowiat ≤ minimum | yes | both populations named and both percentages given | fix_v13.py | check script | C |
| D11 | IV-D | rows "appear in the order in which the flow meter wrote them" stated as fact | same paragraph and XI call it an inference | yes | "appear to be in the order" | fix_v13.py | diff | C |
| D12 | IV-B vs IV-G | explanations "always computed on the held-out test partition" | feature selection uses SHAP on an inner-validation split | yes | "Except in the feature-selection step of Section IV-G" | fix_v13.py | diff | C |
| D13 | VIII-A | transfer F1 "in the same range" as grouped CV 0.47–0.64 | Non-VPN-to-VPN 0.430 (XGB), 0.438 (RF) lie below | yes | "close to", with the two values named as below the range | fix_v13.py | check script | C |
| D14 | VII | "within 2.1 percentage points" | spread 2.15 pp | yes | 2.15 | fix_v13.py | check script | C |
| D15 | IV-D, XI | P2P capture count described three ways; BROWSING "cannot be split by capture at all" | prefix list (munhouiani/Deep-Packet `utils.py`, fetched 2026-10-20): 109 Non-VPN and 30 VPN files for five classes, one VPN P2P (`vpn_bittorrent`), no Non-VPN P2P, no BROWSING class; code counts Non-VPN P2P as 1, no effect with minimum of four groups | yes | one consistent description ("at most one listed capture per context"; "no listed captures" for BROWSING; "with the information available") and the Non-VPN P2P assumption stated | fix_v13.py, designs_meta.py | prefix list read from source | C |
| D16 | IV-D | clusters "assigned at random" | `designs.py` 69–74: groups permuted, then sorted by size, each to the half with fewer flows | yes | greedy size-balanced assignment described | fix_v13.py | code read | C |
| D17 | IV-D | row-block half sizes not reported | stored models: 3,264 (~140), 3,440 (~35), 3,459 (~10), 3,467 (calibrated), 3,476 (halves) vs 3,476–3,477 random | yes | range added | p8_v13_audit.py (sizes read from stored models, both contexts), fix_v13.py | check against pickles | C |
| D18 | reproducibility | k-means settings, tie handling, tree_method, PI partners, S2 scope not stated; code on request but row mapping "regenerated with that code" | `designs.py`: KMeans n_init=3, random_state per repeat, context, class; XGB tree_method default `auto`, identical predictions to `hist`; partners = two features with highest absolute Spearman correlation on the training partition, quintile cells; S2 replaces -1 in every feature | yes | all added (IV-D, IV-C, Table III, VI-E, IV-H); availability statement separates benchmark, code, processed artifacts and generated tables/figures, says the code is private and not released | fix_v13.py | check script (tree_method) | C |
| D19 | VI-E | conditional PI presented as the method of [36] | code is a quintile-cell simplification | yes | "a simplified conditional variant inspired by [36]" | fix_v13.py | diff | C |
| D20 | XII | "at least 10 features" | only k = 10 and 15 tested | yes | "of 10 or 15 features" | fix_v13.py | check script | C |
| D21 | VI-D | "The swap does not bring a context closer" | XGB +0.008, +0.017, within SD across draws 0.009–0.024 | yes | "materially closer"; XGB moves named and set against draw SD | fix_v13.py | diff | C |
| D22 | IV-G | "give them equal weight"; garbled clauses | code applies no weighting across k | yes | paragraph rewritten (same content, no weighting claim) | fix_v13.py | compared with `p2_consequence.py` lines 24–40 (stratified 80/20 inner split, selector on the 80%, SHAP on the 20% or 800 records for RF, reduced model refitted on the full training partition) | C |
| D23 | IV-D | medians 44.5 / 245.5 printed 44 / 246 | `p8_v13.json` flows_per_capture | yes | 44.5 and 245.5 | fix_v13.py | check script | C |
| D24 | refs | [49] details; "Webber et al. [45], Eq. 21" vs code comment "eq. 32" | [49]: title, journal, vol. 31, pp. 1–22, 2025 and article id 660 confirmed by search result of annalsmcs.org; authors and use of ISCX VPN-nonVPN not confirmed (site blocks automated reading). Eq. 21 of Webber et al. is the top-d weight (gives 0.856 for p = 0.9, d = 10, matching the paper's 86%); Eq. 32 is extrapolated RBO, which the code implements. Equation numbers confirmed via two implementations that cite the paper (changyaochen/rbo, gespeR), not from the paper itself | no conflict | none | — | computed Eq. 21 value | N (Eq.), P ([49]) |
| D25 | G | two-thread XGB check not recorded; Table XI clusters from earlier run | re-run: `c2.make_model` defaults XGBoost to one thread (n_jobs=1), so every reported XGBoost fit was single-threaded; one thread, two threads and all cores (n_jobs=-1, two cores here) give identical predictions; clusters at 0.2 regenerated in this environment are identical to `clusters_0.2.json` | records missing; text correct | none to the text (v12 note "run single-threaded; ... two threads gave identical predictions" is accurate and kept); checks recorded in `p8_v13.json` xgb_threads and clusters_0.2 | p8_v13_audit.py | p8_v13.json | C (records) |
| D26 | VI-F | row blocks used as evidence that VPN rankings are steadier | weak locality in VPN FT, MAIL, P2P, VOIP (lag-1 0.81–0.99) | yes | caveat added; "neither proxy settles it" | fix_v13.py | diff | C |
| D27 | VI-C, X-A, XII | blocks "do not exploit the features"; clusters overcorrect / blocks undercorrect | row position correlates with features (|ρ| up to 0.68); first-vs-second-half F1 0.73–0.80 | yes | both caveats added in VI-C, X-A and XII | fix_v13.py | diff | C |
| D28 | IV-D | "none of the designs controls for this" (explained-record asymmetry) | the permutation null explains each pseudo-context on its own test partition | yes | null named as covering record asymmetry plus class mix (+0.028 RF, -0.001 XGB); grouped designs do not | fix_v13.py | design code read | C |
| D29 | VI-C | "only the outer partition and the seeds change" | half subsamples also redrawn | yes | reworded | fix_v13.py | code read | C |
| D30 | IV-D | 0.37–0.76 locality range presented as fixed | seeds 0–3: 0.36–0.73, 0.36–0.74, 0.36–0.75, 0.36–0.70 | yes | "about 0.36–0.76 (depends on which random rows are drawn)" | fix_v13.py | p8_v13.json | C |
| D31 | language | II-C, IV-G, X-E garbled; "3.04 and 3.39" without unit; mixed apostrophes; macro-F1 / macro F1; data set / dataset; Fig. 4 cited before Fig. 3; Table VII cited before Table IV; asterisk without note | text | yes | all corrected; tables and figures now first cited in order | fix_v13.py | structural check | C |
| D32 | code/README | prefix list credited to Lotfollahi et al. | source is munhouiani's implementation [56] | yes | `designs.py` comment, `designs_meta.py`, README | designs.py, designs_meta.py, README | — | C |

Numbering note: the defect review has 30 items; D31 and D32 above split its items 29-30 (language) and 28 (code comment).
Mapping: review 1->D1, 2->D2, 3->D6, 4->D4, 5->D5, 6->D3, 7->D10, 8->D7, 9->D8, 10->D9, 11->D26, 12->D27, 13->D28,
14->D13, 15->D21, 16->D20, 17->D11, 18->D15, 19->D12, 20->D29, 21->D16, 22->D30, 23->D14/D23, 24->D17/D18, 25->D19,
26->D25, 27->D24, 28->D32, 29-30->D22/D31.

## Further defects found in the full-paper pass (not in the review)

| ID | Location | Defect | Evidence | Correction | Status |
|---|---|---|---|---|---|
| F1 | Table VII, VI-C | double rounding: 4-decimal summaries rounded again to 3 decimals in 9 cells (e.g. XGB null Δ -0.002 -> -0.001, range -0.018 -> -0.019; RF calibrated within-VPN 0.296 -> 0.297; RF ordered10 SD 0.029 -> 0.030; RF row-block calibrated cross 0.743 -> 0.742; RF first-vs-second-half ΔRBO 0.046 -> 0.047) | per-repeat CSV means | all 176 numeric cells of Table VII now equal the CSV means rounded once (check script) | C |
| F2 | Table XI | same double rounding (RF mean-per-member cross 0.754 -> 0.755; XGB mean-per-member Δ ~10 +0.054 -> +0.055) | CSV means 0.75455, 0.05455 | corrected | C |
| F3 | Table III note | (retracted) a first v13 build changed "XGBoost was run single-threaded" to "all available cores in the other fits", reading `common.py` instead of the `c2.make_model` wrapper that the revision2 scripts use. The independent check caught it; the v12 wording is correct and was restored | `c2.py` lines 47-51 | restored | retracted |
| F4 | formatting | X-E paragraph in Heading 1 style (appears in the navigation pane); one paragraph in BodyText style; ten body paragraphs without indent/justification; four empty paragraphs before headings; 24 runs tagged en-IN | document.xml | standard body formatting; empty paragraphs removed; language en-US | C |

Figures: all eight rendered and inspected (LibreOffice PDF). Fig. 1 counts match Table II; Fig. 3 matches Table IV; Fig. 4
bars match Table V; Fig. 5 rank moves match VI-B; Fig. 6 printed correlations match VI-B; Fig. 7 Δ labels match the
corrected Table VII (it already showed -0.001 for the XGBoost null); Fig. 8 matches VIII. No figure depends on the
permutation-importance tie handling, so none was regenerated.

Not resolved (needs data or experiments; stated in XI): a capture-level split, a grouped permutation null, and a
comparison of both contexts on a common set of explained records.

## Independent re-check of v13 (separate AI session, 2026-10-20)

Report: 25 of 30 items fixed correctly; five leftovers or new problems, all checked by the drafting session and fixed:
1. Table III thread note made false by the first v13 build (F3 above) -> v12 wording restored; audit now tests n_jobs=-1 explicitly.
2. VI-D: SD range 0.009–0.024 mixed Random Forest and XGBoost draws -> XGBoost natural-mix SDs 0.012 and 0.024, which both moves are below (asserted in fix_v13.py).
3. IV-D still said BROWSING and P2P "cannot be divided by capture at all" -> same wording as XI ("with the information available").
4. VI-F "indicate the same direction" lost its referent -> reworded.
5. Row-block half-models called "accurate" while 0.73–0.80 is read as a warning sign -> "far more accurate than under coarse clusters" (VI-C, X-A).
Also from that report: the k-means settings of the 50-cluster grouped CV in III-C were not stated -> "(three initialisations, fixed seed)" (`p1` line 52: n_init=3, random_state=0).
The verifier confirmed that every SHAP result is unchanged by the tie fix (including XGBoost depth 3/10, S1/S2 and 1,330-flow fits, which it refitted), and that the permutation-importance cross-context ρ is the only reported quantity that depends on tie handling.
Full report kept at paper/reviews/2026-10-20-v13-verification.md.
