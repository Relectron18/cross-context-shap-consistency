An in-project reading by a separate AI session. It is not an outside review.

Scope: the full text of /tmp/claude-0/rev12/v12_revised.txt, including tables, notes and references, checked against revision2/results/*.json/csv and the code in revision2/src and revision/src/common.py, using read-only Python. Figures were not available as images and were not checked.

## Defects, most severe first

1. **B/A. Section IV-C (after Eq. 4) and Section VI-E (permutation-importance ρ).**
   Quote: "Spearman correlation is computed on ranks with tied values given their average rank, so (3) is just the familiar no-tie formula."
   Problem: the code does not average tied ranks. `common.ranks()` uses `imp.rank(ascending=False, method="first")`, so tied values are ordered by their position in the FEATURES list. This has no effect on SHAP rankings, which contain no ties. It does matter for permutation importance. Before ranking, `p4_classmix_miniat.py` and `p5_concerns4.py` set every negative value to 0 and add 1e-12 (`clip(lower=0) + 1e-12`). That creates large ties: in marginal PI, Random Forest has 10 and 10 non-positive features (Non-VPN, VPN), and in conditional PI it has 9 and 12. Those ties are then broken by feature order, identically in both contexts, which artificially aligns the tied tail. The reported cross-context values are 0.611 (RF marginal), 0.719 (XGB marginal), 0.657 (RF conditional) and 0.575 (XGB conditional) (p7_audit.json `cross_rho`). Recomputed with average ranks on the clipped values, they are 0.523, 0.695, 0.618 and 0.557. On the raw, unclipped values they are 0.590, 0.669, 0.565 and 0.563. The qualitative claim that PI agrees less than SHAP still holds, but the printed numbers depend on an arbitrary tie-break and the stated method is wrong.
   Fix: (a) rank with `method="average"`, or rank the raw PI values, and recompute the four PI values in VI-E; or (b) describe exactly what was done. In either case, correct the sentence in IV-C.
   Verified.

2. **D. Section VI-E (closing paragraph), Section X-C and Section XII.**
   Quote: "For Random Forest, credit is redistributed within the block, but the block itself gains importance under VPN, so part of the shift is a change in what the models rely on." The Conclusion says: "Random Forest, in contrast, shows an increase in the importance of the minimum-IAT feature group itself."
   Problem: only the SHAP share supports this (17.0% to 23.3%). The two model-reliance measures reported in the same table both point the other way for Random Forest:
   - Joint permutation of the three minima costs 0.168 macro F1 on Non-VPN and 0.155 on VPN (Table IX; p4 `pi_RF_*` `grouped["min-IAT trio"]` = 0.1678 and 0.1551).
   - Dropping all three and retraining costs 0.028 on Non-VPN and 0.012 on VPN (`drop_RF_*_all3.df1`).

   By these measures the RF models depend less on the block under VPN. The text does not mention this conflict.
   Fix: describe the RF block gain as SHAP-specific, report that permutation and drop-group costs fall, and remove "change in what the models rely on" from the closing paragraph and the Conclusion.
   Verified.

3. **D/B. Section VI-E, "Drop-one and grouped retraining".**
   Quote: "in the VPN Random Forest, dropping min_fiat increases the shares of min_flowiat and min_biat by 2.2 and 1.5 percentage points".
   Problem: these are raw share differences (`share_others` − `base_share` in p4 `drop_RF_vpn_min_fiat`: 9.08−6.84 and 9.52−7.97). Part of each rise is arithmetic. Removing min_fiat's 8.15% share rescales every remaining share by about 1/(1−0.0815). The code also stores the gain against the base shares renormalised over the remaining 22 features (`top_gainers`): min_flowiat +1.63 and min_biat +0.84. The sentence is used as evidence that "dropping one increases the credit of the others", so it overstates that redistribution.
   Fix: report the renormalised gains (1.6 and 0.8 pp), or state that the figures include the renormalisation.
   Verified.

4. **G/B. Sections IV-E and VI-D (class-matched training).**
   Quote: "Class-matched training uses 190 training flows per class … Six draws of 1,330 flows per context …" Later: "The higher cross-context ρ of small matched training sets (approximately 0.85, compared to 0.77–0.82 for full-size models)".
   Problem: in `p4_classmix_miniat.py`, every matched and natural 1,330-flow model is XGBoost (`fit_imp("XGB", …)`). The paper never says so, which leaves readers to assume both models were tested. The comparator range "0.77–0.82 for full-size models" also matches no single XGBoost quantity. XGBoost's full-size primary cross-context ρ is 0.786, and its half-split cross ρ is 0.825 at about 3,476 flows, which is not full-size. The range appears to combine the RF half-split value (0.766) with the XGB one (0.825).
   Fix: state that the matched analysis is XGBoost-only, and compare against XGBoost values (0.786 primary, 0.825 half-split), naming which.
   Verified.

5. **C/G. Table II, "Matched subset" rows.**
   Quote: "Matched subset (per class) 190 | 48 | 14.3 … Matched subset (total) 1,330 | 336" with Test columns.
   Problem: no reported analysis uses a matched test set of 48 per class (336 records). In p4, matched models are trained on 190 per class and explained on the full test partition (`fit_imp` uses `D[c][1]`, the full test set). The 48/336 test figures appear to come from the earlier `data/controlled` subset, which revision2 does not use. The note says these rows give "the per-class sizes used for class-matched training", which covers only the Train columns.
   Fix: remove the Test entries from the matched rows, or state that no matched test set is used.
   Verified.

6. **B. Section IX, paragraph 3.**
   Quote: "Extra Trees agrees less well with Random Forest (ρ = 0.730–0.801 for the path-dependent and 0.709–0.776 for the interventional formulation)."
   Problem: in both `p3_sentinel_robust.py` and `p3d_finalize.py`, `rob_ET_vs_RF_{c}_interv` compares interventional Extra Trees against path-dependent Random Forest (`E[c]["RF_path"]`). The second range is therefore a mixed-formulation comparison, not ET versus RF under the interventional formulation.
   Fix: either compare ET-interventional with RF-interventional, or reword to "ET interventional versus RF path-dependent".
   Verified.

7. **D. Section VI-B, end of paragraph 1.**
   Quote: "Equal weighting of the true classes changes the rankings within contexts very little … and the cross-context values stay at about the same level."
   Problem: this holds for ρ (0.830 to 0.821; 0.786 to 0.793) but not for the top-weighted measures in Table VI:
   - RF RBO rises from 0.580 to 0.732 and its L1 falls from 0.333 to 0.270.
   - RF Top-10 overlap falls from 9 to 8; XGB Top-10 falls from 8 to 7 (κ10 0.65 to 0.47).

   An RBO change of 0.15 is as large as the RF random-half RBO gap discussed elsewhere.
   Fix: limit the claim to ρ and τb, and note the RBO and Top-10 changes.
   Verified (p1 `cross_RF_weighted`, `cross_XGB_weighted`).

8. **D/A. Section VI-C, paragraph 3.**
   Quote: "Preserving clusters of approximately ten similar flows in one half reduces within-context agreement … while cross-context agreement is barely changed".
   Problem: XGB cross ρ is 0.825 versus 0.832, which supports the claim. RF cross ρ falls from 0.766 to 0.730 (ref_summary `RF_random` / `RF_grouped10` `rho_cross`). That fall of 0.036 is about 40% of RF's 0.087 gap reduction.
   Fix: say this for XGBoost only, or give the RF change.
   Verified.

9. **D/H. Section VII, last paragraph.**
   Quote: "… with clusters of about 35 flows or the calibrated grouping, the XGBoost family gap also falls (0.014 and +0.016). Therefore, redundancy among related flows does not fully explain the observed differences."
   Problem: "Therefore" follows a sentence showing that the family gap falls to near zero under coarser grouping. That evidence points the opposite way, so the conclusion does not follow. The only support for persistence is the ~10-cluster setting (0.075 to 0.077).
   Fix: restrict the claim to fine grouping, or drop "Therefore …".
   Verified (ref_summary `XGB_grouped35.fam_gap` = 0.0143, `XGB_calibrated.fam_gap` = 0.0156).

10. **A. Section VI-B, paragraph 3.**
    Quote: "it equals that minimum in only 43% of Non-VPN and 31% of VPN training flows".
    Problem: `p1_audit_primary.py` computes `(X.min_flowiat == mm).mean()` over all training flows, including flows where both min_fiat and min_biat are -1 (`mm = inf`), where equality is impossible. Among flows where the minimum is defined, the shares are 48.5% (Non-VPN) and 33.5% (VPN), recomputed from the data. The adjacent claim that min_flowiat never exceeds that minimum is confirmed: 0 violations among 7,646 and 8,670 flows.
    Fix: use the defined-flow denominator, or say that undefined flows are included.
    Verified.

11. **D. Section IV-D ("Row-order blocks") against Section XI.**
    Quote: "In the released Scenario A2 ARFF files, flows are sorted by class and, within a class, appear in the order in which the flow meter wrote them."
    Problem: this is stated as fact. Later in the same paragraph the paper says "the dataset documentation does not describe the row order". Section XI calls it "our inference, from the similarity of neighbouring rows". The opening sentence contradicts both.
    Fix: hedge it, for example "appear to be in the order …".
    Verified (internal).

12. **C. Section IV-B against Section IV-G.**
    Quote (IV-B): "Explanations are always computed on the held-out test partition of the context in question."
    Problem: the feature-list transfer computes SHAP on an inner-validation split of the training partition (IV-G; `p2_consequence.py` explains `Xtr.iloc[ev]`).
    Fix: add "except for feature selection (Section IV-G)".
    Verified.

13. **A/D. Section VIII-A.**
    Quote: "These scores fall in the same range as the cluster-grouped cross-validation of Section III-C (0.47–0.64)".
    Problem: the transfer scores are 0.430, 0.438, 0.606 and 0.613 (p2 `transfer_*`). Two of the four are below 0.47, so the ranges only partly overlap.
    Fix: say "overlap with" or "are close to".
    Verified.

14. **A. Section VII, paragraph 3.**
    Quote: "the shares of the three leading families on Non-VPN data are within 2.1 percentage points of one another for Random Forest".
    Problem: under the alternative grouping, RF Non-VPN shares are 25.147 (directional), 24.210 (duration and totals) and 23.001 (flow IAT). The spread is 2.15 pp, which is not within 2.1.
    Fix: change to "2.2" or "about 2".
    Verified (recomputed from primary_rec_RF_novpn.npy with ALT_FAMILIES).

15. **C. Section IV-D (calibration) against the row-order paragraph and the code.**
    Quotes: "names 109 Non-VPN and 30 VPN capture files for CHAT, FT, MAIL, STREAMING and VOIP, plus one VPN P2P capture"; "P2P has at most one capture per context"; "P2P has a single capture per context".
    Problem: the first quote implies the list has no Non-VPN P2P capture, and the last says there is exactly one per context. `designs.py` sets `CAPTURE_FILES["novpn"][4] = 1`. The calibrated cluster count is not affected because of `max(4, ·)`, but the text is inconsistent.
    Fix: state whether the prefix list contains a Non-VPN P2P capture, and use one wording.
    Verified for the text and code; the content of the prefix list is unverified.

16. **B. Section IV-D, "Cluster-grouped half-splits".**
    Quote: "Whole clusters are assigned at random to the two halves, larger clusters first, each to the currently smaller half."
    Problem: `grouped_halves` sorts clusters by size, largest first, and assigns each to the currently smaller half. That is a greedy balanced partition. Randomness only breaks ties between clusters of equal size, so "at random" misdescribes the procedure. For row blocks the result is nearly random, because `array_split` gives near-equal sizes. For k-means clusters the assignment is essentially fixed once the clustering is fixed.
    Fix: describe it as greedy balancing in random order among clusters of equal size.
    Verified.

17. **G. Section IV-D (half sizes for row-block designs).**
    Problem: the paper gives common half sizes only for random halves (3,476–3,477) and cluster designs (3,368–3,476). Row-block halves are smaller and are not reported: about 140 per block gives 3,264 (ref_summary `*_ordered140` `n`), about 35 gives 3,440, about 10 gives 3,459, calibrated blocks give 3,467, and first-versus-second half gives 3,476. The ~140-block models therefore train on about 6% fewer flows than the random-half models they are compared with.
    Fix: add the sizes, for example to the Table VII note.
    Verified.

18. **G. Reproducibility details missing from Section IV and Table III.**
    Not stated:
    - k-means settings for the grouped designs (`n_init=3`, seed `seed+k`) and for the 50-cluster CV (`random_state=0`).
    - How ranks are tie-broken (item 1).
    - The XGBoost `tree_method` (library default).
    - That conditional-PI partner features are chosen by absolute training-set Spearman correlation, with quintile edges from training data.
    - That S2 replaces every -1 in all features with NaN, not only in the ten indicator sources.

    The Data and Code Availability section says the mapping to ARFF rows "can be regenerated from the public files with that code", but the code is available only on request.
    Fix: add these details, or release the code.
    Verified (code).

19. **E. Section VI-E.**
    Quote: "a conditional variant … which permutes each feature only within quintile cells of its two most correlated features [36]".
    Problem: Strobl et al. [36] define conditional importance on a grid built from the forest's own split points, for covariates selected by a correlation threshold. The paper's quintile grid on two partners is a simplification, though it reads as the method of [36]. The code comment does say "(Strobl et al.)".
    Fix: write "a simplified variant inspired by [36]".
    Verified for the code; the description of [36] is from memory.

20. **D. Section XII, paragraph 4.**
    Quote: "For SHAP-selected feature sets containing at least 10 features, transferring the selected set … results in an average macro-F1 loss of no more than approximately 0.013".
    Problem: only k = 10 and k = 15 were tested, so "at least 10" extends the result beyond the evidence. Section X-D states it correctly as "At k = 10 and k = 15".
    Fix: write "with 10 or 15 features".
    Verified.

21. **D. Section VI-D, paragraph 1.**
    Quote: "The swap does not bring a context closer to the other one."
    Problem: for XGBoost the swap does move each context slightly closer (+0.008 and +0.017; p4_classmix.csv), although by less than the spread across draws (SD 0.012–0.024). The next sentence gives these numbers, so the categorical wording is too strong rather than wrong.
    Fix: "does not bring a context materially closer".
    Verified.

22. **H. Section IV-G.**
    Quote: "We run k = 5, 10 and 15 for both models, give them equal weight, and repeat …"
    Problem: "give them equal weight" has no clear meaning. Nothing in the code weights the k values or combines them across k.
    Fix: delete it or explain it.
    Verified.

23. **A (minor). Section IV-D, "Calibrating the grouping".**
    Quote: "25–83 flows per capture without a VPN (median 44) and 95–324 with one (median 246)".
    Problem: the medians are 44.5 and 245.5 (from ARFF_COUNTS / CAPTURE_FILES in designs.py). One is rounded down and the other up.
    Fix: "median 44.5" and "245.5", or round both the same way.
    Verified.

24. **E. Reference [49].**
    "S. Mali … 'Analyzing encrypted traffic analysis using explainable AI,' Ann. Math. Comput. Sci., vol. 31, 2025". The title is grammatically odd ("Analyzing … analysis"), and I could not check the venue, volume or DOI offline. Separately, the citation "Webber et al. [45], Eq. 21" for the 86% top-10 weight matches my recollection of that paper's prefix-weight equation but was not checked.
    Unverified.

25. **G/unverified. Table III note and Section III-D.**
    Quote: "Every result was computed under Python 3.13.16 … XGBoost was run single-threaded; refitting with two threads gave identical predictions."
    Problem: no result file records the two-thread check; it appears only as a code comment in c2.py. The correlation clusters in Table XI are loaded from `revision/results/clusters_0.2.json`, which a first-revision script produced. No result file records which environment that script ran in.
    Fix: keep a record of the thread check, and regenerate the clusters in the revision2 environment or note where they came from.
    Unverified.

## Checked and consistent

- **Abstract and Conclusion.** Gaps of 0.13 and 0.19; calibrated-cluster gaps near zero (0.016 and 0.018); calibrated row-block gaps of 0.07 and 0.15; first-versus-second-half gaps of 0.03 and 0.09; transfer F1 of 0.43–0.61; 3 of the top 5 kept; Spearman 0.83 and 0.79; RBO 0.58 and 0.75; macro F1 0.9340 to 0.8836 and 0.9171 to 0.8533; cluster gaps of 0.08–0.10 at about 10 flows and 0.01–0.04 at about 35.
- **Section III.** Class counts in Table II; ARFF counts and duplicate removals (271 and 413; 8,694 and 9,380); unique feature rows; collisions 1 and 13; within-partition duplicates (2 vectors / 6 rows and 11 / 26, all label conflicts); TVD 0.142 (train/test about 0.001); χ² 447.4; Cramér's V 0.176; imbalance ratios 11.9 and 5.8; P2P 941; BROWSING 2,500 in every A2 file; entries below -1 (55 and 43 in training, 43 FT and 22 P2P, 17 and 12 in test, minimum -1,247, all in min_flowiat); sentinel rates; nearest-neighbour medians (std, log and rank); grouped-CV macro F1; LR 0.754 and 0.724; environment versions; Table III hyperparameters against `make_model`.
- **Section V and Table IV.** Every accuracy, macro F1 and per-class recall; the macro F1 losses of 0.0504 and 0.0637.
- **Section VI-A/B, Tables V and VI.** All 40 shares; top-5 and top-10 sums; between-model ρ of 0.814 and 0.917; every value in Table VI including the class-weighted rows; rank movements in Fig. 5's text; feature correlations; ρ of 0.976–0.998 between weighted and unweighted rankings.
- **Section VI-C and Table VII.** All 22 rows (F1, within, cross, Δ mean, SD, range, ΔRBO, ΔTop-5, class-weighted Δ) against ref_summary.json; the 39% and 47% reductions under ~10 clusters and the 43% and 18% reductions under calibrated row blocks; one-sided calibrated gaps (+0.228/−0.193; +0.067/−0.035); orderedhalf sign counts and the minimum of 0.0002; 4.1–4.4 versus 3.1 shared top-5 features; half-model F1 ranges; half sizes of 3,476–3,477 and 3,368–3,476.
- **Section IV-D.** Row-order locality ratios (0.37–0.76, 0.81–0.99 and 0.11–0.77) and |ρ| up to 0.68 from p7 `locality_dedup`; BROWSING and P2P at 34–40%; the RF 800-record check (ρ ≥ 0.997 and ≥ 0.998, same top 5); null and design construction against designs.py.
- **Section VI-D and Table VIII.** All means and SDs; the swap deltas (+0.008, +0.017, −0.035, −0.007); matched values (0.946, 0.945, 0.853, 0.857, 0.971); the 0.09 gap at 1,330 flows.
- **Section VI-E and Table IX.** SHAP ranks; block shares (17.0, 23.3, 26.3, 24.7; half-split means 16.3 and 24.5); all drop-one and drop-all ΔF1 values; marginal and conditional PI ranks; joint PI costs; drop-all-three cross ρ of 0.890 and 0.833; rank change from 10th to 5th; below -1 attribution shares and the refit ρ of 0.986–0.997.
- **Section VI-F.** All within-context comparisons.
- **Section VII, Tables X and XI.** All family shares and ranks; the per-member leaders; Table XI primary, within, cross and Δ values (primary values recomputed); the family-gap ranges under grouping (+0.016 to +0.116).
- **Section VIII, Tables XII and XIII.** Transfer F1 and recalls; every Table XII cell (shared counts, Δ mean/min/max at each k, seed-1 intervals, bottom-10, random-10, beat-random percentages); Table XIII against p5_topk_2x2; the 1–3 distinct lists.
- **Section IX and Table XIV.** Sentinel F1 values, indicator shares, S1-versus-S2 ρ, cross-context ρ, Kendall τb and Top-5 for S1 and S2; robustness ρ, intervals and RBO; path-versus-interventional ρ; seed ρ of 0.9928; depth-3 and depth-10 values; bootstrap centring within 0.015.
- **Table XV.** Every entry.
- **Equations.** Eq. 2, κk and E[O_k] (κ5 = 0.49, κ10 = 0.65 and 0.82), RBO extrapolated form, L1 = 2×TVD, and Eq. 6 all match the code.
- **Cross-references.** Every Section, Table, Fig. and equation reference resolves correctly.
- **References.** Numbering is sequential by first citation. Every reference is cited. Bibliographic details of [1]–[48] and [50]–[59] look internally consistent, checked from memory.
