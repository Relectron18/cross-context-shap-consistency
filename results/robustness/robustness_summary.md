# Task H - SHAP Robustness

## Cross-context robustness
     model              method  spearman  kendall  rank_displacement  top5_overlap  top10_overlap  spearman_ci_lower  spearman_ci_upper  family_spearman  same_top2_families
        RF tree_path_dependent  0.819170 0.628458           3.217391             3              9           0.803360           0.840934         0.942857                True
       XGB tree_path_dependent  0.779644 0.581028           3.478261             3              7           0.776680           0.810277         0.828571                True
        RF      interventional  0.802372 0.612648           2.869565             3              8           0.770751           0.814229         0.942857                True
       XGB      interventional  0.778656 0.573123           3.652174             3              8           0.769763           0.794466         0.828571                True
ExtraTrees tree_path_dependent  0.680830 0.573123           3.739130             3              6           0.633399           0.762846         0.828571                True
ExtraTrees      interventional  0.738142 0.573123           3.565217             3              7           0.661067           0.770751         0.714286                True

## Alternative configuration agreement
context reference_model    reference_method alternative_model  alternative_method  spearman  kendall  rank_displacement  top5_overlap  top10_overlap
  novpn              RF tree_path_dependent                RF      interventional  0.975296 0.881423           1.130435             4              9
    vpn              RF tree_path_dependent                RF      interventional  0.969368 0.881423           1.130435             4             10
  novpn             XGB tree_path_dependent               XGB      interventional  0.981225 0.920949           0.869565             4              9
    vpn             XGB tree_path_dependent               XGB      interventional  0.988142 0.928854           0.695652             4             10
  novpn              RF tree_path_dependent        ExtraTrees tree_path_dependent  0.730237 0.588933           3.826087             3              7
    vpn              RF tree_path_dependent        ExtraTrees tree_path_dependent  0.801383 0.596838           3.478261             2              8
  novpn              RF tree_path_dependent        ExtraTrees      interventional  0.709486 0.557312           3.913043             3              7
    vpn              RF tree_path_dependent        ExtraTrees      interventional  0.775692 0.596838           3.391304             2              8

## Alternative model performance
context      model  accuracy  macro_f1  train_seconds
  novpn         RF  0.945371  0.917074       0.614601
  novpn        XGB  0.958022  0.934733       1.875358
  novpn ExtraTrees  0.931570  0.899493       1.036045
    vpn         RF  0.881130  0.853350       1.411334
    vpn        XGB  0.902985  0.877925       3.641667
    vpn ExtraTrees  0.874733  0.849075       1.592661

Total runtime: 6907.30 seconds