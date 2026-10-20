"""v13 audit and recomputation for the standalone defect review of v12 (paper/reviews/2026-10-20-v12-defect-review.md).

Reads existing result files and the deduplicated data only, except that the Extra Trees section needs the
interventional Random Forest arrays written by p8a_rf_interv.py. Writes results/p8_v13.json. Nothing in the
earlier result files is overwritten: p4_results.json and p5_results.json keep the v12 values (ties broken by
feature order) for audit, and the corrected values live here.
"""
from c2 import *
import os
from scipy.stats import kendalltau
from summarize_refs import summarise, ALT_FAMILIES
from designs_meta import CAPTURE_FILES, ARFF_COUNTS
OUT = {}
P1 = json.load(open(OUT2 / "p1_results.json")); P2 = json.load(open(OUT2 / "p2_results.json"))
P4 = json.load(open(OUT2 / "p4_results.json")); P5 = json.load(open(OUT2 / "p5_results.json"))
S = json.load(open(OUT2 / "ref_summary.json"))
D = {c: load(c) for c in CTX}

# ---------- A. ties in stored SHAP importance vectors (if none, average and first ranks coincide)
def n_ties(v): return int(len(v) - len(np.unique(np.round(np.asarray(v, float), 15))))
ties = {}
for m in ("RF", "XGB"):
    for c in CTX:
        ties[f"primary_{m}_{c}"] = n_ties(np.load(OUT2 / f"primary_rec_{m}_{c}.npy").mean(0))
nvec, nt = 0, 0
for m in ("RF", "XGB"):
    for d in ("random", "grouped10", "grouped35", "grouped", "calibrated", "ordered10", "ordered35", "ordered140", "orderedcal", "orderedhalf", "null"):
        st = pickle.load(open(OUT2 / f"ref_{m}_{d}.pkl", "rb"))
        for k, v in st.items():
            for vec in (v["rec"].mean(0), np.mean([v["rec"][v["y"] == j].mean(0) for j in range(7)], axis=0)):
                nvec += 1; nt += n_ties(vec) > 0
ties["reference_vectors_checked"] = nvec; ties["reference_vectors_with_ties"] = nt
for c in CTX:
    E = np.load(OUT2 / f"et_rec_{c}.npz")
    for k in E.files: ties[f"robust_{k}_{c}"] = n_ties(E[k].mean(0))
OUT["ties_shap"] = ties
# the reference summaries are recomputed with the new ranks() and compared with the stored CSVs
diff = 0.0
for m in ("RF", "XGB"):
    for d in ("random", "grouped10", "calibrated", "orderedhalf", "null"):
        new = summarise(m, d); old = pd.read_csv(OUT2 / f"ref_summary_{m}_{d}.csv")
        diff = max(diff, float((new[old.columns] - old).abs().max().max()))
OUT["ref_summary_max_abs_change_with_average_ranks"] = diff

# ---------- B. permutation importance: cross-context agreement with average ranks
def pi_vec(src, key): return pd.Series(src[key]["values"])[FEATURES]
pi = {}
for lab, src, pre in (("marginal", P4, "pi"), ("conditional", P5, "cpi")):
    for m in ("RF", "XGB"):
        a, b = pi_vec(src, f"{pre}_{m}_novpn"), pi_vec(src, f"{pre}_{m}_vpn")
        ca, cb = a.clip(lower=0) + 1e-12, b.clip(lower=0) + 1e-12
        first = lambda s: s.rank(ascending=False, method="first")
        pi[f"{lab}_{m}"] = dict(
            rho_average=float(compare(ca, cb)["rho"]), tau_b_average=float(compare(ca, cb)["tau"]),
            rho_first_v12=float(spearmanr(first(ca), first(cb))[0]),
            rho_raw_unclipped=float(spearmanr(a, b)[0]),
            n_nonpositive={"novpn": int((a <= 0).sum()), "vpn": int((b <= 0).sum())},
            rank_min={c: {f: float(ranks(v.clip(lower=0) + 1e-12)[f]) for f in ("min_fiat", "min_biat", "min_flowiat")} for c, v in (("novpn", a), ("vpn", b))})
OUT["pi_cross"] = pi
OUT["shap_cross_rho"] = {m: P1[f"cross_{m}"]["rho"] for m in ("RF", "XGB")}

# ---------- C. min_flowiat equals min(min_fiat, min_biat): denominators
mf = {}
for c in CTX:
    X = D[c][0]
    m = np.minimum(X.min_fiat.where(X.min_fiat >= 0, np.inf), X.min_biat.where(X.min_biat >= 0, np.inf))
    defined = np.isfinite(m); eq = (X.min_flowiat == m)
    mf[c] = dict(n_train=len(X), n_defined=int(defined.sum()), share_all=float(eq.mean()), share_defined=float(eq[defined].mean()),
                 exceed=int((X.min_flowiat[defined] > m[defined]).sum()))
OUT["minflowiat_equality"] = mf

# ---------- D. drop-one share changes: raw and against renormalised base shares
d = P4["drop_RF_vpn_min_fiat"]; bs = d["base_share"]
OUT["drop_RF_vpn_min_fiat"] = {f: dict(raw=d["share_others"][f] - bs[f],
                                      renormalised=d["share_others"][f] - bs[f] / (1 - bs["min_fiat"] / 100),
                                      stored_top_gainer=d["top_gainers"].get(f)) for f in ("min_flowiat", "min_biat")}
OUT["drop_RF_vpn_min_fiat_base_share"] = bs["min_fiat"]
OUT["min_iat_reliance_RF"] = {c: dict(shap_block_share=None, joint_perm_df1=P4[f"pi_RF_{c}"]["grouped"]["min-IAT trio"],
                                      drop_all3_df1=-P4[f"drop_RF_{c}_all3"]["df1"]) for c in CTX}
for m in ("RF", "XGB"):
    for c in CTX:
        rec = np.load(OUT2 / f"primary_rec_{m}_{c}.npy").mean(0); s = pd.Series(rec / rec.sum() * 100, index=FEATURES)
        OUT.setdefault("min_iat_block_share", {})[f"{m}_{c}"] = float(s[["min_fiat", "min_biat", "min_flowiat"]].sum())
for c in CTX: OUT["min_iat_reliance_RF"][c]["shap_block_share"] = OUT["min_iat_block_share"][f"RF_{c}"]

# ---------- E. class-matched training: model and comparators
OUT["matched1330"] = dict(model="XGB (p4_classmix_miniat.py calls fit_imp('XGB', ...))", **P4["matched1330"]["mean"],
                         xgb_primary_cross=P1["cross_XGB"]["rho"], xgb_random_half_cross=S["XGB_random"]["mean"]["rho_cross"],
                         rf_primary_cross=P1["cross_RF"]["rho"], rf_random_half_cross=S["RF_random"]["mean"]["rho_cross"])

# ---------- F. half sizes per design
hs = {}
for m in ("RF", "XGB"):
    for dname in ("random", "grouped10", "grouped35", "grouped", "calibrated", "ordered10", "ordered35", "ordered140", "orderedcal", "orderedhalf", "null"):
        ns = [v["n"] for v in pickle.load(open(OUT2 / f"ref_{m}_{dname}.pkl", "rb")).values()]   # both contexts, both halves
        hs[f"{m}_{dname}"] = [int(min(ns)), int(max(ns))]
OUT["half_sizes"] = hs

# ---------- G. ~10-flow clusters: change in cross-context agreement
for m in ("RF", "XGB"):
    a, b = S[f"{m}_random"]["mean"], S[f"{m}_grouped10"]["mean"]
    OUT.setdefault("grouped10_cross", {})[m] = dict(random=a["rho_cross"], grouped10=b["rho_cross"], cross_change=b["rho_cross"] - a["rho_cross"],
                                                   gap_reduction=a["rho_gap"] - b["rho_gap"],
                                                   share_of_reduction_from_cross=(a["rho_cross"] - b["rho_cross"]) / (a["rho_gap"] - b["rho_gap"]))

# ---------- H. alternative family shares (primary RF, Non-VPN)
for m in ("RF", "XGB"):
    for c in CTX:
        imp = pd.Series(np.load(OUT2 / f"primary_rec_{m}_{c}.npy").mean(0), index=FEATURES)
        fam = family_imp(imp, ALT_FAMILIES); OUT.setdefault("alt_family_shares", {})[f"{m}_{c}"] = (fam / fam.sum() * 100).round(3).to_dict()

# ---------- I. flows per capture (ARFF counts / capture files of the prefix list)
for c in CTX:
    per = [ARFF_COUNTS[c][k] / CAPTURE_FILES[c][k] for k in (1, 2, 3, 5, 6)]
    OUT.setdefault("flows_per_capture", {})[c] = dict(min=min(per), max=max(per), median=float(np.median(per)))
OUT["prefix_list_counts"] = dict(novpn_chat_ft_mail_streaming_voip=109, vpn_same=30, vpn_p2p=1, novpn_p2p=0,
                                 note="munhouiani/Deep-Packet utils.py PREFIX_TO_TRAFFIC_ID; designs.py uses 1 for Non-VPN P2P (only affects max(4, k))")

# ---------- J. row-order locality: dependence of the reported range on the random comparison rows
pos = {c: np.asarray(v) for c, v in json.load(open(OUT2 / "arff_order_positions.json")).items()}
loc = {}
for seed in (0, 1, 2, 3):
    rng = np.random.default_rng(seed); vals = []
    for c in CTX:
        X, y = pool(c); Z = StandardScaler().fit_transform(logx2(X[FEATURES].values))
        for k in range(7):
            ids = np.flatnonzero(y == k); ids = ids[np.argsort(pos[c][ids])]; z = Z[ids]
            near = [np.linalg.norm(z[[j for j in range(max(0, i - 20), min(len(z), i + 21)) if j != i]] - z[i], axis=1).min() for i in range(len(z))]
            ran = [np.linalg.norm(z[rng.choice(np.delete(np.arange(len(z)), i), 40, replace=False)] - z[i], axis=1).min() for i in range(len(z))]
            vals.append(float(np.median(near) / np.median(ran)))
    loc[f"seed{seed}"] = [min(vals), max(vals)]
OUT["locality_win20_range_by_seed"] = loc

# ---------- K. BROWSING + P2P share of training flows
OUT["share_browsing_p2p_train"] = {c: float(np.isin(D[c][2], [0, 4]).mean()) for c in CTX}

# ---------- L. Table VII recomputed from the per-repeat CSVs (unrounded means)
t7 = {}
for m in ("XGB", "RF"):
    for dname in ("random", "grouped10", "grouped35", "grouped", "calibrated", "ordered10", "ordered35", "ordered140", "orderedcal", "orderedhalf", "null"):
        df = pd.read_csv(OUT2 / f"ref_summary_{m}_{dname}.csv")
        t7[f"{m}_{dname}"] = dict(f1_nv=df.f1_nv.mean(), f1_vpn=df.f1_vpn.mean(), w_nv=df.rho_wnv.mean(), w_vpn=df.rho_wvpn.mean(), cross=df.rho_cross.mean(),
                                  gap=df.rho_gap.mean(), sd=df.rho_gap.std(ddof=1), lo=df.rho_gap.min(), hi=df.rho_gap.max(),
                                  rbo=df.rbo_gap.mean(), top5=df.top5_gap.mean(), wgap=df.rho_w_gap.mean())
OUT["table7"] = t7

# ---------- M. model transfer against grouped CV
OUT["transfer_f1"] = {k: P2[k]["f1"] for k in P2 if k.startswith("transfer_") and k.split("_")[2] != k.split("_")[4]}
OUT["grouped_cv_f1"] = {k: P1[k]["f1"] if isinstance(P1[k], dict) and "f1" in P1[k] else P1[k] for k in P1 if k.startswith("cv_") and k.endswith("grouped")}

# ---------- N. class-mix swap: change in agreement with the other context
cm = pd.read_csv(OUT2 / "p4_classmix.csv")
for (m, c), g in cm[cm.mix != "nat_vs_swap"].groupby(["model", "ctx"]):
    p = g.pivot(index="r", columns="mix", values="rho_other")
    OUT.setdefault("classmix_delta", {})[f"{m}_{c}"] = dict(mean=float((p.swapped - p.natural).mean()), sd_natural=float(p.natural.std()), sd_swapped=float(p.swapped.std()))

# ---------- O. equal class weighting on the primary fits
OUT["class_weighting"] = {m: {k: (P1[f"cross_{m}"].get(k), P1[f"cross_{m}_weighted"].get(k)) for k in ("rho", "tau", "rbo", "l1", "top5", "top10")} for m in ("RF", "XGB")}

# ---------- P. Extra Trees vs Random Forest under the same TreeSHAP formulation (needs p8a_rf_interv.py)
try:
    et = {}
    for c in CTX:
        E = np.load(OUT2 / f"et_rec_{c}.npz"); F = np.load(OUT2 / f"rf_interv_rec_{c}.npz")
        S_ = lambda a: pd.Series(a.mean(0), index=FEATURES)
        et[c] = dict(path_path=compare(S_(E["ET_path"]), S_(E["RF_path"]))["rho"],
                     interv_interv=compare(S_(E["ET_interv"]), S_(F["RF_interv"]))["rho"],
                     interv_vs_path_v12=compare(S_(E["ET_interv"]), S_(E["RF_path"]))["rho"],
                     rf_path_arrays_identical=bool(np.allclose(E["RF_path"], F["RF_path"])),
                     rf_path_vs_interv=compare(S_(F["RF_path"]), S_(F["RF_interv"]))["rho"])
    rf_cross = compare(pd.Series(np.load(OUT2 / "rf_interv_rec_novpn.npz")["RF_interv"].mean(0), index=FEATURES),
                       pd.Series(np.load(OUT2 / "rf_interv_rec_vpn.npz")["RF_interv"].mean(0), index=FEATURES))["rho"]
    et["rf_interv_cross_rho_recomputed"] = rf_cross
    et["rf_interv_cross_rho_p3"] = json.load(open(OUT2 / "p3_results.json"))["rob_RF_interv"]["rho"]
    OUT["et_vs_rf"] = et
except FileNotFoundError as e:
    OUT["et_vs_rf"] = f"missing: {e}"

# ---------- Q. correlation clusters of Table XI regenerated in this environment
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
C = np.mean([np.abs(D[c][0].corr(method="spearman").values) for c in CTX], axis=0)
Dm = 1 - C; np.fill_diagonal(Dm, 0)
lab = fcluster(linkage(squareform(Dm, checks=False), method="average"), t=0.2, criterion="distance")
new = sorted(sorted(np.array(FEATURES)[lab == g].tolist()) for g in np.unique(lab))
old = sorted(sorted(v) for v in json.load(open(ROOT / "revision" / "results" / "clusters_0.2.json")).values())
OUT["clusters_0.2"] = dict(n_clusters=len(new), identical_to_revision1_file=new == old, clusters=new)

# ---------- R. XGBoost thread count: primary models refitted with one and two threads
th = {}
for c in CTX:
    Xtr, Xte, ytr, yte = D[c]
    p1 = make_model("XGB", n_jobs=1).fit(Xtr, ytr).predict_proba(Xte); p2 = make_model("XGB", n_jobs=2).fit(Xtr, ytr).predict_proba(Xte)
    pall = make_model("XGB", n_jobs=-1).fit(Xtr, ytr).predict_proba(Xte)   # all cores; c2.make_model defaults XGBoost to one thread
    th[c] = dict(identical_predictions=bool((p1.argmax(1) == p2.argmax(1)).all()), max_abs_prob_diff=float(np.abs(p1 - p2).max()),
                 all_cores_vs_one_max_abs_prob_diff=float(np.abs(p1 - pall).max()), n_cores=os.cpu_count(),
                 default_n_jobs_xgb=make_model("XGB").get_params()["n_jobs"])
OUT["xgb_threads"] = th
import xgboost
cfg = json.loads(make_model("XGB").fit(D["novpn"][0], D["novpn"][2]).get_booster().save_config())
mh = make_model("XGB"); mh.set_params(tree_method="hist")
ph = mh.fit(D["novpn"][0], D["novpn"][2]).predict_proba(D["novpn"][1])
pa = make_model("XGB").fit(D["novpn"][0], D["novpn"][2]).predict_proba(D["novpn"][1])
OUT["xgb_tree_method"] = dict(configured=make_model("XGB").get_params().get("tree_method"),
                              used=cfg["learner"]["gradient_booster"]["gbtree_train_param"]["tree_method"], xgboost_version=xgboost.__version__,
                              auto_vs_hist_max_abs_prob_diff_novpn=float(np.abs(pa - ph).max()))

jdump(OUT, "p8_v13.json")
print(json.dumps(OUT, indent=1, default=float)[:9000])
