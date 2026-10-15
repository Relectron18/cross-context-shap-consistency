"""Figures for the second revision."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from c2 import *
NV, VP = "#245a8d", "#d97a2b"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
CLS = ["BROWSING", "CHAT", "FT", "MAIL", "P2P", "STREAMING", "VOIP"]
which = sys.argv[1:] or ["2", "3", "4", "5", "7", "8"]
P1 = json.load(open(OUT2 / "p1_results.json"))

if "2" in which:
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=170); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    boxes = [
        (3, 80, 42, 14, "Primary train/test partitions (authors' 80/20 split)\nScenario A2, 15 s, duplicates removed\nSame 23 features and seven labels"),
        (53, 80, 44, 14, "Fit RF and XGB separately by context\nPrimary TreeSHAP rankings on test sets\nSpearman, Kendall, Top-k, $\\kappa_k$, RBO, L1"),
        (3, 52, 42, 21, "Within-context references (11 repeats each,\n80/20 partition redrawn every repeat)\n\u2022 random halves; permutation null\n\u2022 feature clusters (10/35/140, calibrated)\n\u2022 row blocks (10/35/140, calibrated, 1st/2nd half)"),
        (53, 52, 44, 21, "Confound and mechanism checks\nClass weighting; class-mix swap; matched\ntraining with a same-size reference\nDrop-one retraining; permutation importance\nFamilies, alternative grouping, clusters"),
        (3, 27, 42, 18, "Consequence tests\nDirect model transfer between contexts\nOwn vs transferred Top-5/10/15 (5 selection\nseeds; bottom-10 and 20 fixed random-10)"),
        (53, 27, 44, 18, "Sensitivity checks\nSentinel encodings S1/S2\nRF / XGB / Extra Trees; two SHAP methods\nXGB seeds and depth"),
        (3, 9, 94, 8, "All results computed in one environment: Python 3.13, scikit-learn 1.9.1, XGBoost 3.4.1, SHAP 0.53.0 (Linux)"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#eef2f8", ec="#61768f", lw=1.3))
        ax.text(x + 1.5, y + h / 2, t, va="center", ha="left", fontsize=9.4, linespacing=1.35)
    kw = dict(arrowprops=dict(arrowstyle="->", color="#61768f", lw=1.3))
    for a, b in [((52.4, 87), (45.6, 87)), ((24, 73.6), (24, 79.4)), ((75, 73.6), (75, 79.4)),
                 ((24, 45.6), (24, 51.4)), ((75, 45.6), (75, 51.4))]:
        ax.annotate("", xy=a, xytext=b, **kw)
    plt.savefig(FIG2 / "fig2_pipeline.png", bbox_inches="tight"); plt.close()

if "3" in which:
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), dpi=170, gridspec_kw=dict(width_ratios=[1, 1.6]))
    ax = axs[0]; x = np.arange(2); w = 0.36
    for i, (c, col, lab) in enumerate((("novpn", NV, "Non-VPN"), ("vpn", VP, "VPN"))):
        f1 = [P1[f"base_{m}_{c}"]["f1"] for m in ("RF", "XGB")]; acc = [P1[f"base_{m}_{c}"]["acc"] for m in ("RF", "XGB")]
        ax.bar(x + (i - .5) * w, f1, w, color=col, label=f"{lab} macro F1")
        ax.scatter(x + (i - .5) * w, acc, marker="D", color="k", s=18, zorder=3, label="Accuracy" if i == 0 else None)
        for xx, v in zip(x + (i - .5) * w, f1): ax.text(xx, v - 0.03, f"{v:.3f}", ha="center", va="top", color="white", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(["Random Forest", "XGBoost"]); ax.set_ylim(0.6, 1.0); ax.set_ylabel("Score")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False); ax.set_title("(a) Macro F1 and accuracy", fontsize=10)
    ax = axs[1]
    M_ = np.array([P1[f"base_{m}_{c}"]["recall_per_class"] for c in CTX for m in ("RF", "XGB")])
    im = ax.imshow(M_, cmap="Blues", vmin=0.7, vmax=1.0, aspect="auto")
    for i in range(4):
        for j in range(7): ax.text(j, i, f"{M_[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M_[i, j] > 0.9 else "black")
    ax.set_yticks(range(4)); ax.set_yticklabels(["Non-VPN RF", "Non-VPN XGB", "VPN RF", "VPN XGB"])
    ax.set_xticks(range(7)); ax.set_xticklabels(CLS, rotation=30, fontsize=8); ax.axhline(1.5, color="white", lw=3)
    ax.set_title("(b) Recall per class", fontsize=10); fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(FIG2 / "fig3_baseline.png", bbox_inches="tight"); plt.close()

def prim_share(m, c):
    s = pd.Series(np.load(OUT2 / f"primary_rec_{m}_{c}.npy").mean(0), index=FEATURES); return s / s.sum() * 100

if "4" in which:
    fig, axs = plt.subplots(1, 2, figsize=(11, 8.2), dpi=170)
    for ax, m, t in zip(axs, ("RF", "XGB"), ("Random Forest", "XGBoost")):
        a, b = prim_share(m, "novpn"), prim_share(m, "vpn"); order = a.sort_values(ascending=False).index[::-1]
        y = np.arange(23); ax.barh(y - .19, a[order], .38, color=NV, label="Non-VPN"); ax.barh(y + .19, b[order], .38, color=VP, label="VPN")
        ax.set_yticks(y); ax.set_yticklabels(order); ax.set_xlabel("Share of total mean |SHAP| (%)"); ax.set_title(t); ax.legend(frameon=False, loc="lower right")
    plt.tight_layout(); plt.savefig(FIG2 / "fig4_shares.png", bbox_inches="tight"); plt.close()

if "5" in which:
    fig, axs = plt.subplots(1, 2, figsize=(11, 8.6), dpi=170)
    for ax, m, t in zip(axs, ("RF", "XGB"), ("Random Forest", "XGBoost")):
        ra = ranks(prim_share(m, "novpn")); rb = ranks(prim_share(m, "vpn"))
        for f in FEATURES:
            moved = abs(ra[f] - rb[f]) >= 3
            ax.plot([0, 1], [ra[f], rb[f]], color=VP if moved else "#aab4bf", lw=1.6 if moved else 1.2)
            ax.text(-0.03, ra[f], f, ha="right", va="center", fontsize=8); ax.text(1.03, rb[f], f, ha="left", va="center", fontsize=8)
        for k in (5.5, 10.5): ax.axhline(k, color="grey", ls=":", lw=0.8)
        ax.set_ylim(23.8, 0.2); ax.set_xlim(-0.6, 1.6); ax.set_xticks([0, 1]); ax.set_xticklabels(["Non-VPN", "VPN"]); ax.set_yticks([])
        ax.spines["left"].set_visible(False); ax.set_title(t)
    plt.tight_layout(); plt.savefig(FIG2 / "fig5_ranks.png", bbox_inches="tight"); plt.close()

if "7" in which:
    designs = [("random", "Random\nhalves"), ("grouped10", "Grouped\n~10/cluster"), ("grouped35", "Grouped\n~35/cluster"),
               ("grouped", "Grouped\n~140/cluster"), ("calibrated", "Grouped\ncalibrated"),
               ("ordered10", "Rows\n~10/block"), ("ordered35", "Rows\n~35/block"), ("ordered140", "Rows\n~140/block"),
               ("orderedcal", "Rows\ncalibrated"), ("orderedhalf", "Rows\n1st/2nd half"), ("null", "Permutation\nnull")]
    fig, axs = plt.subplots(2, 1, figsize=(12, 8.4), dpi=170)
    for ax, m, t in zip(axs, ("XGB", "RF"), ("XGBoost", "Random Forest")):
        xs = []
        for i, (d, lab) in enumerate(designs):
            p = OUT2 / f"ref_summary_{m}_{d}.csv"
            if not p.exists(): continue
            df = pd.read_csv(p); xs.append((i, f"{lab}\n\u0394 = {df.rho_gap.mean():+.3f}"))
            w = np.r_[df.rho_wnv, df.rho_wvpn]; cr = df.rho_cross
            j = np.random.default_rng(i).uniform(-.07, .07, len(w))
            ax.scatter(i - .15 + j, w, s=12, color="#4c8c4a", alpha=.75, label="Within context" if i == 0 else None)
            ax.scatter(i + .15 + j[:len(cr)], cr, s=12, color="#8e44ad", alpha=.75, label="Across contexts" if i == 0 else None)
            ax.hlines([w.mean()], i - .3, i, color="#2d5a2b", lw=2); ax.hlines([cr.mean()], i, i + .3, color="#5b2a72", lw=2)
        ax.set_xticks([x for x, _ in xs]); ax.set_xticklabels([l for _, l in xs], rotation=0, fontsize=7.5)
        for xv in (0.5, 4.5, 9.5): ax.axvline(xv, color="grey", lw=0.6, ls=":")
        ax.set_ylim(0.55 if m == "XGB" else 0.0, 1.0); ax.set_title(t); ax.grid(axis="y", alpha=.3)
    for ax in axs: ax.set_ylabel("Spearman ρ between rankings")
    fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(); plt.savefig(FIG2 / "fig7_references.png", bbox_inches="tight"); plt.close()

if "8" in which:
    P2 = json.load(open(OUT2 / "p2_results.json")); tk = pd.read_csv(OUT2 / "p2_topk.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.0), dpi=170)
    ax = axs[0]; labs, own, trn = [], [], []
    for m in ("RF", "XGB"):
        for c, cn in (("novpn", "Non-VPN"), ("vpn", "VPN")):
            o = OTHER = "vpn" if c == "novpn" else "novpn"
            labs.append(f"{m}\n{cn} test"); own.append(P2[f"transfer_{m}_{c}_to_{c}"]["f1"]); trn.append(P2[f"transfer_{m}_{o}_to_{c}"]["f1"])
    x = np.arange(4); ax.bar(x - .19, own, .38, color="#5b7fa6", label="Trained in same context"); ax.bar(x + .19, trn, .38, color="#c0504d", label="Trained in other context")
    for xx, v in zip(np.r_[x - .19, x + .19], own + trn): ax.text(xx, v + .01, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8); ax.set_ylim(0, 1.18); ax.set_ylabel("Macro F1"); ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_title("(a) Direct model transfer", fontsize=10)
    ax = axs[1]
    for i, (m, c) in enumerate([(m, c) for m in ("RF", "XGB") for c in CTX]):
        for jk, (k, mk_) in enumerate(((5, "o"), (10, "s"), (15, "^"))):
            d = tk[(tk.model == m) & (tk.target == c) & (tk.k == k)]["diff"]
            xx = i + (jk - 1) * .22
            ax.scatter(np.full(len(d), xx), d, s=10, color="#888", zorder=2)
            ax.scatter([xx], [d.mean()], marker=mk_, s=36, color="k", zorder=3, label=f"k = {k} (mean of 5 seeds)" if i == 0 else None)
    ax.axhline(0, color="grey", lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels([f"{m}\n{'Non-VPN' if c=='novpn' else 'VPN'} target" for m in ("RF", "XGB") for c in CTX], fontsize=8)
    ax.set_ylabel("Macro F1: own − transferred Top-k"); ax.legend(frameon=False, fontsize=7.5); ax.set_title("(b) Transfer of a Top-k feature list", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG2 / "fig8_consequence.png", bbox_inches="tight"); plt.close()
print("ok", which)
