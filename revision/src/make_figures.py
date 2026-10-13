"""Redraw Fig. 2 (pipeline) and Fig. 6 (feature correlation heatmaps) of the revised paper.
Run from the repository root: python revision/src/make_figures.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from common import *

FIG = ROOT / "revision" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ---- Fig. 2: pipeline
fig, ax = plt.subplots(figsize=(10.5, 6.6), dpi=170); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
boxes = [
    (3, 80, 42, 14, "Provided train/test partitions\n(Scenario A2, 15 s, duplicates removed)\nSame 23 features and seven labels"),
    (53, 80, 44, 14, "Fit models separately by context\nRF 200 • XGB 200 • LR 2,000 iterations\nPrimary TreeSHAP rankings on test sets"),
    (3, 55, 42, 18, "Within-context references\n30 training bootstraps (fixed seed)\nDisjoint, size-matched half-splits\nwith separate seeds (main reference)"),
    (53, 55, 44, 18, "Agreement measures\nSpearman, Kendall, Top-k overlap,\nRBO and L1 share distance;\neffect sizes with split-based spread"),
    (3, 28, 42, 20, "Confound and grouping checks\nClass weighting and class-matched training\nSentinel prevalence; feature correlations\nSix families, per-member means,\ncorrelation clusters"),
    (53, 28, 44, 20, "Sensitivity checks\nSentinel encodings S1/S2\nRF / XGB / Extra Trees; two SHAP methods\nModel seeds and XGBoost depth"),
    (3, 3, 94, 18, "Top-k consequence test (training-only selection, untouched test evaluation)\nOwn vs transferred Top-5 / Top-10 / Top-15, with bottom-10 and random-10 controls"),
]
for x, y, w, h, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#eef2f8", ec="#61768f", lw=1.3))
    ax.text(x + 1.5, y + h / 2, t, va="center", ha="left", fontsize=9.6, linespacing=1.35)
kw = dict(arrowprops=dict(arrowstyle="->", color="#61768f", lw=1.3))
for a, b in [((52.4, 87), (45.6, 87)), ((24, 73.6), (24, 79.4)), ((75, 73.6), (75, 79.4)),
             ((24, 48.6), (24, 54.4)), ((75, 48.6), (75, 54.4)), ((50, 21.6), (50, 27.4))]:
    ax.annotate("", xy=a, xytext=b, **kw)
plt.savefig(FIG / "fig2_pipeline.png", bbox_inches="tight"); plt.close()

# ---- Fig. 6: Spearman correlation matrices on the training partitions
order = ["duration", "total_fiat", "total_biat", "min_fiat", "min_biat", "max_fiat", "max_biat", "mean_fiat", "mean_biat",
         "min_flowiat", "max_flowiat", "mean_flowiat", "std_flowiat", "flowPktsPerSecond", "flowBytesPerSecond",
         "min_active", "mean_active", "max_active", "std_active", "min_idle", "mean_idle", "max_idle", "std_idle"]
mins = {"min_fiat", "min_biat", "min_flowiat"}
fig, axs = plt.subplots(1, 2, figsize=(12.5, 6.2), dpi=170, constrained_layout=True)
for ax, (ctx, name) in zip(axs, (("novpn", "(a) Non-VPN"), ("vpn", "(b) VPN"))):
    c = load(ctx)[0].corr(method="spearman").loc[order, order].to_numpy()
    im = ax.imshow(c, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(23)); ax.set_yticks(range(23))
    ax.set_xticklabels(order, rotation=90, fontsize=7); ax.set_yticklabels(order, fontsize=7)
    ax.set_title(name, fontsize=11)
    for i in range(23):
        for j in range(23):
            if i != j and order[i] in mins and order[j] in mins:
                ax.text(j, i, f"{c[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
fig.colorbar(im, ax=axs, shrink=0.75, label="Spearman correlation (training data)")
plt.savefig(FIG / "fig6_feature_correlations.png", bbox_inches="tight"); plt.close()
print("figures written to", FIG)
