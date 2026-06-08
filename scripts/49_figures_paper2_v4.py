"""
Phase B v4 - Nature/Cell-style figure rebuild for paper v7.

Changes vs script 46:
- Palette: matplotlib tab10 defaults replaced with Okabe-Ito 8-colour palette
  (colour-blind safe, grayscale-readable, low-saturation Nature/Science standard).
- Font: Arial/Helvetica family with DejaVu Sans fallback.
- Grid: thin gray (#e8e8e8) horizontal-only on quantitative axes; reduces
  visual noise vs the chart background.
- Spines: top and right hidden; remaining spines slightly thicker (1.0 vs 0.8)
  for crispness at print resolution.
- F6 traffic-light tiers desaturated but semantically preserved.
- F7 ROI diverging cmap RdYlGn -> RdBu_r (Nature convention for signed-saving).
- F1 workflow boxes: soft fills with darker stroke borders, Nature workflow style.

Output naming unchanged from script 46 (figures overwrite in place so v7 export
picks them up without changing any image path):
  figures/F1_workflow.{png,pdf}
  figures/F2_band_sweep.{png,pdf}
  figures/F3_transfer_risk.{png,pdf}
  figures/F4_cross_batch_CIs.{png,pdf}
  figures/F5_gbm_pdp.{png,pdf}
  figures/F6_three_tier_cube.{png,pdf}
  figures/F7_roi_sensitivity.{png,pdf}
  figures/supplementary/S1_reliability.{png,pdf}
  figures/supplementary/S2_three_tier_pixel.{png,pdf}
  figures/supplementary/S3_healthy_bias.{png,pdf}
"""
import os, sys
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

RES  = _os.environ.get("PISTACHIO_RES", "results")
FIG  = _os.environ.get("PISTACHIO_FIG", _os.path.join(RES, "figures"))
SUP  = os.path.join(FIG, "supplementary")
os.makedirs(SUP, exist_ok=True)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 1.0, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e8e8e8", "grid.linewidth": 0.6, "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "legend.frameon": True, "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc", "legend.fancybox": False,
})

# Palette D: Mono-Teal + Accent (single-hue family + one warm accent).
# Designed to encode narrative: SG2 trio = three teal shades (one perceptual
# family, "they're being compared as one group"); Ridge+SNV gets the only
# warm accent ("this is the recommended pipeline"); GBM+SNV gets charcoal
# ("the failure mode is intentionally subdued").
# Saturation is coordinated across the 5 colours - no vivid-vs-muted mix.
PAL = {
    "dark_teal":   "#0F4C5C",
    "med_teal":    "#5B868C",
    "pale_teal":   "#93B7BE",
    "burnt_orange":"#E36414",  # ACCENT - reserved for the recommended pipeline
    "charcoal":    "#444444",
    # Coordinated traffic-light tier hues (semantic = safety, NOT pipeline)
    "tier_safe":   "#3E7B68",  # muted sage green (release)
    "tier_caution":"#C99B45",  # muted gold (lab-confirm)
    "tier_alert":  "#B33E3E",  # muted brick red (reject) - distinct from accent
}

# Per-pipeline assignments. The whole story of Paper 2 - "preprocessing
# outweighs model class; Ridge+SNV is the answer" - is encoded directly
# in the colour: SG2 trio reads as "control group", Ridge+SNV reads as
# "hero", GBM+SNV reads as "failure mode".
C = {
    "Ridge_SG2":   PAL["dark_teal"],
    "RF_SG2":      PAL["med_teal"],
    "GBM_SG2":     PAL["pale_teal"],
    "Ridge_SNV":   PAL["burnt_orange"],
    "GBM_SNV":     PAL["charcoal"],
    "imgmean":     PAL["pale_teal"],
    "neutral":     PAL["charcoal"],
    "accent":      PAL["burnt_orange"],
}
LABEL = {
    "Ridge_SG2":  "Ridge + SG2",     "RF_SG2":   "RF + SG2",
    "GBM_SG2":    "GBM + SG2",       "Ridge_SNV":"Ridge + SG2+SNV",
    "GBM_SNV":    "GBM + SG2+SNV",
}
W1, W2 = 3.54, 7.09


def save_main(name):
    out = os.path.join(FIG, name)
    plt.savefig(out + ".png", dpi=300); plt.savefig(out + ".pdf")
    plt.close(); print(f"[main] {name}", flush=True)


def save_sup(name):
    out = os.path.join(SUP, name)
    plt.savefig(out + ".png", dpi=300); plt.savefig(out + ".pdf")
    plt.close(); print(f"[supp] {name}", flush=True)


# ============================================================
# F1 - Workflow (renamed boxes)
# ============================================================
def f1():
    fig, ax = plt.subplots(figsize=(W2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off()
    # F1 disables the global y-grid; it's a schematic, not a plot.
    ax.grid(False)

    # Soft fills (light tint) with a darker stroke from the same hue.
    # Coordinated with palette D: teal family for the analytical pipeline,
    # burnt-orange accent for the recommended preprocessing (SG2+SNV).
    BOX_DATA   = ("#FFF0E5", PAL["burnt_orange"])  # warm cream / burnt orange (input data calls back to accent)
    BOX_PIPE   = ("#E5EFF1", PAL["med_teal"])      # pale teal / medium teal (pipeline step)
    BOX_PREPA  = ("#D9E5E8", PAL["dark_teal"])     # pale teal / dark teal (SG2 = convention)
    BOX_PREPB  = ("#FCEBE2", PAL["burnt_orange"])  # pale apricot / burnt orange (SG2+SNV = the recommended branch)
    BOX_MODEL  = ("#FFFFFF", PAL["charcoal"])      # clean white / charcoal (model)
    BOX_OUT    = ("#F4F7F8", PAL["dark_teal"])     # pale cool gray / dark teal (output / metrics)

    def box(xy, w, h, text, fc_ec, **kw):
        fc, ec = fc_ec
        ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.05",
                                     fc=fc, ec=ec, lw=1.2))
        ax.text(xy[0]+w/2, xy[1]+h/2, text, ha="center", va="center",
                fontsize=8.5, color="#222", **kw)

    def arrow(x1,y1,x2,y2):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle="-|>",
                                      mutation_scale=12, lw=1, color="#666"))

    box((0.2, 4.5), 2.0, 1.0, "v1 (Sep 2025)\n57 cubes\n0.40-33.17 µg/kg", BOX_DATA)
    box((0.2, 3.1), 2.0, 1.0, "v3 (May 2026)\n52 cubes\n0.00-114.67 µg/kg", BOX_DATA)
    box((2.7, 3.8), 1.8, 1.0,
        "Pixel sampling\nq=0.55 @ 650 nm\n300 px / cube", BOX_PIPE)
    box((4.9, 4.7), 1.6, 0.9, "SG2", BOX_PREPA)
    box((4.9, 3.1), 1.6, 0.9, "SG2 + SNV", BOX_PREPB)
    box((6.9, 4.7), 1.6, 0.9, "Ridge / RF / GBM", BOX_MODEL)
    box((6.9, 3.1), 1.6, 0.9, "Ridge / GBM", BOX_MODEL)
    box((1.2, 1.0), 3.0, 1.0,
        "Per-cube ('lot') aggregation\n(mean of pixel scores)", BOX_OUT)
    box((4.7, 1.0), 4.8, 1.0,
        "Pre-screening decision metrics:\nAUC, PR-AUC, FPR @100% recall,\n3-tier risk, scenario-analysis ROI",
        BOX_OUT)

    arrow(2.2, 5.0, 2.7, 4.6)
    arrow(2.2, 3.6, 2.7, 4.0)
    arrow(4.5, 4.6, 4.9, 5.1)
    arrow(4.5, 4.0, 4.9, 3.5)
    arrow(6.5, 5.1, 6.9, 5.1)
    arrow(6.5, 3.5, 6.9, 3.5)
    arrow(7.7, 4.7, 4.2, 2.0)
    arrow(7.7, 3.1, 4.2, 2.0)
    arrow(4.2, 1.5, 4.7, 1.5)
    save_main("F1_workflow")


# ============================================================
# F2 - Band sweep (unchanged geometry, restored)
# ============================================================
def f2():
    df = pd.read_csv(os.path.join(RES, "26_band_sweep.tsv"), sep="\t")
    sup = df[df["scheme"].str.startswith("sup_top") | (df["scheme"] == "all_462")].copy()
    sup["k"] = sup["k"].astype(int); sup = sup.sort_values("k")
    fig, ax = plt.subplots(figsize=(W1*1.5, 3.2))
    # 8 µg/kg main analysis = dark teal (consistent with the "main analytical
    # pipeline" tone in palette D). 2 µg/kg sub-detection-limit comparison =
    # brick red (the same hue palette D reserves for the "reject / high-risk"
    # tier in F6) — semantically a "warning / unreliable" colour that gives
    # strong warm-vs-cool hue contrast with the dark-teal main line.
    ax.plot(sup["k"], sup["AUC_8ppb"], "o-",
             color=C["Ridge_SG2"], lw=1.8, ms=5, label="AUC @ 8 µg/kg")
    ax.plot(sup["k"], sup["AUC_2ppb"], "s--",
             color=PAL["tier_alert"], lw=1.5, ms=4.5, label="AUC @ 2 µg/kg")
    diaper = df[df["scheme"] == "diaper_5"]["AUC_8ppb"].iloc[0]
    uniform = df[df["scheme"] == "uniform_5"]["AUC_8ppb"].iloc[0]
    ax.scatter([5], [diaper], marker="^", color="#444", s=40, zorder=5,
                label=f"Williams OSP k=5 ({diaper:.2f})")
    ax.scatter([5], [uniform], marker="v", color="#888", s=40, zorder=5,
                label=f"uniform k=5 ({uniform:.2f})")
    ax.axhline(0.5, color="#999", ls=":", lw=0.8, label="chance")
    ax.axhline(0.8, color="#bbb", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10, 20, 50, 100, 200, 462])
    ax.set_xticklabels([1, 3, 5, 10, 20, 50, 100, 200, 462])
    ax.set_xlabel("Number of bands $k$ (supervised |$\\beta$| top-$k$, in-fold)")
    ax.set_ylabel("In-domain pixel ROC-AUC")
    ax.set_ylim(0.35, 0.85)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7)
    save_main("F2_band_sweep")


# ============================================================
# F3 - Transfer-risk diagnostic (unchanged from v2 wide version)
# ============================================================
def f3():
    r29 = pd.read_csv(os.path.join(RES, "29_confound_regression.tsv"), sep="\t")
    r30 = pd.read_csv(os.path.join(RES, "30_gbm_residual.tsv"), sep="\t")
    cmp = pd.read_csv(os.path.join(RES, "42_compare_vs_full_pixel.tsv"), sep="\t")

    ridge_raw = r29[(r29.input=="raw")&(r29.model=="Ridge")]["AUC_8ppb"].iloc[0]
    ridge_res = r29[(r29.input=="residual")&(r29.model=="Ridge")]["AUC_8ppb"].iloc[0]
    rf_raw    = r29[(r29.input=="raw")&(r29.model=="RF")]["AUC_8ppb"].iloc[0]
    rf_res    = r29[(r29.input=="residual")&(r29.model=="RF")]["AUC_8ppb"].iloc[0]
    gbm_raw = r30[r30["input"]=="raw"]["AUC_8ppb"].iloc[0]
    gbm_res = r30[r30["input"]=="residual"]["AUC_8ppb"].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W2, 3.8))
    plt.subplots_adjust(top=0.85, wspace=0.30)

    models = ["Ridge", "RF", "GBM"]
    raw = [ridge_raw, rf_raw, gbm_raw]
    res = [ridge_res, rf_res, gbm_res]
    x = np.arange(len(models)); w = 0.35
    ax1.bar(x - w/2, raw, w, color=[C["Ridge_SG2"], C["RF_SG2"], C["GBM_SG2"]],
            label="raw spectrum (in-domain pixel)")
    ax1.bar(x + w/2, res, w,
            color=[C["Ridge_SG2"], C["RF_SG2"], C["GBM_SG2"]],
            alpha=0.4, hatch="///",
            label="image-mean residual (in-domain pixel)")
    ax1.axhline(0.5, color="#999", ls=":", lw=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_ylabel("In-domain pixel AUC @ 8 µg/kg"); ax1.set_ylim(0.35, 1.0)
    ax1.set_title("A. Residual-regression diagnostic", loc="left", fontweight="bold")
    ax1.legend(loc="lower left", fontsize=7, framealpha=0.9)
    for i, (r, e) in enumerate(zip(raw, res)):
        ax1.text(i - w/2, r + 0.01, f"{r:.2f}", ha="center", fontsize=7)
        ax1.text(i + w/2, e + 0.01, f"{e:.2f}", ha="center", fontsize=7)

    preps = ["SG2", "SG2+SNV"]
    full_in = [cmp.iloc[0]["full_pixel_indomain"], cmp.iloc[1]["full_pixel_indomain"]]
    mean_in = [cmp.iloc[0]["image_mean_indomain"], cmp.iloc[1]["image_mean_indomain"]]
    full_xb = [cmp.iloc[0]["full_pixel_xbatch"], cmp.iloc[1]["full_pixel_xbatch"]]
    mean_xb = [cmp.iloc[0]["image_mean_xbatch"], cmp.iloc[1]["image_mean_xbatch"]]
    x = np.arange(len(preps)); w = 0.18
    ax2.bar(x - 1.5*w, full_in, w, color=C["Ridge_SG2"], label="Full-pixel, in-domain")
    ax2.bar(x - 0.5*w, mean_in, w, color=C["Ridge_SG2"], alpha=0.5,
            label="Image-mean, in-domain (LOO)")
    ax2.bar(x + 0.5*w, full_xb, w, color=C["Ridge_SNV"], label="Full-pixel, cross-batch")
    ax2.bar(x + 1.5*w, mean_xb, w, color=C["Ridge_SNV"], alpha=0.5,
            label="Image-mean, cross-batch")
    ax2.axhline(0.5, color="#999", ls=":", lw=0.8)
    ax2.set_xticks(x); ax2.set_xticklabels(preps)
    ax2.set_ylabel("Ridge cube AUC @ 8 µg/kg"); ax2.set_ylim(0.35, 1.05)
    ax2.set_title("B. Image-mean baseline vs full-pixel", loc="left", fontweight="bold")
    ax2.legend(loc="lower right", fontsize=7, framealpha=0.9)
    for i, vs in enumerate(zip(full_in, mean_in, full_xb, mean_xb)):
        for off, v in zip([-1.5*w, -0.5*w, 0.5*w, 1.5*w], vs):
            ax2.text(i + off, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
    save_main("F3_transfer_risk")


# ============================================================
# F4 - Cross-batch with GBM+SNV included (v3 fix)
# ============================================================
def f4():
    df = pd.read_csv(os.path.join(RES, "41_bootstrap_CIs.tsv"), sep="\t")
    pipelines = ["Ridge_SG2", "RF_SG2", "GBM_SG2", "Ridge_SNV", "GBM_SNV"]
    thresholds = [8, 10, 15]
    fig, ax = plt.subplots(figsize=(W2 * 0.8, 3.5))
    x = np.arange(len(thresholds))
    bar_w = 0.15
    offsets = np.linspace(-2, 2, len(pipelines)) * bar_w
    for i, p in enumerate(pipelines):
        sub = df[df.pipeline == p].sort_values("threshold")
        if sub.empty: continue
        means = sub["AUC"].values
        lo = sub["AUC"].values - sub["AUC_lo"].values
        hi = sub["AUC_hi"].values - sub["AUC"].values
        pos = x + offsets[i]
        ax.bar(pos, means, bar_w,
               yerr=[lo, hi], capsize=2.0,
               color=C[p], label=LABEL[p],
               error_kw=dict(lw=0.7, ecolor="#333"))
    ax.axhline(0.5, color="#999", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{t} µg/kg" for t in thresholds])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Cross-batch cube ROC-AUC (95% CI)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", fontsize=7, framealpha=0.9, ncol=2)
    save_main("F4_cross_batch_CIs")


# ============================================================
# F5 - GBM PDP (unchanged from v2 fixed version, just re-rendered)
# ============================================================
def f5():
    df = pd.read_csv(os.path.join(RES, "40_pdp_summary.tsv"),
                     sep="\t").head(10).reset_index(drop=True)
    sign_flip = np.sign(df["GBM_SG2_PDP_corr"]) != np.sign(df["GBM_SNV_PDP_corr"])
    n_flip = int(sign_flip.sum())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W1*1.8, 4.2),
                                    gridspec_kw=dict(height_ratios=[1.4, 1.0],
                                                     hspace=0.35))
    x = np.arange(len(df)); bar_w = 0.4
    ax1.bar(x - bar_w/2, df["GBM_SG2_PDP_corr"], bar_w,
             color=C["GBM_SG2"], label="GBM + SG2")
    ax1.bar(x + bar_w/2, df["GBM_SNV_PDP_corr"], bar_w,
             color=C["GBM_SNV"], label="GBM + SG2+SNV")
    for i, flip in enumerate(sign_flip):
        if flip:
            ax1.axvspan(i - 0.5, i + 0.5, color="#ffe7e0", alpha=0.7, zorder=0)
    ax1.axhline(0, color="#444", lw=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{w:.0f}" for w in df["wavelength_nm"]], rotation=0, fontsize=7)
    ax1.set_ylabel("PDP-band correlation\n(monotonic direction)")
    ax1.set_ylim(-1.05, 1.05); ax1.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax1.set_title(f"A. SNV flips PDP direction on {n_flip} of 10 top bands (shaded)",
                   loc="left", fontweight="bold")
    ax2.bar(x - bar_w/2, df["GBM_SG2_PDP_range"], bar_w, color=C["GBM_SG2"])
    ax2.bar(x + bar_w/2, df["GBM_SNV_PDP_range"], bar_w, color=C["GBM_SNV"])
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{w:.0f}" for w in df["wavelength_nm"]], rotation=0, fontsize=7)
    ax2.set_xlabel("Wavelength (nm), top-10 Ridge |$\\beta$| bands")
    ax2.set_ylabel("PDP range\n(magnitude)")
    ax2.set_title("B. Magnitude is not uniformly reduced", loc="left", fontweight="bold")
    save_main("F5_gbm_pdp")


# ============================================================
# F6 - Three-tier matrix, cube-level only (was F7 Panel A)
# ============================================================
def f6_three_tier_cube():
    df = pd.read_csv(os.path.join(RES, "37c_ridgeSNV_tier.tsv"), sep="\t")
    lot = df[(df.scope == "cross-batch") & (df.level == "lot_mean")]
    fig, ax = plt.subplots(figsize=(W1*1.4, 3.2))
    tiers = ["low", "mid", "high"]
    tier_label = ["release\n(low risk)", "lab-confirm\n(mid risk)", "reject\n(high risk)"]
    # Traffic-light semantics preserved with palette-D-coordinated tier hues.
    # Distinct from the burnt-orange accent so reject-tier never confuses with
    # the recommended-pipeline accent in adjacent figures.
    colors = [PAL["tier_safe"], PAL["tier_caution"], PAL["tier_alert"]]
    ax.bar(tiers, lot["pct"], color=colors, alpha=0.92,
           edgecolor="#444444", linewidth=0.6)
    for i, (n, p, u) in enumerate(zip(lot["n"], lot["pct"], lot["true_unsafe_rate"])):
        ax.text(i, p + 1.5, f"n={int(n)} ({p:.1f}%)\nactual unsafe={u*100:.0f}%",
                ha="center", fontsize=7.5, color="#222")
    ax.set_xticks(range(3)); ax.set_xticklabels(tier_label)
    ax.set_ylabel("Fraction of cubes (%)")
    ax.set_ylim(0, max(lot["pct"]) * 1.4)
    save_main("F6_three_tier_cube")


# ============================================================
# F7 - ROI heatmap (was F8)
# ============================================================
def f7_roi():
    from matplotlib.colors import LinearSegmentedColormap
    df = pd.read_csv(os.path.join(RES, "35_roi_sensitivity.tsv"), sep="\t")
    piv = df.pivot(index="hplc_per_lot_usd", columns="liability_per_lot_usd",
                   values="saving_RidgeSNV_xb_vs_A") / 1e6
    fig, ax = plt.subplots(figsize=(W1*1.5, 3.2))
    # F7 disables the inherited y-grid: this is a heatmap.
    ax.grid(False)
    # Custom diverging cmap aligned with palette D: dark teal (loss) -> white
    # (neutral) -> burnt orange (gain). The "favourable" gain colour is the
    # same accent the rest of the paper reserves for the recommended pipeline.
    teal_orange = LinearSegmentedColormap.from_list(
        "teal_orange",
        [PAL["dark_teal"], "#5B868C", "#FFFFFF", "#F2A56B", PAL["burnt_orange"]],
    )
    vmax = max(abs(piv.values.min()), abs(piv.values.max()))
    im = ax.imshow(piv.values, cmap=teal_orange, aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"${int(c/1000)}k" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"${int(r)}" for r in piv.index])
    ax.set_xlabel("Recall liability per unsafe cube (USD)")
    ax.set_ylabel("HPLC cost per cube (USD)")
    # Cell text colour: white on deep red/blue, dark on light cells.
    for i in range(len(piv.index)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:+.2f}M", ha="center", va="center", fontsize=7,
                    color="white" if abs(v) > vmax * 0.55 else "#222")
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Annual saving vs HPLC-every-cube\n(USD M, 20k cubes/yr)", fontsize=8)
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("#cccccc")
    save_main("F7_roi_sensitivity")


# ============================================================
# S1 - Reliability (was F6)
# ============================================================
def s1_reliability():
    y3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"),
                     sep="\t")["AFB1_ppb"].values
    pred_snv = np.load(os.path.join(RES, "37_pred_v3_RidgeSNV.npy"))
    pred_gbm = np.load(os.path.join(RES, "32_pred_v3_GBM_iter200_d8.npy"))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 3.5))
    plt.subplots_adjust(wspace=0.35, bottom=0.18, top=0.88)
    for ax, pred, name, col in [
        (axes[0], pred_snv, "Ridge + SG2+SNV (cross-batch)", C["Ridge_SNV"]),
        (axes[1], pred_gbm, "GBM + SG2 (cross-batch)", C["GBM_SG2"]),
    ]:
        p = (pred - pred.min()) / (pred.max() - pred.min() + 1e-9)
        yb = (y3 >= 8).astype(int)
        bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        ob, pr, ct = [], [], []
        for b in range(10):
            sel = idx == b
            if sel.sum() < 30: continue
            ob.append(yb[sel].mean()); pr.append(p[sel].mean()); ct.append(sel.sum())
        ax.plot([0,1], [0,1], "--", color="#999", lw=0.8, label="perfect")
        ax.plot(pr, ob, "o-", color=col, lw=1.5, ms=4, label="observed")
        ax.scatter(pr, ob, s=np.array(ct)/30, color=col, alpha=0.3)
        ax.set_xlabel("Normalised risk score")
        ax.set_ylabel("Empirical positive rate")
        ax.set_title(name, loc="left", fontweight="bold", fontsize=8.5)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    save_sup("S1_reliability")


# ============================================================
# S2 - Three-tier pixel-level (was F7 Panel B)
# ============================================================
def s2_three_tier_pixel():
    df = pd.read_csv(os.path.join(RES, "37c_ridgeSNV_tier.tsv"), sep="\t")
    pix = df[(df.scope == "in-domain") & (df.level == "pixel")]
    fig, ax = plt.subplots(figsize=(W1*1.4, 3.2))
    tiers = ["low", "mid", "high"]
    tier_label = ["release\n(low risk)", "lab-confirm\n(mid risk)", "reject\n(high risk)"]
    colors = [PAL["tier_safe"], PAL["tier_caution"], PAL["tier_alert"]]
    ax.bar(tiers, pix["pct"], color=colors, alpha=0.92,
           edgecolor="#444444", linewidth=0.6)
    for i, (n, p, u) in enumerate(zip(pix["n"], pix["pct"], pix["true_unsafe_rate"])):
        ax.text(i, p + 1.5, f"n={int(n)}\n({p:.1f}%)\nunsafe={u*100:.1f}%",
                ha="center", fontsize=7, color="#222")
    ax.set_xticks(range(3)); ax.set_xticklabels(tier_label)
    ax.set_ylabel("Fraction of pixels (%)")
    ax.set_ylim(0, max(pix["pct"]) * 1.35)
    save_sup("S2_three_tier_pixel")


# ============================================================
# S3 - Healthy bias (was F9)
# ============================================================
def s3_healthy_bias():
    df = pd.read_csv(os.path.join(RES, "39_healthy_bias_mitigation.tsv"), sep="\t")
    fwd = df[df.direction == "forward (v1->v3)"].copy()
    rev = df[df.direction == "reverse (v3->v1)"].copy()
    models = fwd["model"].tolist()
    nice = {"Ridge_baseline":"Ridge baseline",
            "GBM_MSE_baseline":"GBM (MSE)",
            "GBM_quantile":"GBM (quantile q=0.5)",
            "GBM_monotonic":"GBM (monotonic top-10)",
            "TwoStage_Ridge":"Two-stage Ridge"}
    labels = [nice.get(m, m) for m in models]
    bias_fwd = fwd["healthy_bias_lot"].values
    bias_rev = rev["healthy_bias_lot"].values
    auc_fwd  = fwd["lot_AUC_8"].values
    fig, ax = plt.subplots(figsize=(W2, 3.5))
    plt.subplots_adjust(wspace=0.35, bottom=0.22, top=0.88)
    x = np.arange(len(models)); w = 0.35
    ax.bar(x - w/2, bias_fwd, w, color=C["Ridge_SG2"],
            label="Forward (v1→v3) bias on Level 01")
    ax.bar(x + w/2, bias_rev, w, color=C["GBM_SG2"],
            label="Reverse (v3→v1) bias on Level 01")
    for i, (b, a) in enumerate(zip(bias_fwd, auc_fwd)):
        ax.text(i - w/2, b + 0.15, f"AUC={a:.2f}", ha="center",
                 fontsize=6.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mean predicted µg/kg on true-zero cubes (bias)")
    ax.axhline(0, color="#444", lw=0.8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    save_sup("S3_healthy_bias")


if __name__ == "__main__":
    print(f"[start] writing figures to {FIG}", flush=True)
    f1(); f2(); f3(); f4(); f5(); f6_three_tier_cube(); f7_roi()
    s1_reliability(); s2_three_tier_pixel(); s3_healthy_bias()
    print("DONE", flush=True)
