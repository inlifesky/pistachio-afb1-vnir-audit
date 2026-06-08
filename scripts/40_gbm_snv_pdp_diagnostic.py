"""
Phase 4.8 - GBM+SNV collapse diagnostic via partial dependence + prediction distribution.

Two visual diagnostics for paper-2 Discussion:

  (A) Partial dependence plots: for the top-10 Ridge |beta| bands, plot how
      GBM+SG2 vs GBM+SNV predicted ppb varies with the band's feature value.
      If GBM+SNV PDP is flat or sign-flipped, SNV destroyed the band-target
      relationship at the model level (not just for downstream metrics).

  (B) Prediction distribution by true zone: histogram of GBM+SNV cross-batch
      preds in (low, mid, high) zones. If predictions collapse toward training
      mean (i.e., all preds cluster near ~6 ppb regardless of true zone), it
      confirms regression-to-mean is amplified by SNV normalisation.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg, _snv
warnings.filterwarnings("ignore")

RES = r"D:\bioinformatics\project_pistachio_AFB1\results"

X1_raw = np.load(os.path.join(RES, "pistachio_spectra.npy"))
meta1 = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
y1 = meta1["AFB1_ppb"].values
X3_raw = np.load(os.path.join(RES, "pistachio_v3_spectra.npy"))
meta3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t")
y3 = meta3["AFB1_ppb"].values
zones3 = meta3["zone"].values
print(f"[load] v1 {X1_raw.shape}, v3 {X3_raw.shape}", flush=True)

X1_sg2 = _sg(X1_raw.astype(float), 2)
X1_snv = _snv(X1_sg2)
X3_sg2 = _sg(X3_raw.astype(float), 2)
X3_snv = _snv(X3_sg2)

DATA_ROOT = r"D:\bioinformatics\project_pistachio_AFB1\data\pistachio\extracted\Dataset"
text = open(os.path.join(DATA_ROOT, "Level 01", "L01_0001.hdr"), "r",
            encoding="utf-8").read()
s = text.find("wavelength = {") + len("wavelength = {")
e = text.find("}", s)
WL = np.array([float(v.strip()) for v in text[s:e].split(",")])


# top-10 |beta| bands from Ridge on SG2
print("[setup] Ridge for band selection (SG2)", flush=True)
ridge = Pipeline([("scale", StandardScaler()),
                  ("ridge", RidgeCV(alphas=np.logspace(-3,3,13)))])
ridge.fit(X1_sg2, y1)
beta = ridge.named_steps["ridge"].coef_
top10 = np.argsort(np.abs(beta))[::-1][:10].tolist()
print(f"  top-10 bands (|beta|): {top10} = {[f'{WL[i]:.0f}nm' for i in top10]}",
      flush=True)


# Fit both GBM variants
print("\n[fit] GBM+SG2 and GBM+SNV on v1", flush=True)
def mk_gbm():
    return HistGradientBoostingRegressor(
        max_iter=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=20, random_state=42)
gbm_sg2 = mk_gbm().fit(X1_sg2, y1)
gbm_snv = mk_gbm().fit(X1_snv, y1)
print("  done", flush=True)


# (A) Partial dependence for top-10 bands
print("\n[A] partial dependence top-10", flush=True)
fig, axes = plt.subplots(2, 5, figsize=(15, 6), sharey=True)
for k, idx in enumerate(top10):
    ax = axes[k//5, k%5]
    # SG2 PDP
    pd_sg2 = partial_dependence(gbm_sg2, X1_sg2, [idx], grid_resolution=50)
    ax.plot(pd_sg2['grid_values'][0], pd_sg2['average'][0], "b-",
            label="GBM+SG2", linewidth=2)
    # SNV PDP
    pd_snv = partial_dependence(gbm_snv, X1_snv, [idx], grid_resolution=50)
    ax.plot(pd_snv['grid_values'][0], pd_snv['average'][0], "r--",
            label="GBM+SNV", linewidth=2)
    ax.set_title(f"band {idx} = {WL[idx]:.0f}nm\n|beta|={abs(beta[idx]):.2f}",
                 fontsize=9)
    ax.set_xlabel("feature value")
    if k%5==0: ax.set_ylabel("predicted ppb")
    if k==0: ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
fig.suptitle("Partial Dependence Plots: GBM+SG2 vs GBM+SNV (top-10 |beta| bands)\n"
             "Flat/flipped SNV curves indicate SNV destroyed the band-target relationship at the model level",
             fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RES, "40a_pdp_sg2_vs_snv.png"), dpi=140)
plt.close()
print("  written results/40a_pdp_sg2_vs_snv.png", flush=True)


# (B) Prediction distribution by zone, cross-batch v3
print("\n[B] cross-batch prediction distribution on v3", flush=True)
pred_sg2_v3 = gbm_sg2.predict(X3_sg2)
pred_snv_v3 = gbm_snv.predict(X3_snv)

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
COLORS = {"low": "#4C72B0", "mid": "#DD8452", "high": "#C44E52"}
ZONE_LABELS = {"low":"low (<2 ppb)", "mid":"mid (2-8 ppb)", "high":"high (>=8 ppb)"}

for ax, pred, name in [(axes[0], pred_sg2_v3, "GBM+SG2"),
                        (axes[1], pred_snv_v3, "GBM+SNV")]:
    for z in ["low","mid","high"]:
        sel = zones3 == z
        ax.hist(pred[sel], bins=40, alpha=0.5, color=COLORS[z],
                label=f"true {ZONE_LABELS[z]}, n={sel.sum()}, "
                      f"pred mean={pred[sel].mean():.1f}")
    ax.set_title(f"{name} cross-batch (v1 train -> v3 test) predicted ppb by true zone",
                 fontsize=10)
    ax.set_ylabel("count")
    ax.legend(fontsize=8, loc="upper right")
    ax.axvline(8, color="black", ls="--", lw=0.5, label="8 ppb threshold")
    ax.axvline(y1.mean(), color="green", ls=":", lw=0.5,
               label=f"v1 train mean = {y1.mean():.1f}")
    ax.grid(alpha=0.3)
axes[1].set_xlabel("predicted ppb")
plt.tight_layout()
plt.savefig(os.path.join(RES, "40b_pred_distribution_v3.png"), dpi=140)
plt.close()
print("  written results/40b_pred_distribution_v3.png", flush=True)


# Summary statistics
rows = []
for name, pred in [("GBM+SG2", pred_sg2_v3), ("GBM+SNV", pred_snv_v3)]:
    for z in ["low","mid","high"]:
        sel = zones3 == z
        rows.append(dict(model=name, true_zone=z, n=int(sel.sum()),
                         pred_mean=float(pred[sel].mean()),
                         pred_std=float(pred[sel].std()),
                         pred_p25=float(np.percentile(pred[sel], 25)),
                         pred_p75=float(np.percentile(pred[sel], 75)),
                         true_y_mean=float(y3[sel].mean()),
                         true_y_range=f"{y3[sel].min():.2f}-{y3[sel].max():.2f}"))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "40_summary.tsv"), sep="\t", index=False)


# Per-band PDP curves saved (numeric)
pdp_rows = []
for idx in top10:
    pd_sg2 = partial_dependence(gbm_sg2, X1_sg2, [idx], grid_resolution=20)
    pd_snv = partial_dependence(gbm_snv, X1_snv, [idx], grid_resolution=20)
    # Range of PDP variation = how strongly the band drives predictions
    rng_sg2 = float(pd_sg2['average'][0].max() - pd_sg2['average'][0].min())
    rng_snv = float(pd_snv['average'][0].max() - pd_snv['average'][0].min())
    # Sign of correlation between grid and PDP value (does feature push pred up or down?)
    sign_sg2 = float(np.corrcoef(pd_sg2['grid_values'][0], pd_sg2['average'][0])[0,1])
    sign_snv = float(np.corrcoef(pd_snv['grid_values'][0], pd_snv['average'][0])[0,1])
    pdp_rows.append(dict(band_idx=int(idx), wavelength_nm=float(WL[idx]),
                         ridge_beta=float(beta[idx]),
                         GBM_SG2_PDP_range=rng_sg2,
                         GBM_SNV_PDP_range=rng_snv,
                         GBM_SG2_PDP_corr=sign_sg2,
                         GBM_SNV_PDP_corr=sign_snv))
pdp_df = pd.DataFrame(pdp_rows)
pdp_df.to_csv(os.path.join(RES, "40_pdp_summary.tsv"), sep="\t", index=False)

print("\n[summary] PDP variation (range over feature distribution)", flush=True)
print(pdp_df[["band_idx","wavelength_nm","GBM_SG2_PDP_range","GBM_SNV_PDP_range"]].to_string(
    index=False), flush=True)
print("\n[summary] PDP sign (direction of feature-pred correlation)", flush=True)
print(pdp_df[["band_idx","wavelength_nm","ridge_beta",
              "GBM_SG2_PDP_corr","GBM_SNV_PDP_corr"]].to_string(index=False),
      flush=True)


# Markdown
md = ["# Phase 4.8 - GBM+SNV collapse diagnostic via PDP and prediction distribution",
      "",
      "## (A) Partial Dependence Plots (top-10 Ridge |β| bands)",
      "",
      "PDP measures how the model's predicted ppb varies as a single feature is "
      "swept across its observed range, marginalising over all others. Flat or "
      "sign-flipped SNV PDPs would show SNV destroyed the band's information value.",
      "",
      "See `results/40a_pdp_sg2_vs_snv.png`.",
      "",
      "## PDP summary stats",
      "",
      "| band_idx | wavelength_nm | Ridge β | GBM+SG2 PDP range | GBM+SNV PDP range | SG2 corr | SNV corr |",
      "|---|---|---|---|---|---|---|"]
for _, r in pdp_df.iterrows():
    md.append(f"| {int(r['band_idx'])} | {r['wavelength_nm']:.2f} | "
              f"{r['ridge_beta']:+.3f} | "
              f"{r['GBM_SG2_PDP_range']:.3f} | {r['GBM_SNV_PDP_range']:.3f} | "
              f"{r['GBM_SG2_PDP_corr']:+.3f} | {r['GBM_SNV_PDP_corr']:+.3f} |")

mean_range_sg2 = pdp_df["GBM_SG2_PDP_range"].mean()
mean_range_snv = pdp_df["GBM_SNV_PDP_range"].mean()
sign_flips = ((pdp_df["GBM_SG2_PDP_corr"] * pdp_df["GBM_SNV_PDP_corr"]) < 0).sum()

md += ["",
       f"- Mean PDP range SG2 = {mean_range_sg2:.3f} ppb",
       f"- Mean PDP range SNV = {mean_range_snv:.3f} ppb",
       f"- Bands where PDP sign flips between SG2 and SNV: {int(sign_flips)} / 10",
       "",
       "## (B) Cross-batch prediction distribution on v3 by true zone",
       "",
       "If GBM+SNV preds collapse near training mean regardless of true zone, "
       "regression-to-mean is amplified by SNV. See `results/40b_pred_distribution_v3.png`.",
       "",
       "| model | true zone | n | pred mean | pred std | true y mean | true y range |",
       "|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append(f"| {r['model']} | {r['true_zone']} | {r['n']} | "
              f"{r['pred_mean']:.2f} | {r['pred_std']:.2f} | "
              f"{r['true_y_mean']:.2f} | {r['true_y_range']} |")

md += ["",
       "## Outputs",
       "- `results/40a_pdp_sg2_vs_snv.png` - PDP curves comparison",
       "- `results/40b_pred_distribution_v3.png` - pred distribution by zone",
       "- `results/40_summary.tsv`",
       "- `results/40_pdp_summary.tsv`"]

with open(os.path.join(RES, "40_pdp_diagnostic.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/40_pdp_diagnostic.{{md,tsv}} + 2 png", flush=True)
print("DONE", flush=True)
