"""
Phase 4.6 - Reverse cross-batch (v3 train -> v1 test) + GBM+SNV collapse diagnostic.

Two coupled questions:

  Q1 (HEALTHY BIAS): v1 has no true 0 ppb sample (min 0.40 ppb). Models trained
      on v1 systematically over-predict on true 0 ppb (script 37: Ridge+SNV mean
      pred on v3 Level 01 = 6.08 ppb). If we instead train on v3 (which has
      Level 01 = 0 ppb), does the bias disappear when predicting on v1's lowest
      level (Level 01 = 0.40 ppb)?

  Q2 (GBM+SNV MECHANISM): GBM+SNV cross-batch lot AUC@8 collapses to 0.289 in
      v1->v3 direction (script 36). Is this an intrinsic SNV-GBM incompatibility,
      or an artifact of v1's narrower ppb range? Test reverse direction. Also
      check feature importance shift SG2 vs SNV.

Setup:
  Train: v3 (15,600 px, 52 imgs, ppb 0.00-114.67)
  Test:  v1 (17,100 px, 57 imgs, ppb 0.40-33.17)
  Models: Ridge+SG2, Ridge+SNV, GBM+SG2, GBM+SNV

Outputs:
  results/38_reverse_cross_batch.tsv (metrics)
  results/38_reverse_low_level_preds.tsv (predicted ppb per v1 level)
  results/38_gbm_feature_importance_sg2_vs_snv.tsv
  results/38_pred_v1_*.npy (saved preds for further use)
"""
import os, sys, warnings
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg, _snv
from pistachio_io import EU_AFB1_THRESHOLD_PPB
warnings.filterwarnings("ignore")

RES = _os.environ.get("PISTACHIO_RES", "results")

X1_raw = np.load(os.path.join(RES, "pistachio_spectra.npy"))
meta1 = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
y1 = meta1["AFB1_ppb"].values
images1 = meta1["image"].values
levels1 = meta1["level"].values

X3_raw = np.load(os.path.join(RES, "pistachio_v3_spectra.npy"))
meta3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t")
y3 = meta3["AFB1_ppb"].values
images3 = meta3["image"].values
levels3 = meta3["level"].values
print(f"[load] v1 train range 0.40-33.17 ppb, v3 train range 0.00-114.67", flush=True)
print(f"[load] v1 {X1_raw.shape}, v3 {X3_raw.shape}", flush=True)


def prep(X, mode):
    X = _sg(X.astype(float), 2)
    if mode == "SG2":
        return X
    if mode == "SG2_SNV":
        return _snv(X)
    raise ValueError(mode)


def auc_at(y, pred, thr):
    yb = (y > thr).astype(int)
    return roc_auc_score(yb, pred) if 0 < yb.sum() < len(yb) else np.nan


def pr_at(y, pred, thr):
    yb = (y > thr).astype(int)
    return average_precision_score(yb, pred) if 0 < yb.sum() < len(yb) else np.nan


def recall95_fpr(yb, pred):
    fpr, tpr, _ = roc_curve(yb, pred)
    sel = np.where(tpr >= 0.95)[0]
    return float(fpr[sel[0]]) if len(sel) else np.nan


def aggregate_lots(pred, y, images, agg="mean"):
    df = pd.DataFrame(dict(pred=pred, y=y, image=images))
    if agg == "mean":
        a = df.groupby("image").agg(pred=("pred", "mean"), y=("y", "first"))
    return a["pred"].values, a["y"].values


# ============================================================
# Q1: Reverse cross-batch (v3 train -> v1 test)
# ============================================================
print("\n=== Q1: reverse cross-batch v3 -> v1 ===", flush=True)
preds_v1 = {}
rows = []
for prep_name in ["SG2", "SG2_SNV"]:
    X3_p = prep(X3_raw, prep_name)
    X1_p = prep(X1_raw, prep_name)

    for model_name, factory in [
        ("Ridge", lambda: Pipeline([("scale", StandardScaler()),
                                     ("ridge", RidgeCV(alphas=np.logspace(-3,3,13)))])),
        ("GBM", lambda: HistGradientBoostingRegressor(
                            max_iter=200, max_depth=8, learning_rate=0.1,
                            min_samples_leaf=20, random_state=42))]:
        tag = f"{model_name}_{prep_name}"
        m = factory()
        m.fit(X3_p, y3)
        pred1 = m.predict(X1_p)
        preds_v1[tag] = pred1

        # pixel and lot metrics on v1 target
        lp, ly = aggregate_lots(pred1, y1, images1, "mean")
        row = dict(direction="v3->v1", model=model_name, prep=prep_name)
        for level_name, yb_use, pred_use, n_pos in [
            ("pixel@8", (y1>8).astype(int), pred1, int((y1>8).sum())),
            ("lot@8",   (ly>8).astype(int), lp, int((ly>8).sum())),
            ("pixel@10",(y1>10).astype(int), pred1, int((y1>10).sum())),
            ("lot@10",  (ly>10).astype(int), lp, int((ly>10).sum())),
        ]:
            if yb_use.sum() == 0 or yb_use.sum() == len(yb_use):
                continue
            row[f"AUC_{level_name}"] = roc_auc_score(yb_use, pred_use)
            row[f"PRAUC_{level_name}"] = average_precision_score(yb_use, pred_use)
            row[f"FPR95_{level_name}"] = recall95_fpr(yb_use, pred_use)
        # healthy-equivalent stats on v1 Level 01 (true 0.40 ppb, n=900)
        sel_low = levels1 == "Level 01"
        ph = pred1[sel_low]
        row["v1_Level01_n"] = int(sel_low.sum())
        row["v1_Level01_true_ppb"] = float(y1[sel_low][0]) if sel_low.sum() else np.nan
        row["v1_Level01_pred_mean"] = float(ph.mean())
        row["v1_Level01_pred_median"] = float(np.median(ph))
        row["v1_Level01_pred_p95"] = float(np.percentile(ph, 95))
        rows.append(row)
        print(f"  [{tag}] pixel AUC@8 {row.get('AUC_pixel@8', np.nan):.3f}  "
              f"lot AUC@8 {row.get('AUC_lot@8', np.nan):.3f}  "
              f"L01 pred mean {row['v1_Level01_pred_mean']:.2f}", flush=True)
        # save pred vector
        np.save(os.path.join(RES, f"38_pred_v1_{tag}.npy"), pred1)

df_rev = pd.DataFrame(rows)
df_rev.to_csv(os.path.join(RES, "38_reverse_cross_batch.tsv"), sep="\t", index=False)

# ============================================================
# Q1 supplementary: forward direction Level 01 prediction
# ============================================================
print("\n=== Q1 supp: forward direction v1 -> v3 Level 01 ===", flush=True)
fwd_rows = []
for prep_name in ["SG2", "SG2_SNV"]:
    X1_p = prep(X1_raw, prep_name)
    X3_p = prep(X3_raw, prep_name)
    for model_name, factory in [
        ("Ridge", lambda: Pipeline([("scale", StandardScaler()),
                                     ("ridge", RidgeCV(alphas=np.logspace(-3,3,13)))])),
        ("GBM", lambda: HistGradientBoostingRegressor(
                            max_iter=200, max_depth=8, learning_rate=0.1,
                            min_samples_leaf=20, random_state=42))]:
        tag = f"{model_name}_{prep_name}"
        m = factory()
        m.fit(X1_p, y1)
        pred3 = m.predict(X3_p)
        sel_h = levels3 == "Level 01"  # v3 healthy
        ph = pred3[sel_h]
        fwd_rows.append(dict(direction="v1->v3", model=model_name, prep=prep_name,
                              v3_Level01_n=int(sel_h.sum()),
                              v3_Level01_true_ppb=0.00,
                              v3_Level01_pred_mean=float(ph.mean()),
                              v3_Level01_pred_median=float(np.median(ph)),
                              v3_Level01_pred_p95=float(np.percentile(ph, 95))))
        print(f"  [{tag}] v3 healthy Level 01 pred mean {ph.mean():.2f}", flush=True)
df_fwd = pd.DataFrame(fwd_rows)
df_fwd.to_csv(os.path.join(RES, "38_forward_healthy_preds.tsv"), sep="\t", index=False)


# ============================================================
# Q2: GBM+SNV mechanism — feature importance comparison
# ============================================================
print("\n=== Q2: GBM feature importance SG2 vs SNV (in-domain v1) ===", flush=True)
# Use permutation importance on a held-out validation set
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X1_sg2 = prep(X1_raw, "SG2")
X1_snv = prep(X1_raw, "SG2_SNV")

# DATA_ROOT for wavelengths
DATA_ROOT = _os.environ["PISTACHIO_V1_DATA"]  # unzipped Zenodo v1 cubes
hdr_path = os.path.join(DATA_ROOT, "Level 01", "L01_0001.hdr")
text = open(hdr_path, "r", encoding="utf-8").read()
s = text.find("wavelength = {") + len("wavelength = {")
e = text.find("}", s)
WAVELENGTHS = np.array([float(v.strip()) for v in text[s:e].split(",")])

# Split v1 by image to avoid pixel leakage
rng = np.random.default_rng(42)
imgs = np.unique(images1)
rng.shuffle(imgs)
train_imgs = imgs[:int(len(imgs)*0.8)]
test_imgs  = imgs[int(len(imgs)*0.8):]
tr_mask = np.isin(images1, train_imgs)
te_mask = np.isin(images1, test_imgs)
print(f"  permimp split: {tr_mask.sum()} train px / {te_mask.sum()} test px", flush=True)

fi_rows = []
for prep_name, Xp in [("SG2", X1_sg2), ("SG2_SNV", X1_snv)]:
    gbm = HistGradientBoostingRegressor(
        max_iter=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=20, random_state=42)
    gbm.fit(Xp[tr_mask], y1[tr_mask])
    # Permutation importance (top-15 to keep runtime tractable)
    pi = permutation_importance(gbm, Xp[te_mask], y1[te_mask],
                                  n_repeats=5, random_state=42, n_jobs=4)
    order = np.argsort(pi.importances_mean)[::-1][:15]
    for rank, idx in enumerate(order, 1):
        fi_rows.append(dict(prep=prep_name, rank=rank,
                            band_idx=int(idx),
                            wavelength_nm=float(WAVELENGTHS[idx]),
                            importance_mean=float(pi.importances_mean[idx]),
                            importance_std=float(pi.importances_std[idx])))
df_fi = pd.DataFrame(fi_rows)
df_fi.to_csv(os.path.join(RES, "38_gbm_feature_importance.tsv"), sep="\t", index=False)

print("  top-5 GBM permimp by prep:", flush=True)
for prep_name in ["SG2", "SG2_SNV"]:
    top5 = df_fi[df_fi["prep"]==prep_name].head(5)
    print(f"    {prep_name}: {[f'{w:.0f}nm' for w in top5['wavelength_nm']]}", flush=True)


# ============================================================
# GBM+SNV reverse direction check (v3 train -> v1 test, already done above)
# ============================================================
gbm_snv_rev = df_rev[(df_rev.model=="GBM") & (df_rev.prep=="SG2_SNV")].iloc[0]
gbm_snv_fwd_in_domain = 0.789  # from script 36 V1 SG2_SNV GBM in_pix
gbm_snv_fwd_xb = 0.338  # from script 36 V1 SG2_SNV GBM xb_pix

# ============================================================
# Markdown
# ============================================================
md = ["# Phase 4.6 - Reverse cross-batch + GBM+SNV diagnostic", "",
      "## Q1: Reverse cross-batch (v3 train -> v1 test) - healthy-bias test", "",
      "v3 includes Level 01 = 0.00 ppb (true healthy, n=600). If training on v3 "
      "fixes the regression-to-mean bias seen in script 37 (Ridge+SNV mean pred on "
      "true 0 ppb = 6.08), then v1 Level 01 (0.40 ppb, n=900) predictions should be "
      "near 0.40, not at the dataset-mean (~6 ppb).", "",
      "### Cross-batch performance (v3 -> v1)", "",
      "| model | prep | pixel AUC@8 | lot AUC@8 | pixel FPR@95 | lot FPR@95 |",
      "|---|---|---|---|---|---|"]
for _, r in df_rev.iterrows():
    md.append(f"| {r['model']} | {r['prep']} | "
              f"{r.get('AUC_pixel@8', np.nan):.3f} | "
              f"{r.get('AUC_lot@8', np.nan):.3f} | "
              f"{r.get('FPR95_pixel@8', np.nan):.3f} | "
              f"{r.get('FPR95_lot@8', np.nan):.3f} |")

md += ["",
       "### Healthy-bias comparison (true low-ppb level predictions)", "",
       "**Reverse direction (v3 train -> v1 test) — v1 Level 01 = 0.40 ppb true:**",
       "",
       "| model | prep | pred mean | pred median | pred p95 |",
       "|---|---|---|---|---|"]
for _, r in df_rev.iterrows():
    md.append(f"| {r['model']} | {r['prep']} | "
              f"{r['v1_Level01_pred_mean']:.2f} | "
              f"{r['v1_Level01_pred_median']:.2f} | "
              f"{r['v1_Level01_pred_p95']:.2f} |")
md += ["",
       "**Forward direction (v1 train -> v3 test) — v3 Level 01 = 0.00 ppb true:**",
       "",
       "| model | prep | pred mean | pred median | pred p95 |",
       "|---|---|---|---|---|"]
for _, r in df_fwd.iterrows():
    md.append(f"| {r['model']} | {r['prep']} | "
              f"{r['v3_Level01_pred_mean']:.2f} | "
              f"{r['v3_Level01_pred_median']:.2f} | "
              f"{r['v3_Level01_pred_p95']:.2f} |")

md += ["",
       "## Q2: GBM+SNV collapse mechanism", "",
       "### Bidirectional collapse evidence", "",
       f"- Forward (v1 train -> v3 test) GBM+SNV: in-domain pixel AUC@8 = {gbm_snv_fwd_in_domain:.3f}, cross-batch pixel AUC@8 = {gbm_snv_fwd_xb:.3f} (script 36)",
       f"- Reverse (v3 train -> v1 test) GBM+SNV: cross-batch pixel AUC@8 = {gbm_snv_rev.get('AUC_pixel@8', np.nan):.3f}, lot AUC@8 = {gbm_snv_rev.get('AUC_lot@8', np.nan):.3f}",
       "",
       "Interpretation: if reverse direction GBM+SNV is also below chance, the collapse is an intrinsic SNV-GBM incompatibility (model class issue). If reverse is fine, it's a v1-train-specific overfit.",
       "",
       "### GBM permutation importance top-5 bands per preprocessing (in-domain v1 80/20 image split)", "",
       "| prep | rank | band_idx | wavelength_nm | importance_mean |",
       "|---|---|---|---|---|"]
for _, r in df_fi[df_fi["rank"]<=10].iterrows():
    md.append(f"| {r['prep']} | {r['rank']} | {r['band_idx']} | "
              f"{r['wavelength_nm']:.2f} | {r['importance_mean']:.4f} |")

# overlap of top-5 between SG2 and SG2_SNV
top5_sg2 = set(df_fi[(df_fi.prep=="SG2") & (df_fi["rank"]<=5)]["band_idx"])
top5_snv = set(df_fi[(df_fi.prep=="SG2_SNV") & (df_fi["rank"]<=5)]["band_idx"])
overlap = top5_sg2 & top5_snv

md += ["",
       f"- GBM SG2 top-5 band_idx: {sorted(top5_sg2)}",
       f"- GBM SG2_SNV top-5 band_idx: {sorted(top5_snv)}",
       f"- **Overlap: {len(overlap)} / 5 bands** ({sorted(overlap)})",
       "",
       "If overlap is small (0-1), SNV reshapes which bands GBM uses entirely — GBM under SNV is learning a different problem. If overlap is high, SNV breaks the relationship between feature values and target despite using similar bands.",
       "",
       "## Outputs",
       "- `results/38_reverse_cross_batch.tsv` - full reverse-direction metrics",
       "- `results/38_forward_healthy_preds.tsv` - forward healthy predictions",
       "- `results/38_gbm_feature_importance.tsv` - GBM permimp SG2 vs SNV",
       "- `results/38_pred_v1_*.npy` - saved pred vectors"]

with open(os.path.join(RES, "38_reverse_cross_batch.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/38_reverse_cross_batch.{{tsv,md}} + 4 derived files", flush=True)
print("DONE", flush=True)
