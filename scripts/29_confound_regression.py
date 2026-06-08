"""
Phase 3.11 - Image-level baseline confound regression.

Question: how much of the AUC = 0.90 ceiling (script 25b RF) depends on
between-image variation (image-level baseline / batch / illumination drift)
vs within-image kernel-level features?

Method:
  1. For each pixel x_i belonging to image I, subtract image-mean spectrum:
       x_i_residual = x_i - mean(X[image==I])
  2. The image-mean carries all between-image drift. The residual is pure
     within-image variation.
  3. Run Ridge and RF on the residual under same GroupKFold setup.

Comparison:
  Raw spectra:       Ridge R^2 = 0.337 / AUC = 0.80; RF R^2 = 0.466 / AUC = 0.90
  Residual spectra:  Ridge R^2 = ? ;                 RF R^2 = ?

If residual R^2/AUC collapse vs raw -> AUC = 0.90 is mostly image-level baseline
discrimination, not kernel-level AFB1 features. The paper should report this as
a caveat: "the model is effectively learning batch/image identity that happens
to correlate with the per-image ppb label".

If residual R^2/AUC are preserved -> within-image kernel features carry the
signal, image-level baseline is incidental.
"""
import os, sys, time, warnings
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
from pistachio_io import EU_AFB1_THRESHOLD_PPB
warnings.filterwarnings("ignore")

RES = _os.environ.get("PISTACHIO_RES", "results")
LOW_CONC_THRESHOLD = 2.0

X_raw = np.load(os.path.join(RES, "pistachio_spectra.npy"))
meta = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
y = meta["AFB1_ppb"].values
zones = meta["zone"].values
images = meta["image"].values
print(f"[load] {X_raw.shape[0]} pixels x {X_raw.shape[1]} bands", flush=True)
X = _sg(X_raw.astype(float), 2)

# --- subtract image-mean spectrum ---------------------------------------
print("[confound] computing image-mean spectra and residuals", flush=True)
unique_imgs = np.unique(images)
img_means = {}
for img in unique_imgs:
    sel = images == img
    img_means[img] = X[sel].mean(axis=0)
img_mean_per_px = np.array([img_means[img] for img in images])
X_resid = X - img_mean_per_px
print(f"[confound] X_resid mean abs = {np.abs(X_resid).mean():.4g} "
      f"(raw X mean abs = {np.abs(X).mean():.4g})", flush=True)


def auc_at(y_true, pred, thr):
    truth = (y_true > thr).astype(int)
    return roc_auc_score(truth, pred) if 0 < truth.sum() < len(truth) else np.nan


def metrics(y_true, pred):
    resid = y_true - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y_true - y_true.mean())**2)
    rmse = np.sqrt(np.mean(resid**2))
    r = pearsonr(y_true, pred)[0]
    return dict(R2=r2, RMSE=rmse, r=r,
                AUC_8ppb=auc_at(y_true, pred, EU_AFB1_THRESHOLD_PPB),
                AUC_2ppb=auc_at(y_true, pred, LOW_CONC_THRESHOLD))


def run_ridge(Xuse, tag):
    gkf = GroupKFold(n_splits=5)
    pred = np.full(len(y), np.nan)
    for fold_i, (tr, te) in enumerate(gkf.split(Xuse, y, groups=images)):
        pipe = Pipeline([("scale", StandardScaler()),
                         ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
        pipe.fit(Xuse[tr], y[tr])
        pred[te] = pipe.predict(Xuse[te])
        print(f"  [{tag}-Ridge] fold {fold_i+1}/5 done", flush=True)
    return pred


def run_rf(Xuse, tag):
    gkf = GroupKFold(n_splits=5)
    pred = np.full(len(y), np.nan)
    t0 = time.time()
    for fold_i, (tr, te) in enumerate(gkf.split(Xuse, y, groups=images)):
        rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                                   min_samples_leaf=10, max_features="sqrt",
                                   n_jobs=4, random_state=42)
        rf.fit(Xuse[tr], y[tr])
        pred[te] = rf.predict(Xuse[te])
        print(f"  [{tag}-RF] fold {fold_i+1}/5 @ {time.time()-t0:.1f}s", flush=True)
    return pred


rows = []

for Xuse, src_tag in [(X, "raw"), (X_resid, "residual")]:
    print(f"\n=== Ridge on {src_tag} ===", flush=True)
    pred = run_ridge(Xuse, src_tag)
    m = metrics(y, pred)
    row = dict(input=src_tag, model="Ridge", **m)
    for z in ("low","mid","high"):
        sel = zones == z
        row[f"r_{z}"] = metrics(y[sel], pred[sel])["r"] if sel.sum() >= 5 else np.nan
    rows.append(row)
    print(f"  ==> R^2={m['R2']:+.3f} AUC@8={m['AUC_8ppb']:.3f} AUC@2={m['AUC_2ppb']:.3f}",
          flush=True)

    print(f"\n=== RF on {src_tag} ===", flush=True)
    pred = run_rf(Xuse, src_tag)
    m = metrics(y, pred)
    row = dict(input=src_tag, model="RF", **m)
    for z in ("low","mid","high"):
        sel = zones == z
        row[f"r_{z}"] = metrics(y[sel], pred[sel])["r"] if sel.sum() >= 5 else np.nan
    rows.append(row)
    print(f"  ==> R^2={m['R2']:+.3f} AUC@8={m['AUC_8ppb']:.3f} AUC@2={m['AUC_2ppb']:.3f}",
          flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "29_confound_regression.tsv"), sep="\t", index=False)

md = [
    "# Phase 3.11 - Confound regression: image-level baseline vs kernel features",
    "",
    "**Question**: does the AUC = 0.90 ceiling on raw spectra depend on "
    "image-level baseline differences (batch / illumination drift), or on "
    "within-image kernel-level features? Subtract image-mean spectrum per pixel "
    "and rerun.",
    "",
    f"**Samples**: {X.shape[0]} pixels, {meta['image'].nunique()} images.",
    "",
    "## Headline",
    "",
    "| input | model | R^2 | r | RMSE | AUC@8 | AUC@2 | r_low | r_mid | r_high |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for _, r_ in df.iterrows():
    md.append(f"| {r_['input']} | {r_['model']} | {r_['R2']:+.3f} | {r_['r']:+.3f} | "
              f"{r_['RMSE']:.2f} | {r_['AUC_8ppb']:.3f} | {r_['AUC_2ppb']:.3f} | "
              f"{r_['r_low']:+.3f} | {r_['r_mid']:+.3f} | {r_['r_high']:+.3f} |")

# delta interpretation
raw_rf = df[(df["input"]=="raw") & (df["model"]=="RF")].iloc[0]
res_rf = df[(df["input"]=="residual") & (df["model"]=="RF")].iloc[0]
raw_rg = df[(df["input"]=="raw") & (df["model"]=="Ridge")].iloc[0]
res_rg = df[(df["input"]=="residual") & (df["model"]=="Ridge")].iloc[0]

md += [
    "",
    "## Read-out",
    "",
    f"- Ridge AUC@8: raw {raw_rg['AUC_8ppb']:.3f} -> residual {res_rg['AUC_8ppb']:.3f} "
    f"(Delta = {res_rg['AUC_8ppb']-raw_rg['AUC_8ppb']:+.3f})",
    f"- RF AUC@8:    raw {raw_rf['AUC_8ppb']:.3f} -> residual {res_rf['AUC_8ppb']:.3f} "
    f"(Delta = {res_rf['AUC_8ppb']-raw_rf['AUC_8ppb']:+.3f})",
    "",
    "**Interpretation by RF AUC delta**:",
    "- If residual AUC@8 stays > 0.80 (Delta > -0.10): within-image kernel features carry the signal; image-level baseline is incidental.",
    "- If residual AUC@8 drops to 0.5-0.7 (Delta -0.20 to -0.40): a significant portion of the 0.90 AUC was image-level baseline / batch effect, not kernel-level AFB1 features.",
    "- If residual AUC@8 collapses to <=0.5 (Delta <= -0.40): the model was effectively learning image identity. Must report this as the primary caveat.",
    "",
    "## Outputs",
    "- `results/29_confound_regression.tsv`",
]
with open(os.path.join(RES, "29_confound_regression.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/29_confound_regression.{{tsv,md}}", flush=True)
print("DONE", flush=True)
