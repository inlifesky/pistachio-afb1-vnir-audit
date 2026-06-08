"""
Phase A.2 - Image-mean baseline: test whether Ridge+SG2 is essentially an
            image-mean predictor.

ChatGPT review flagged that our 'image-mean confound diagnostic' (script 29)
is ambiguous: since AFB1 labels are constant within each cube, residual-collapse
could mean either (a) Ridge learned a confound, or (b) the residual operation
removes the only signal available to a linear model. Both readings predict the
same residual collapse.

The decisive test is a baseline that USES image-mean spectrum AS the feature:

  per-cube mean spectrum (462 bands)  -> SG2  -> Ridge  -> ppb

If this baseline's in-domain AUC roughly matches Ridge+SG2 (full-pixel) and
its cross-batch AUC also collapses, we have direct evidence that Ridge+SG2's
performance is achievable from image-mean alone -- AND that image-mean does
not transfer across batches. That converts the 'confound diagnostic' into a
defensible 'transfer-risk diagnostic'.

We also fit the same baseline under SG2+SNV to show whether SNV's cross-batch
rescue survives at the image-mean level.

Inputs
------
- results/pistachio_spectra.npy        (17100, 462) v1 pixel spectra
- results/pistachio_meta.tsv           image, AFB1_ppb per pixel
- results/pistachio_v3_spectra.npy     (15600, 462) v3 pixel spectra
- results/pistachio_v3_meta.tsv

Outputs
-------
- results/42_image_mean_baseline.tsv     model x prep x scope x metric
- results/42_image_mean_baseline.md      narrative table
"""
import os, sys
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg

RES = _os.environ.get("PISTACHIO_RES", "results")
SEED = 42

# ---- load ----
X1 = np.load(os.path.join(RES, "pistachio_spectra.npy"))
m1 = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
X3 = np.load(os.path.join(RES, "pistachio_v3_spectra.npy"))
m3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t")
print(f"[load] v1 pixels {X1.shape}  v3 pixels {X3.shape}", flush=True)

def cube_means(X, meta):
    """Return (n_cube, n_band) array of per-cube mean spectra and aligned ppb."""
    df = meta.copy()
    df["row"] = np.arange(len(df))
    out_specs, out_ppb, out_img = [], [], []
    for img, g in df.groupby("image", sort=True):
        out_specs.append(X[g["row"].values].mean(axis=0))
        out_ppb.append(g["AFB1_ppb"].iloc[0])
        out_img.append(img)
    return np.vstack(out_specs), np.array(out_ppb), np.array(out_img)

M1, y1_lot, I1 = cube_means(X1, m1)
M3, y3_lot, I3 = cube_means(X3, m3)
print(f"[cube] v1 lots {M1.shape}  v3 lots {M3.shape}", flush=True)

def snv(X):
    X = np.asarray(X, float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    return (X - mu) / sd

def ridge():
    return Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])

def auc_at(y, p, thr):
    yb = (y >= thr).astype(int)
    if yb.sum() == 0 or yb.sum() == len(yb):
        return np.nan
    return roc_auc_score(yb, p)

def prauc_at(y, p, thr):
    yb = (y >= thr).astype(int)
    if yb.sum() == 0:
        return np.nan
    return average_precision_score(yb, p)

PIPES = {
    "imgmean_SG2":     lambda X: _sg(X, 2),
    "imgmean_SG2_SNV": lambda X: snv(_sg(X, 2)),
}

rows = []
for prep_name, prep in PIPES.items():
    M1t = prep(M1); M3t = prep(M3)
    # in-domain: per-cube features now -> LeaveOneOut over cubes (n=57)
    # use LOO for honest in-domain since each cube IS one sample now
    loo = LeaveOneGroupOut()
    pred_in = np.full(len(y1_lot), np.nan)
    for tr, te in loo.split(M1t, y1_lot, groups=I1):
        r = ridge(); r.fit(M1t[tr], y1_lot[tr]); pred_in[te] = r.predict(M1t[te])
    # cross-batch: train on all v1 cubes, predict v3 cubes
    r = ridge(); r.fit(M1t, y1_lot)
    pred_xb = r.predict(M3t)

    for thr in [8, 10, 15]:
        rows.append(dict(model="Ridge", prep=prep_name, scope="in-domain (LOO)",
                          n=len(y1_lot), n_pos=int((y1_lot >= thr).sum()),
                          threshold=thr,
                          AUC=auc_at(y1_lot, pred_in, thr),
                          PRAUC=prauc_at(y1_lot, pred_in, thr)))
        rows.append(dict(model="Ridge", prep=prep_name, scope="cross-batch (v1->v3)",
                          n=len(y3_lot), n_pos=int((y3_lot >= thr).sum()),
                          threshold=thr,
                          AUC=auc_at(y3_lot, pred_xb, thr),
                          PRAUC=prauc_at(y3_lot, pred_xb, thr)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "42_image_mean_baseline.tsv"), sep="\t", index=False)

# ---- comparison vs full-pixel pipeline ----
# Read script 36 numbers for the same model x prep x scope @ 8 ppb
ref36 = pd.read_csv(os.path.join(RES, "36_domain_adaptation.tsv"), sep="\t")
def get36(variant, model, col):
    row = ref36[(ref36.variant == variant) & (ref36.model == model)]
    return float(row[col].iloc[0]) if len(row) else np.nan

compare_rows = []
for prep_name, variant in [("imgmean_SG2", "V0_SG2"),
                            ("imgmean_SG2_SNV", "V1_SG2_SNV")]:
    full_in  = get36(variant, "Ridge", "in_lot_AUC8")
    full_xb  = get36(variant, "Ridge", "xb_lot_AUC8")
    mean_in  = df[(df.prep == prep_name) & (df.scope == "in-domain (LOO)")
                  & (df.threshold == 8)]["AUC"].iloc[0]
    mean_xb  = df[(df.prep == prep_name) & (df.scope == "cross-batch (v1->v3)")
                  & (df.threshold == 8)]["AUC"].iloc[0]
    compare_rows.append(dict(
        prep=prep_name,
        full_pixel_indomain=full_in, image_mean_indomain=mean_in,
        diff_indomain=mean_in - full_in,
        full_pixel_xbatch=full_xb, image_mean_xbatch=mean_xb,
        diff_xbatch=mean_xb - full_xb,
    ))
cmp_df = pd.DataFrame(compare_rows)
cmp_df.to_csv(os.path.join(RES, "42_compare_vs_full_pixel.tsv"),
              sep="\t", index=False)

# ---- markdown ----
md = ["# Phase A.2 — Image-mean baseline (transfer-risk diagnostic)", "",
      "Pred = ridge(SG2 or SG2+SNV applied to per-cube mean spectrum). "
      "If this baseline reproduces the full-pixel Ridge result, then Ridge+SG2 "
      "is effectively a function of image-mean only — confirming the residual "
      "diagnostic of script 29 as a **transfer-risk diagnostic** rather than a "
      "spurious-feature diagnostic.",
      "",
      "## Image-mean Ridge at three thresholds",
      "",
      "| prep | scope | thr | n / n_pos | AUC | PR-AUC |",
      "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append(f"| {r['prep']} | {r['scope']} | {r['threshold']} | "
              f"{int(r['n'])}/{int(r['n_pos'])} | "
              f"{r['AUC']:.3f} | {r['PRAUC']:.3f} |")

md += ["",
       "## Direct comparison vs full-pixel Ridge (from script 36) at 8 ppb",
       "",
       "| prep | full-pixel in-domain lot AUC | image-mean in-domain LOO AUC | Δ | full-pixel cross-batch lot AUC | image-mean cross-batch AUC | Δ |",
       "|---|---|---|---|---|---|---|"]
for _, r in cmp_df.iterrows():
    md.append(f"| {r['prep']} | {r['full_pixel_indomain']:.3f} | "
              f"{r['image_mean_indomain']:.3f} | "
              f"{r['diff_indomain']:+.3f} | "
              f"{r['full_pixel_xbatch']:.3f} | {r['image_mean_xbatch']:.3f} | "
              f"{r['diff_xbatch']:+.3f} |")

md += ["",
       "## Interpretation guide",
       "",
       "- **If image-mean baseline matches full-pixel Ridge in-domain AND collapses "
       "cross-batch** (under SG2): Ridge+SG2 is essentially an image-mean predictor, "
       "and image-mean does not survive batch-level baseline shifts. This rebrands "
       "the 'confound diagnostic' as a 'transfer-risk diagnostic' (paper §3.3/§4.1).",
       "- **If image-mean baseline collapses in-domain too**: Ridge+SG2 must be "
       "exploiting more than per-cube mean (e.g. pixel-level texture or spatial "
       "variance), and the residual-collapse interpretation needs further work.",
       "- **If image-mean baseline transfers under SNV but full-pixel SNV transfers "
       "even better**: SNV not only removes per-image baseline but also extracts "
       "within-image shape information that helps.",
       "",
       "## Outputs",
       "- `results/42_image_mean_baseline.tsv`",
       "- `results/42_compare_vs_full_pixel.tsv`",
       ]
with open(os.path.join(RES, "42_image_mean_baseline.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("[write] results/42_image_mean_baseline.{tsv,md}", flush=True)
print("DONE", flush=True)
