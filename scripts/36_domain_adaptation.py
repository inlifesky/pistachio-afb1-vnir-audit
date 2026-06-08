"""
Phase 4.4 - Domain adaptation via preprocessing variants.

Question: can simple per-image normalization or feature alignment push the
cross-batch (v1->v3) AUC above the current 0.865 pixel-level / 0.935 lot-level?

Variants (each runs Ridge + GBM under v1 GKF and v1->v3):
  V0 baseline    : SG2 only (matches scripts 24-34)
  V1 SNV         : SG2 + Standard Normal Variate (per-pixel mean=0, std=1)
  V2 per-image   : SG2 + subtract image-mean spectrum (test-time too)
  V3 per-img-z   : SG2 + per-image z-score (mean=0, std=1 per image)
  V4 SNV+pimean  : V1 + V2 stacked
  V5 CORAL       : SG2 + Correlation Alignment (align v3 feature cov to v1)

CORAL: A = cov(X_train), B = cov(X_test); transform X_test = X_test @
sqrtm(A) @ sqrtm(inv(B)). Implemented numerically.
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
from pistachio_io import EU_AFB1_THRESHOLD_PPB
warnings.filterwarnings("ignore")

RES = r"D:\bioinformatics\project_pistachio_AFB1\results"

X1_raw = np.load(os.path.join(RES, "pistachio_spectra.npy"))
meta1 = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
y1 = meta1["AFB1_ppb"].values
images1 = meta1["image"].values

X3_raw = np.load(os.path.join(RES, "pistachio_v3_spectra.npy"))
meta3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t")
y3 = meta3["AFB1_ppb"].values
images3 = meta3["image"].values
print(f"[load] v1 {X1_raw.shape}  v3 {X3_raw.shape}", flush=True)

# SG2 once
X1_sg = _sg(X1_raw.astype(float), 2)
X3_sg = _sg(X3_raw.astype(float), 2)


# ============================================================
# Preprocessing variants (stateless ones don't need fit on train)
# ============================================================
def snv(X):
    X = np.asarray(X, float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    return (X - mu) / sd


def per_image_subtract(X, images):
    """For each unique image, subtract its mean spectrum (test-time domain adaptation)."""
    Xout = np.empty_like(X)
    for img in np.unique(images):
        sel = images == img
        Xout[sel] = X[sel] - X[sel].mean(axis=0, keepdims=True)
    return Xout


def per_image_zscore(X, images):
    Xout = np.empty_like(X)
    for img in np.unique(images):
        sel = images == img
        mu = X[sel].mean(axis=0, keepdims=True)
        sd = X[sel].std(axis=0, keepdims=True); sd[sd == 0] = 1.0
        Xout[sel] = (X[sel] - mu) / sd
    return Xout


def coral_align(X_src, X_tgt, eps=1e-3):
    """Align X_tgt covariance to X_src.
    Whiten X_tgt by its own cov, then color by X_src cov."""
    Cs = np.cov(X_src, rowvar=False) + eps * np.eye(X_src.shape[1])
    Ct = np.cov(X_tgt, rowvar=False) + eps * np.eye(X_tgt.shape[1])
    # sqrt and inv-sqrt (real symmetric)
    Cs_sqrt = sqrtm(Cs).real
    Ct_invsqrt = np.linalg.pinv(sqrtm(Ct).real)
    return (X_tgt @ Ct_invsqrt) @ Cs_sqrt


VARIANTS = ["V0_SG2", "V1_SG2_SNV", "V2_SG2_pImgSub", "V3_SG2_pImgZ",
            "V4_SG2_SNV_pImgSub", "V5_SG2_CORAL"]


def transform(variant, X_train_raw_sg, train_images,
              X_test_raw_sg, test_images):
    """Return (X_train, X_test) under the given variant."""
    if variant == "V0_SG2":
        return X_train_raw_sg, X_test_raw_sg
    if variant == "V1_SG2_SNV":
        return snv(X_train_raw_sg), snv(X_test_raw_sg)
    if variant == "V2_SG2_pImgSub":
        return (per_image_subtract(X_train_raw_sg, train_images),
                per_image_subtract(X_test_raw_sg, test_images))
    if variant == "V3_SG2_pImgZ":
        return (per_image_zscore(X_train_raw_sg, train_images),
                per_image_zscore(X_test_raw_sg, test_images))
    if variant == "V4_SG2_SNV_pImgSub":
        X_tr = per_image_subtract(snv(X_train_raw_sg), train_images)
        X_te = per_image_subtract(snv(X_test_raw_sg), test_images)
        return X_tr, X_te
    if variant == "V5_SG2_CORAL":
        # CORAL aligns test feature distribution to train
        return X_train_raw_sg, coral_align(X_train_raw_sg, X_test_raw_sg)
    raise ValueError(variant)


# ============================================================
# Eval helpers
# ============================================================
def auc_at(y_true, pred, thr):
    yb = (y_true > thr).astype(int)
    return roc_auc_score(yb, pred) if 0 < yb.sum() < len(yb) else np.nan


def score(y_true, pred, thresholds=(8, 10)):
    resid = y_true - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y_true - y_true.mean())**2)
    return dict(R2=r2,
                AUC_8=auc_at(y_true, pred, 8),
                AUC_10=auc_at(y_true, pred, 10),
                PR_AUC_8=average_precision_score((y_true>8).astype(int), pred),
                PR_AUC_10=average_precision_score((y_true>10).astype(int), pred),
                r=pearsonr(y_true, pred)[0])


def aggregate_to_lots(pred, y_true, images, agg="mean"):
    df = pd.DataFrame(dict(pred=pred, y=y_true, image=images))
    g = df.groupby("image")
    if agg == "mean":
        agg_df = g.agg(pred=("pred", "mean"), y=("y", "first"))
    elif agg == "p90":
        agg_df = g.agg(pred=("pred", lambda x: np.percentile(x, 90)),
                        y=("y", "first"))
    else:
        raise ValueError(agg)
    return agg_df["pred"].values, agg_df["y"].values


# ============================================================
# Models
# ============================================================
def ridge_pipeline():
    return Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])

def gbm_factory():
    return HistGradientBoostingRegressor(max_iter=200, max_depth=8,
                                          learning_rate=0.1,
                                          min_samples_leaf=20,
                                          random_state=42)


def run_in_domain_gkf(X_use, model_factory, tag):
    gkf = GroupKFold(n_splits=5)
    pred = np.full(len(y1), np.nan)
    for tr, te in gkf.split(X_use, y1, groups=images1):
        m = model_factory()
        m.fit(X_use[tr], y1[tr])
        pred[te] = m.predict(X_use[te])
    return pred


def run_cross_batch(X1_use, X3_use, model_factory, tag):
    m = model_factory()
    m.fit(X1_use, y1)
    return m.predict(X3_use)


rows = []
for variant in VARIANTS:
    print(f"\n=== {variant} ===", flush=True)
    # In-domain v1 GKF (variant applied within-domain too, where applicable)
    # For per-image variants, apply on full v1 then GKF split
    if variant in ("V0_SG2", "V1_SG2_SNV"):
        X1_t = transform(variant, X1_sg, images1, X1_sg, images1)[0]
    elif variant in ("V2_SG2_pImgSub", "V3_SG2_pImgZ"):
        X1_t = transform(variant, X1_sg, images1, X1_sg, images1)[0]
    elif variant == "V4_SG2_SNV_pImgSub":
        X1_t = transform(variant, X1_sg, images1, X1_sg, images1)[0]
    elif variant == "V5_SG2_CORAL":
        X1_t = X1_sg   # in-domain has no v3, CORAL collapses to identity
    else:
        raise ValueError(variant)

    # Cross-batch: also apply variant
    X1_train, X3_test = transform(variant, X1_sg, images1, X3_sg, images3)

    for model_name, factory in [("Ridge", ridge_pipeline), ("GBM", gbm_factory)]:
        t0 = time.time()
        # in-domain
        pred_in = run_in_domain_gkf(X1_t, factory, f"{variant}/{model_name}/in")
        s_in = score(y1, pred_in)
        lp_in, ly_in = aggregate_to_lots(pred_in, y1, images1, agg="mean")
        s_in_lot = score(ly_in, lp_in)

        # cross-batch
        pred_xb = run_cross_batch(X1_train, X3_test, factory,
                                  f"{variant}/{model_name}/xb")
        s_xb = score(y3, pred_xb)
        lp_xb, ly_xb = aggregate_to_lots(pred_xb, y3, images3, agg="mean")
        s_xb_lot = score(ly_xb, lp_xb)

        rows.append(dict(variant=variant, model=model_name,
                         in_pix_R2=s_in["R2"], in_pix_AUC8=s_in["AUC_8"],
                         in_pix_AUC10=s_in["AUC_10"], in_pix_PRAUC8=s_in["PR_AUC_8"],
                         in_lot_AUC8=s_in_lot["AUC_8"],
                         in_lot_AUC10=s_in_lot["AUC_10"],
                         xb_pix_R2=s_xb["R2"], xb_pix_AUC8=s_xb["AUC_8"],
                         xb_pix_AUC10=s_xb["AUC_10"], xb_pix_PRAUC8=s_xb["PR_AUC_8"],
                         xb_lot_AUC8=s_xb_lot["AUC_8"],
                         xb_lot_AUC10=s_xb_lot["AUC_10"]))
        print(f"  [{model_name}] in_pix AUC@8 {s_in['AUC_8']:.3f}  "
              f"in_lot {s_in_lot['AUC_8']:.3f}  "
              f"xb_pix {s_xb['AUC_8']:.3f}  "
              f"xb_lot {s_xb_lot['AUC_8']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "36_domain_adaptation.tsv"), sep="\t", index=False)

md = ["# Phase 4.4 - Domain adaptation via preprocessing variants", "",
      "Test simple normalisation and feature-alignment tricks to push cross-batch "
      "AUC above the SG2 baseline (script 32: pixel 0.865, lot 0.935 at GBM @ 8 ppb).",
      "",
      "Each variant is applied to both v1 train and v3 test under the same model "
      "(Ridge or GBM). In-domain is v1 GroupKFold-by-image; cross-batch is v1->v3.",
      "",
      "| variant | model | in_pix AUC@8 | in_lot AUC@8 | xb_pix AUC@8 | xb_lot AUC@8 | xb_pix AUC@10 | xb_lot AUC@10 |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append(f"| {r['variant']} | {r['model']} | {r['in_pix_AUC8']:.3f} | "
              f"{r['in_lot_AUC8']:.3f} | "
              f"**{r['xb_pix_AUC8']:.3f}** | **{r['xb_lot_AUC8']:.3f}** | "
              f"{r['xb_pix_AUC10']:.3f} | {r['xb_lot_AUC10']:.3f} |")

# best xb_lot AUC@8
best = df.loc[df["xb_lot_AUC8"].idxmax()]
md += ["",
       "## Read-out",
       "",
       f"- **Best cross-batch lot-level AUC@8 = {best['xb_lot_AUC8']:.3f}** "
       f"({best['variant']} + {best['model']}).",
       f"- Baseline V0 (SG2 only): GBM xb_lot AUC@8 = "
       f"{df[(df.variant=='V0_SG2')&(df.model=='GBM')]['xb_lot_AUC8'].iloc[0]:.3f}.",
       f"- Delta from variant choice: {best['xb_lot_AUC8'] - df[(df.variant=='V0_SG2')&(df.model=='GBM')]['xb_lot_AUC8'].iloc[0]:+.3f}.",
       "",
       "Interpretation depends on the table:",
       "- If best variant beats V0 by >= 0.02 lot AUC@8, adopt as paper-2 main pipeline.",
       "- If gain < 0.01, baseline SG2 is already near domain-adaptation ceiling on this dataset.",
       "- If per-image variants (V2/V3/V4) hurt cross-batch, the image-mean signal carries part of the AFB1 information transferable to v3 — confound regression on v3 should not be applied at deployment time.",
       "",
       "## Outputs",
       "- `results/36_domain_adaptation.tsv`"]
with open(os.path.join(RES, "36_domain_adaptation.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/36_domain_adaptation.{{tsv,md}}", flush=True)
print("DONE", flush=True)
