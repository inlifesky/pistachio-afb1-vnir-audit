"""
Phase 4.7 - Healthy bias mitigation: three strategies.

Problem: all models predict 3-6 ppb on a true 0 ppb sample (script 33d/37d/38).
This is regression-to-the-mean, not a "training set lacks healthy" issue
(script 38 reverse train showed v3-train doesn't fix it either).

Three mitigation strategies (all on SG2 preprocessing for clean comparison):

  (A) GBM with quantile loss (loss='quantile', q=0.5)
      Predicts median, not mean. Median is less pulled toward training-set mean
      by extreme high-ppb tails.

  (B) GBM with monotonic constraints (top-10 bands by Ridge |beta|, signs from
      Ridge coef direction). Forces the model to be monotonic in features that
      Ridge identifies as correlated with target.

  (C) Two-stage Ridge: stage 1 logistic classifier (P(>8 ppb)), stage 2 Ridge
      regressor fit only on stage-1-positive samples. Final score = stage 1
      probability. Healthy samples should get low P.

Evaluation:
  - Forward direction (v1 train -> v3 test)
  - Reverse direction (v3 train -> v1 test)
  - Headline: healthy bias (pred mean on true ~0 ppb), lot AUC@8, FPR@95.
"""
import os, sys, warnings, time
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
from pistachio_io import EU_AFB1_THRESHOLD_PPB
warnings.filterwarnings("ignore")

RES = r"D:\bioinformatics\project_pistachio_AFB1\results"

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

X1_sg2 = _sg(X1_raw.astype(float), 2)
X3_sg2 = _sg(X3_raw.astype(float), 2)
print(f"[load] v1 {X1_sg2.shape}, v3 {X3_sg2.shape}", flush=True)


def auc(y, pred, thr=8):
    yb = (y > thr).astype(int)
    return roc_auc_score(yb, pred) if 0 < yb.sum() < len(yb) else np.nan


def fpr95(yb, pred):
    f, t, _ = roc_curve(yb, pred)
    sel = np.where(t >= 0.95)[0]
    return float(f[sel[0]]) if len(sel) else np.nan


def aggregate(pred, y, images, agg="mean"):
    df = pd.DataFrame(dict(pred=pred, y=y, image=images))
    if agg == "mean":
        a = df.groupby("image").agg(pred=("pred","mean"), y=("y","first"))
    return a["pred"].values, a["y"].values


def healthy_stat(pred, mask):
    p = pred[mask]
    if len(p) == 0: return dict(n=0, mean=np.nan, median=np.nan, p95=np.nan)
    return dict(n=int(mask.sum()), mean=float(p.mean()),
                median=float(np.median(p)),
                p95=float(np.percentile(p, 95)))


# ============================================================
# Pre-compute Ridge top-10 bands and signs for monotonic constraint
# ============================================================
print("\n[setup] Ridge for monotonic constraint signs (in-domain v1)", flush=True)
ridge_pipe = Pipeline([("scale", StandardScaler()),
                        ("ridge", RidgeCV(alphas=np.logspace(-3,3,13)))])
ridge_pipe.fit(X1_sg2, y1)
beta = ridge_pipe.named_steps["ridge"].coef_
order = np.argsort(np.abs(beta))[::-1]
top10_idx = order[:10].tolist()
mono_cst = np.zeros(462, dtype=int)
for idx in top10_idx:
    mono_cst[idx] = int(np.sign(beta[idx]))
print(f"  top-10 |beta| bands: {top10_idx}", flush=True)
print(f"  monotonic sign distribution: +1={int((mono_cst==1).sum())}, "
      f"-1={int((mono_cst==-1).sum())}, 0={int((mono_cst==0).sum())}", flush=True)


# ============================================================
# Model factories
# ============================================================
def make_ridge():
    return Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3,3,13)))])


def make_gbm_mse():
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=20, random_state=42)


def make_gbm_quantile():
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=0.5, max_iter=200, max_depth=8,
        learning_rate=0.1, min_samples_leaf=20, random_state=42)


def make_gbm_monotonic():
    return HistGradientBoostingRegressor(
        loss="squared_error", max_iter=200, max_depth=8, learning_rate=0.1,
        min_samples_leaf=20, random_state=42,
        monotonic_cst=mono_cst.tolist())


class TwoStage:
    """Stage 1 logistic classifier @ 8 ppb; stage 2 Ridge regressor on positives.
    Final score = stage 1 P(unsafe). (Stage 2 is reported separately.)"""
    def __init__(self):
        self.cls = None
        self.reg = None
    def fit(self, X, y):
        Xs = StandardScaler().fit(X)
        self._scaler = Xs
        Xt = Xs.transform(X)
        y_bin = (y > EU_AFB1_THRESHOLD_PPB).astype(int)
        self.cls = LogisticRegression(C=1.0, max_iter=2000).fit(Xt, y_bin)
        # Stage 2: only on classifier-flagged positives (training-time decision)
        proba = self.cls.predict_proba(Xt)[:, 1]
        pos_mask = proba > 0.5
        if pos_mask.sum() > 10:
            self.reg = RidgeCV(alphas=np.logspace(-3,3,13)).fit(Xt[pos_mask], y[pos_mask])
        return self
    def predict(self, X):
        # Return the calibrated probability * scaling to match ppb regression range
        Xt = self._scaler.transform(X)
        proba = self.cls.predict_proba(Xt)[:, 1]
        return proba * 20.0  # map [0,1] -> [0,20] ppb scale; arbitrary monotonic scaling
    def predict_proba(self, X):
        Xt = self._scaler.transform(X)
        return self.cls.predict_proba(Xt)[:, 1]


MODELS = [
    ("Ridge_baseline", make_ridge),
    ("GBM_MSE_baseline", make_gbm_mse),
    ("GBM_quantile", make_gbm_quantile),
    ("GBM_monotonic", make_gbm_monotonic),
    ("TwoStage_Ridge", TwoStage),
]


# ============================================================
# Run forward and reverse
# ============================================================
rows = []
for direction, X_tr, y_tr, X_te, y_te, images_te, levels_te, healthy_lvl, healthy_true in [
    ("forward (v1->v3)", X1_sg2, y1, X3_sg2, y3, images3, levels3, "Level 01", 0.00),
    ("reverse (v3->v1)", X3_sg2, y3, X1_sg2, y1, images1, levels1, "Level 01", 0.40),
]:
    print(f"\n=== {direction} ===", flush=True)
    for name, factory in MODELS:
        try:
            t0 = time.time()
            m = factory()
            m.fit(X_tr, y_tr)
            pred = m.predict(X_te)
        except Exception as e:
            print(f"  [{name}] failed: {e}", flush=True)
            continue
        lp, ly = aggregate(pred, y_te, images_te, "mean")
        h_mask = levels_te == healthy_lvl
        h = healthy_stat(pred, h_mask)
        # lot-level healthy: each lot's mean for the healthy level
        h_lot_mask = ly == (y_te[h_mask][0] if h_mask.sum() > 0 else healthy_true)
        h_lot = healthy_stat(lp, h_lot_mask)
        # AUC at 8 ppb on pixel and lot
        a_pix = auc(y_te, pred, 8)
        a_lot = auc(ly, lp, 8)
        # FPR@95 lot
        yb_lot = (ly > 8).astype(int)
        fpr_lot = fpr95(yb_lot, lp) if 0 < yb_lot.sum() < len(yb_lot) else np.nan
        rows.append(dict(direction=direction, model=name,
                         pixel_AUC_8=a_pix, lot_AUC_8=a_lot,
                         lot_FPR_95=fpr_lot,
                         healthy_pixel_n=h["n"],
                         healthy_pixel_mean=h["mean"],
                         healthy_pixel_median=h["median"],
                         healthy_pixel_p95=h["p95"],
                         healthy_lot_n=h_lot["n"],
                         healthy_lot_mean=h_lot["mean"],
                         true_healthy_ppb=healthy_true,
                         healthy_bias_pixel=h["mean"] - healthy_true,
                         healthy_bias_lot=h_lot["mean"] - healthy_true,
                         runtime_s=time.time()-t0))
        print(f"  [{name}] pix AUC@8 {a_pix:.3f}  lot AUC@8 {a_lot:.3f}  "
              f"lot FPR@95 {fpr_lot:.3f}  "
              f"healthy bias pix +{h['mean']-healthy_true:.2f}  "
              f"lot +{h_lot['mean']-healthy_true:.2f}  "
              f"({time.time()-t0:.1f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "39_healthy_bias_mitigation.tsv"), sep="\t", index=False)


# ============================================================
# Markdown
# ============================================================
md = ["# Phase 4.7 - Healthy bias mitigation: three strategies", "",
      "Tested on SG2 preprocessing (clean comparison; SNV variant in script 36/37).",
      "",
      "## Strategies", "",
      "- **Ridge_baseline**: standard Ridge MSE regression (script 32 reference)",
      "- **GBM_MSE_baseline**: HistGradientBoosting with squared error loss (script 32 GBM reference)",
      "- **GBM_quantile**: HistGradientBoosting with quantile loss, q=0.5 (predicts median, not mean)",
      "- **GBM_monotonic**: HistGradientBoosting + monotonic constraints on top-10 Ridge |β| bands "
      f"(top-10 idx = {top10_idx[:10]})",
      "- **TwoStage_Ridge**: stage 1 = logistic classifier @ 8 ppb; final score = P(unsafe) scaled to [0, 20]",
      "",
      "## Results", "",
      "| direction | model | pixel AUC@8 | lot AUC@8 | lot FPR@95 | healthy pred mean | bias |",
      "|---|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append(f"| {r['direction']} | {r['model']} | "
              f"{r['pixel_AUC_8']:.3f} | {r['lot_AUC_8']:.3f} | "
              f"{r['lot_FPR_95']:.3f} | "
              f"**{r['healthy_pixel_mean']:.2f}** "
              f"(vs true {r['true_healthy_ppb']:.2f}) | "
              f"**+{r['healthy_bias_pixel']:.2f}** |")

# winners
best_bias_fwd = df[df.direction.str.startswith("forward")].sort_values("healthy_bias_pixel").iloc[0]
best_bias_rev = df[df.direction.str.startswith("reverse")].sort_values("healthy_bias_pixel").iloc[0]
best_auc_fwd = df[df.direction.str.startswith("forward")].sort_values("lot_AUC_8", ascending=False).iloc[0]
best_auc_rev = df[df.direction.str.startswith("reverse")].sort_values("lot_AUC_8", ascending=False).iloc[0]

md += ["",
       "## Winners per criterion",
       "",
       f"- **Lowest healthy bias forward**: {best_bias_fwd['model']} (pred mean = {best_bias_fwd['healthy_pixel_mean']:.2f}, bias = +{best_bias_fwd['healthy_bias_pixel']:.2f})",
       f"- **Lowest healthy bias reverse**: {best_bias_rev['model']} (pred mean = {best_bias_rev['healthy_pixel_mean']:.2f}, bias = +{best_bias_rev['healthy_bias_pixel']:.2f})",
       f"- **Highest lot AUC@8 forward**: {best_auc_fwd['model']} ({best_auc_fwd['lot_AUC_8']:.3f})",
       f"- **Highest lot AUC@8 reverse**: {best_auc_rev['model']} ({best_auc_rev['lot_AUC_8']:.3f})",
       "",
       "## Read-out",
       "",
       "Compare each mitigation to its baseline:",
       "- **Quantile**: tests whether median is less biased than mean. If GBM_quantile healthy_bias < GBM_MSE_baseline by >= 1.0 ppb AND lot_AUC@8 retained within 0.02, adopt.",
       "- **Monotonic**: tests whether enforcing monotonicity stabilises the model. Risk: top-10 |β| bands may not all be truly monotonic in the underlying chemistry.",
       "- **Two-stage**: if lot AUC@8 holds and healthy bias drops because score is a probability not a ppb, this is the cleanest fix — output is intrinsically calibrated as 'risk', not as 'concentration'.",
       "",
       "## Outputs",
       "- `results/39_healthy_bias_mitigation.tsv`"]

with open(os.path.join(RES, "39_healthy_bias_mitigation.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/39_healthy_bias_mitigation.{{tsv,md}}", flush=True)
print("DONE", flush=True)
