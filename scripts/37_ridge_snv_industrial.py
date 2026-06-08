"""
Phase 4.5 - Full industrial metrics for Ridge+SNV (new cross-batch champion).

Rebuilds the metrics from scripts 33/34 for the Ridge+SNV combination:
  - In-domain v1 GroupKFold preds (pixel + lot-level)
  - Cross-batch v1->v3 preds (pixel + lot-level)
  - Recall@95 operating point: FPR (pixel + lot)
  - Three-tier matrix (NPV>=0.95 + PPV>=0.90)
  - Healthy control on v3 Level 01 (true 0 ppb)

Outputs sensitivity / specificity at the recall@95 op point so script 35 ROI
can be updated to use Ridge+SNV cross-batch numbers.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                              confusion_matrix)
from scipy.stats import pearsonr
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg, _snv
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
levels3 = meta3["level"].values
print(f"[load] v1 {X1_raw.shape}  v3 {X3_raw.shape}", flush=True)

# Apply SG2 + SNV pipeline
X1 = _snv(_sg(X1_raw.astype(float), 2))
X3 = _snv(_sg(X3_raw.astype(float), 2))


def fit_ridge_predict(X_tr, y_tr, X_te):
    pipe = Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    pipe.fit(X_tr, y_tr)
    return pipe.predict(X_te)


print("[in-domain] v1 Ridge+SNV GroupKFold(5)", flush=True)
gkf = GroupKFold(n_splits=5)
pred_in = np.full(len(y1), np.nan)
for fold_i, (tr, te) in enumerate(gkf.split(X1, y1, groups=images1)):
    pred_in[te] = fit_ridge_predict(X1[tr], y1[tr], X1[te])
    print(f"  fold {fold_i+1}/5 done", flush=True)

print("[cross-batch] v1 train -> v3 predict", flush=True)
pred_xb = fit_ridge_predict(X1, y1, X3)

np.save(os.path.join(RES, "37_pred_v1_RidgeSNV.npy"), pred_in)
np.save(os.path.join(RES, "37_pred_v3_RidgeSNV.npy"), pred_xb)


# Aggregation
def aggregate(pred, y, images, agg="mean"):
    df = pd.DataFrame(dict(pred=pred, y=y, image=images))
    if agg == "mean":
        a = df.groupby("image").agg(pred=("pred", "mean"), y=("y", "first"))
    elif agg == "p90":
        a = df.groupby("image").agg(
            pred=("pred", lambda x: np.percentile(x, 90)), y=("y", "first"))
    elif agg == "median":
        a = df.groupby("image").agg(pred=("pred", "median"), y=("y", "first"))
    return a["pred"].values, a["y"].values


def auc_at(y, pred, thr):
    yb = (y > thr).astype(int)
    return roc_auc_score(yb, pred) if 0 < yb.sum() < len(yb) else np.nan


def pr_at(y, pred, thr):
    yb = (y > thr).astype(int)
    return (average_precision_score(yb, pred)
            if 0 < yb.sum() < len(yb) else np.nan)


def class_at_threshold(y_bin, pred, decision_thr):
    pb = (pred > decision_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_bin, pb, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    fpr  = fp / (fp + tn) if (fp + tn) else np.nan
    return dict(n=len(y_bin), n_pos=int(y_bin.sum()),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv, FPR=fpr)


def recall95_op(y_bin, pred, target=0.95):
    fpr, tpr, thr = roc_curve(y_bin, pred)
    sel = np.where(tpr >= target)[0]
    if len(sel) == 0:
        return np.nan, np.nan, np.nan
    i = sel[0]
    return float(thr[i]), float(tpr[i]), float(fpr[i])


# ============================================================
# (a) AUC + PR-AUC at thresholds, pixel & lot
# ============================================================
print("\n[a] AUC/PR-AUC table", flush=True)
THRESHOLDS = [8, 10, 12, 15]
rows_a = []
for scope, y, pred, images in [("in-domain", y1, pred_in, images1),
                                ("cross-batch", y3, pred_xb, images3)]:
    for agg_name in ["pixel", "lot_mean", "lot_p90"]:
        if agg_name == "pixel":
            yu, pu = y, pred
            n = len(yu)
        elif agg_name == "lot_mean":
            pu, yu = aggregate(pred, y, images, "mean")
            n = len(yu)
        elif agg_name == "lot_p90":
            pu, yu = aggregate(pred, y, images, "p90")
            n = len(yu)
        for thr in THRESHOLDS:
            yb = (yu > thr).astype(int)
            if yb.sum() == 0 or yb.sum() == n:
                continue
            rows_a.append(dict(scope=scope, level=agg_name, threshold=thr, n=n,
                               n_pos=int(yb.sum()),
                               base_rate=float(yb.mean()),
                               ROC_AUC=auc_at(yu, pu, thr),
                               PR_AUC=pr_at(yu, pu, thr)))
df_a = pd.DataFrame(rows_a)
df_a.to_csv(os.path.join(RES, "37a_ridgeSNV_auc.tsv"), sep="\t", index=False)


# ============================================================
# (b) Recall@95 operating point
# ============================================================
print("[b] recall@95 FPR table", flush=True)
rows_b = []
for scope, y, pred, images in [("in-domain", y1, pred_in, images1),
                                ("cross-batch", y3, pred_xb, images3)]:
    for agg_name in ["pixel", "lot_mean", "lot_p90"]:
        if agg_name == "pixel":
            yu, pu = y, pred
        elif agg_name == "lot_mean":
            pu, yu = aggregate(pred, y, images, "mean")
        elif agg_name == "lot_p90":
            pu, yu = aggregate(pred, y, images, "p90")
        for thr in THRESHOLDS:
            yb = (yu > thr).astype(int)
            if yb.sum() == 0 or yb.sum() == len(yu):
                continue
            dec, rec, fpr = recall95_op(yb, pu, target=0.95)
            if np.isnan(dec):
                continue
            m = class_at_threshold(yb, pu, dec)
            rows_b.append(dict(scope=scope, level=agg_name, threshold=thr,
                               decision_thr=dec, achieved_recall=rec,
                               FPR_at_recall95=fpr,
                               specificity=m["specificity"],
                               PPV=m["PPV"], NPV=m["NPV"],
                               n_pos=m["n_pos"], n=m["n"]))
df_b = pd.DataFrame(rows_b)
df_b.to_csv(os.path.join(RES, "37b_ridgeSNV_recall95.tsv"), sep="\t", index=False)


# ============================================================
# (c) Three-tier matrix @ 8 ppb (lot-level mean)
# ============================================================
print("[c] three-tier matrix", flush=True)


def tier_thresholds(yb, pred, npv_target=0.95, ppv_target=0.90):
    cands = np.linspace(pred.min(), pred.max(), 300)
    L, H = None, None
    for c in cands:
        m = class_at_threshold(yb, pred, c)
        if not np.isnan(m["NPV"]) and m["NPV"] >= npv_target:
            L = c
    for c in cands[::-1]:
        m = class_at_threshold(yb, pred, c)
        if not np.isnan(m["PPV"]) and m["PPV"] >= ppv_target:
            H = c
    return L, H


rows_c = []
for scope, y, pred, images in [("in-domain", y1, pred_in, images1),
                                ("cross-batch", y3, pred_xb, images3)]:
    for agg_name in ["pixel", "lot_mean"]:
        if agg_name == "pixel":
            yu, pu = y, pred
        else:
            pu, yu = aggregate(pred, y, images, "mean")
        yb = (yu > 8).astype(int)
        if yb.sum() < 5: continue
        L, H = tier_thresholds(yb, pu)
        if L is None or H is None or L > H:
            print(f"  [skip] {scope}/{agg_name}: no valid tier thresholds", flush=True)
            continue
        low = pu < L; high = pu >= H; mid = ~low & ~high
        for name, sel in [("low", low), ("mid", mid), ("high", high)]:
            if sel.sum() == 0:
                rows_c.append(dict(scope=scope, level=agg_name, tier=name,
                                   n=0, pct=0.0, true_unsafe_rate=np.nan,
                                   mean_true_ppb=np.nan, L_thr=L, H_thr=H))
            else:
                rows_c.append(dict(scope=scope, level=agg_name, tier=name,
                                   n=int(sel.sum()),
                                   pct=100*sel.sum()/len(pu),
                                   true_unsafe_rate=float(yb[sel].mean()),
                                   mean_true_ppb=float(yu[sel].mean()),
                                   L_thr=L, H_thr=H))
df_c = pd.DataFrame(rows_c)
df_c.to_csv(os.path.join(RES, "37c_ridgeSNV_tier.tsv"), sep="\t", index=False)


# ============================================================
# (d) Healthy control v3 Level 01
# ============================================================
print("[d] healthy control", flush=True)
healthy3 = (y3 == 0.0)
n_h = int(healthy3.sum())
pred_h = pred_xb[healthy3]
print(f"  v3 Level 01 n={n_h}, pred mean={pred_h.mean():.2f}, "
      f"median={np.median(pred_h):.2f}", flush=True)
healthy_stat = dict(n=n_h,
                    mean=float(pred_h.mean()),
                    median=float(np.median(pred_h)),
                    p25=float(np.percentile(pred_h, 25)),
                    p75=float(np.percentile(pred_h, 75)),
                    p95=float(np.percentile(pred_h, 95)),
                    std=float(pred_h.std()),
                    min=float(pred_h.min()), max=float(pred_h.max()))
pd.DataFrame([healthy_stat]).to_csv(
    os.path.join(RES, "37d_ridgeSNV_healthy.tsv"), sep="\t", index=False)


# ============================================================
# Markdown
# ============================================================
xb_lot_8 = df_b[(df_b.scope=="cross-batch") & (df_b.level=="lot_mean") &
                (df_b.threshold==8)].iloc[0]
in_lot_8 = df_b[(df_b.scope=="in-domain") & (df_b.level=="lot_mean") &
                (df_b.threshold==8)].iloc[0]

md = ["# Phase 4.5 - Ridge+SNV full industrial metrics",
      "",
      "Ridge with SNV preprocessing is the new cross-batch champion per script 36 "
      f"(cross-batch lot AUC@8 = 0.971 vs GBM+SG2 baseline 0.935). This script "
      "delivers the same metric package as scripts 33/34 for Ridge+SNV.",
      "",
      "## (a) AUC and PR-AUC",
      "",
      "| scope | level | threshold | n | n_pos | base_rate | ROC-AUC | PR-AUC |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in df_a.iterrows():
    md.append(f"| {r['scope']} | {r['level']} | {r['threshold']} ppb | "
              f"{r['n']} | {r['n_pos']} | {r['base_rate']:.3f} | "
              f"**{r['ROC_AUC']:.3f}** | {r['PR_AUC']:.3f} |")

md += ["",
       "## (b) Recall@95 operating point",
       "",
       "| scope | level | threshold | decision_thr | achieved_recall | FPR | Spec | PPV | NPV |",
       "|---|---|---|---|---|---|---|---|---|"]
for _, r in df_b.iterrows():
    md.append(f"| {r['scope']} | {r['level']} | {r['threshold']} ppb | "
              f"{r['decision_thr']:.2f} | {r['achieved_recall']:.3f} | "
              f"**{r['FPR_at_recall95']:.3f}** | {r['specificity']:.3f} | "
              f"{r['PPV']:.3f} | {r['NPV']:.3f} |")

md += ["",
       "## (c) Three-tier matrix @ 8 ppb",
       "",
       "| scope | level | tier | n | % | actual unsafe rate | mean true ppb | L | H |",
       "|---|---|---|---|---|---|---|---|---|"]
for _, r in df_c.iterrows():
    md.append(f"| {r['scope']} | {r['level']} | **{r['tier']}** | {r['n']} | "
              f"{r['pct']:.1f}% | {r['true_unsafe_rate']:.3f} | "
              f"{r['mean_true_ppb']:.2f} | {r['L_thr']:.2f} | {r['H_thr']:.2f} |")

md += ["",
       "## (d) Healthy control (v3 Level 01, true AFB1 = 0.00 ppb)",
       "",
       f"| n | mean | median | p25 | p75 | p95 | std | min | max |",
       f"|---|---|---|---|---|---|---|---|---|",
       f"| {healthy_stat['n']} | {healthy_stat['mean']:.2f} | "
       f"{healthy_stat['median']:.2f} | {healthy_stat['p25']:.2f} | "
       f"{healthy_stat['p75']:.2f} | {healthy_stat['p95']:.2f} | "
       f"{healthy_stat['std']:.2f} | {healthy_stat['min']:.2f} | "
       f"{healthy_stat['max']:.2f} |",
       "",
       "## Operating-point summary for ROI plug-in",
       "",
       f"- In-domain lot-level @ 8 ppb recall=95: FPR = **{in_lot_8['FPR_at_recall95']:.3f}**, Spec = {in_lot_8['specificity']:.3f}",
       f"- Cross-batch lot-level @ 8 ppb recall=95: FPR = **{xb_lot_8['FPR_at_recall95']:.3f}**, Spec = {xb_lot_8['specificity']:.3f}",
       "",
       "Plug these into script 35 (ROI) as Ridge+SNV deployment numbers.",
       "",
       "## Outputs",
       "- `results/37a_ridgeSNV_auc.tsv`",
       "- `results/37b_ridgeSNV_recall95.tsv`",
       "- `results/37c_ridgeSNV_tier.tsv`",
       "- `results/37d_ridgeSNV_healthy.tsv`",
       "- `results/37_pred_{v1,v3}_RidgeSNV.npy` (saved pred vectors)"]

with open(os.path.join(RES, "37_ridgeSNV_industrial.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/37_ridgeSNV_industrial.md + 4 tsv + 2 npy", flush=True)
print("DONE", flush=True)
