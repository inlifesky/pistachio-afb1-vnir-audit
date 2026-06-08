"""
Phase 4.2 - Three industrial gaps the previous scripts missed:

  (a) PR-AUC (Average Precision) at 8/10/12/15 ppb thresholds. For class-
      imbalanced food safety tasks (base-rate < 20% at low thresholds),
      PR-AUC is a more honest metric than ROC-AUC.

  (b) Calibration: model scores in ppb are not probabilities. Fit a Platt
      logistic on (pred, y_bin) per (model, scope, threshold), then evaluate
      reliability (10-bin), Brier score, ECE. Tells the user whether the
      model's score values can be trusted as risk probabilities.

  (c) Lot-level aggregation: industrial decisions are made per lot/.bil image,
      not per pixel. Aggregate per-image scores with mean / median / 90th
      percentile of pixel preds, then evaluate AUC and recall@95 at image level.

Inputs: saved .npy preds from scripts 27 (in-domain RF/GBM) and 32 (cross-batch
Ridge/RF/GBM); rebuild v1 Ridge in-domain GKF pred locally.
"""
import os, sys, warnings
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                              roc_auc_score, roc_curve, brier_score_loss)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
warnings.filterwarnings("ignore")

RES = _os.environ.get("PISTACHIO_RES", "results")
THRESHOLDS = [8, 10, 12, 15]
TARGET_RECALL = 0.95

print("[load v1]", flush=True)
X1_raw = np.load(os.path.join(RES, "pistachio_spectra.npy"))
meta1 = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
y1 = meta1["AFB1_ppb"].values
images1 = meta1["image"].values
X1 = _sg(X1_raw.astype(float), 2)

print("[load v3]", flush=True)
X3_raw = np.load(os.path.join(RES, "pistachio_v3_spectra.npy"))
meta3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t")
y3 = meta3["AFB1_ppb"].values
images3 = meta3["image"].values

print("[rebuild] v1 Ridge GKF preds", flush=True)
gkf = GroupKFold(n_splits=5)
pred_v1_ridge = np.full(len(y1), np.nan)
for tr, te in gkf.split(X1, y1, groups=images1):
    pipe = Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    pipe.fit(X1[tr], y1[tr])
    pred_v1_ridge[te] = pipe.predict(X1[te])
print(f"  AUC@8 = {roc_auc_score((y1>8).astype(int), pred_v1_ridge):.3f}",
      flush=True)

PRED = {
    ("Ridge", "in-domain"):    (y1, pred_v1_ridge, images1),
    ("RF",    "in-domain"):    (y1, np.load(os.path.join(RES, "27_pred_RF_seed42.npy")), images1),
    ("GBM",   "in-domain"):    (y1, np.load(os.path.join(RES, "27_pred_GBM.npy")), images1),
    ("Ridge", "cross-batch"):  (y3, np.load(os.path.join(RES, "32_pred_v3_Ridge.npy")), images3),
    ("RF",    "cross-batch"):  (y3, np.load(os.path.join(RES, "32_pred_v3_RF_n200_d15.npy")), images3),
    ("GBM",   "cross-batch"):  (y3, np.load(os.path.join(RES, "32_pred_v3_GBM_iter200_d8.npy")), images3),
}

MODELS = ["Ridge", "RF", "GBM"]
SCOPES = ["in-domain", "cross-batch"]


# ============================================================
# (a) PR-AUC
# ============================================================
print("\n[a] PR-AUC ...", flush=True)
rows_a = []
for model in MODELS:
    for scope in SCOPES:
        y_true, pred, _ = PRED[(model, scope)]
        for thr in THRESHOLDS:
            y_bin = (y_true > thr).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            base_rate = float(y_bin.mean())
            roc_auc = roc_auc_score(y_bin, pred)
            ap = average_precision_score(y_bin, pred)
            rows_a.append(dict(model=model, scope=scope, threshold_ppb=thr,
                               n=len(y_bin), n_pos=int(y_bin.sum()),
                               base_rate=base_rate,
                               ROC_AUC=roc_auc, PR_AUC=ap,
                               PR_AUC_lift_over_baseline=ap - base_rate))
df_a = pd.DataFrame(rows_a)
df_a.to_csv(os.path.join(RES, "34a_pr_auc.tsv"), sep="\t", index=False)


# ============================================================
# (b) Calibration via Platt scaling
# ============================================================
print("\n[b] Calibration ...", flush=True)


def platt_calibrate_split(pred_scores, y_bin, n_folds=5, rng_seed=42):
    """Fit logistic on half, evaluate on the other half (cross-validated to
    avoid the leakage of calibrating on the same data used for AUC). Returns
    a calibrated probability for every sample."""
    n = len(y_bin)
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(n); rng.shuffle(idx)
    folds = np.array_split(idx, n_folds)
    prob = np.full(n, np.nan)
    for f in range(n_folds):
        te = folds[f]
        tr = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        if y_bin[tr].sum() < 2 or y_bin[tr].sum() > len(tr) - 2:
            continue
        lr = LogisticRegression(C=1.0, max_iter=2000)
        lr.fit(pred_scores[tr].reshape(-1, 1), y_bin[tr])
        prob[te] = lr.predict_proba(pred_scores[te].reshape(-1, 1))[:, 1]
    return prob


def reliability_curve(prob, y_bin, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = bin_id == b
        if sel.sum() == 0:
            continue
        rows.append(dict(bin=b, bin_low=bins[b], bin_high=bins[b+1],
                         n=int(sel.sum()), mean_pred=float(prob[sel].mean()),
                         frac_pos=float(y_bin[sel].mean())))
    return pd.DataFrame(rows)


def ece(prob, y_bin, n_bins=10):
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_id = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    e = 0.0; N = len(y_bin)
    for b in range(n_bins):
        sel = bin_id == b
        if sel.sum() == 0:
            continue
        e += sel.sum() / N * abs(prob[sel].mean() - y_bin[sel].mean())
    return float(e)


rows_b = []
reliability_data = {}
for model in MODELS:
    for scope in SCOPES:
        y_true, pred, _ = PRED[(model, scope)]
        for thr in THRESHOLDS:
            y_bin = (y_true > thr).astype(int)
            if y_bin.sum() < 5 or y_bin.sum() > len(y_bin) - 5:
                continue
            prob = platt_calibrate_split(pred, y_bin)
            valid = ~np.isnan(prob)
            if valid.sum() < 100:
                continue
            brier = brier_score_loss(y_bin[valid], prob[valid])
            e_cal = ece(prob[valid], y_bin[valid])
            base_brier = brier_score_loss(y_bin[valid],
                                          np.full(valid.sum(), y_bin[valid].mean()))
            rows_b.append(dict(model=model, scope=scope, threshold_ppb=thr,
                               n=int(valid.sum()),
                               brier_score=brier,
                               brier_baseline=base_brier,
                               brier_skill_score=1 - brier/base_brier if base_brier > 0 else np.nan,
                               ECE=e_cal))
            if thr == 8:
                reliability_data[(model, scope)] = reliability_curve(prob[valid],
                                                                      y_bin[valid])
df_b = pd.DataFrame(rows_b)
df_b.to_csv(os.path.join(RES, "34b_calibration.tsv"), sep="\t", index=False)

# plot reliability diagrams @ 8 ppb
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
for i, model in enumerate(MODELS):
    for j, scope in enumerate(SCOPES):
        ax = axes[j, i]
        rc = reliability_data.get((model, scope))
        if rc is None or rc.empty:
            ax.set_title(f"{model} / {scope}\n(no data)")
            continue
        ax.plot([0, 1], [0, 1], "k--", lw=0.7, alpha=0.5, label="ideal")
        ax.plot(rc["mean_pred"], rc["frac_pos"], "o-", color="#4C72B0",
                label="model")
        ax.scatter(rc["mean_pred"], rc["frac_pos"],
                   s=np.clip(rc["n"]/30, 10, 200), alpha=0.4, color="#4C72B0")
        ax.set_title(f"{model} / {scope}")
        ax.grid(alpha=0.3)
        if i == 0: ax.set_ylabel("actual frac positive")
        if j == 1: ax.set_xlabel("Platt-calibrated probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="upper left")
fig.suptitle("Reliability diagrams @ 8 ppb threshold (Platt-calibrated, 10 bins, "
             "marker size proportional to n)", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(RES, "34b_reliability_diagrams.png"), dpi=140)
plt.close()


# ============================================================
# (c) Lot-level aggregation
# ============================================================
print("\n[c] Lot-level aggregation ...", flush=True)


def aggregate_to_lots(pred, y_true, images, aggregator):
    """Aggregate pixel preds to per-image (lot) preds.
    Aggregator: 'mean' | 'median' | 'p90'."""
    df = pd.DataFrame(dict(pred=pred, y=y_true, image=images))
    if aggregator == "mean":
        agg = df.groupby("image").agg(pred=("pred", "mean"), y=("y", "first"))
    elif aggregator == "median":
        agg = df.groupby("image").agg(pred=("pred", "median"), y=("y", "first"))
    elif aggregator == "p90":
        agg = df.groupby("image").agg(
            pred=("pred", lambda x: np.percentile(x, 90)), y=("y", "first"))
    else:
        raise ValueError(aggregator)
    return agg["pred"].values, agg["y"].values


def find_recall_op(y_bin, scores, target=0.95):
    fpr, tpr, thr = roc_curve(y_bin, scores)
    sel = np.where(tpr >= target)[0]
    if len(sel) == 0:
        return np.nan, np.nan, np.nan
    i = sel[0]
    return float(thr[i]), float(tpr[i]), float(fpr[i])


rows_c = []
for model in MODELS:
    for scope in SCOPES:
        y_true, pred, images = PRED[(model, scope)]
        for agg in ["mean", "median", "p90"]:
            lot_pred, lot_y = aggregate_to_lots(pred, y_true, images, agg)
            for thr in THRESHOLDS:
                y_bin = (lot_y > thr).astype(int)
                if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                    continue
                roc_auc = roc_auc_score(y_bin, lot_pred)
                ap = average_precision_score(y_bin, lot_pred)
                dec_thr, rec, fpr95 = find_recall_op(y_bin, lot_pred,
                                                     target=TARGET_RECALL)
                rows_c.append(dict(model=model, scope=scope, aggregator=agg,
                                   threshold_ppb=thr, n_lots=len(lot_y),
                                   n_pos=int(y_bin.sum()),
                                   ROC_AUC=roc_auc, PR_AUC=ap,
                                   decision_thr_at_recall95=dec_thr,
                                   FPR_at_recall95=fpr95))
df_c = pd.DataFrame(rows_c)
df_c.to_csv(os.path.join(RES, "34c_lot_level.tsv"), sep="\t", index=False)


# ============================================================
# Markdown
# ============================================================
md = ["# Phase 4.2 - PR-AUC + Calibration + Lot-level (gaps from discussion)",
      "",
      "Three industrial metrics the earlier scripts missed:",
      "- **PR-AUC** (Average Precision) for class-imbalanced food-safety classification.",
      "- **Calibration** (Platt scaling + reliability + Brier + ECE) for using model scores as risk probabilities.",
      "- **Lot-level aggregation** per .bil image, the true industrial decision unit.",
      "",
      "## (a) PR-AUC at 8/10/12/15 ppb",
      "",
      "PR-AUC = 1.0 means perfect ranking; PR-AUC = base_rate means model is no better than always predicting positive. Lift = PR_AUC - base_rate.",
      "",
      "| model | scope | thr | base_rate | ROC-AUC | PR-AUC | Lift |",
      "|---|---|---|---|---|---|---|"]
for _, r in df_a.iterrows():
    md.append(f"| {r['model']} | {r['scope']} | {r['threshold_ppb']} | "
              f"{r['base_rate']:.3f} | {r['ROC_AUC']:.3f} | "
              f"**{r['PR_AUC']:.3f}** | {r['PR_AUC_lift_over_baseline']:+.3f} |")

md += ["",
       "## (b) Calibration (Platt-scaled, 10-bin reliability)",
       "",
       "Brier Skill Score (BSS): 1.0 perfect, 0 no better than constant base rate, "
       "negative worse than baseline. ECE: lower is better, 0 perfect.",
       "",
       "| model | scope | thr | Brier | BSS | ECE | n |",
       "|---|---|---|---|---|---|---|"]
for _, r in df_b.iterrows():
    md.append(f"| {r['model']} | {r['scope']} | {r['threshold_ppb']} | "
              f"{r['brier_score']:.4f} | {r['brier_skill_score']:+.3f} | "
              f"{r['ECE']:.4f} | {r['n']} |")
md += ["",
       "Reliability diagrams @ 8 ppb: `results/34b_reliability_diagrams.png`.",
       ""]

md += ["",
       "## (c) Lot-level aggregation (per-image .bil decision unit)",
       "",
       "Pixel predictions aggregated to lot-level by mean / median / p90. Reports "
       "AUC, PR-AUC, and FPR@95-recall at the lot level.",
       "",
       "| model | scope | agg | thr | n_lots | n_pos | ROC-AUC | PR-AUC | FPR@95rec |",
       "|---|---|---|---|---|---|---|---|---|"]
for _, r in df_c.iterrows():
    fpr_str = f"{r['FPR_at_recall95']:.3f}" if not np.isnan(r['FPR_at_recall95']) else "n/a"
    md.append(f"| {r['model']} | {r['scope']} | {r['aggregator']} | "
              f"{r['threshold_ppb']} | {int(r['n_lots'])} | {int(r['n_pos'])} | "
              f"{r['ROC_AUC']:.3f} | {r['PR_AUC']:.3f} | {fpr_str} |")

md += ["",
       "## Read-out hints",
       "- For class-imbalanced thresholds (8 ppb in-domain base rate = 15.8%), PR-AUC is the honest metric. ROC-AUC tends to be optimistic when negatives dominate.",
       "- BSS > 0 means the calibrated score adds information over knowing only the base rate. BSS near 0 means the model output cannot be trusted as a probability even after calibration.",
       "- Lot-level metrics are the industrially-relevant decision; pixel-level metrics are upper-bound surrogates because they treat each pixel as an independent observation.",
       "",
       "## Outputs",
       "- `results/34a_pr_auc.tsv`",
       "- `results/34b_calibration.tsv`",
       "- `results/34b_reliability_diagrams.png`",
       "- `results/34c_lot_level.tsv`"]

with open(os.path.join(RES, "34_pr_auc_calibration_lot.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/34_pr_auc_calibration_lot.md + 3 tsv + 1 png", flush=True)
print("DONE", flush=True)
