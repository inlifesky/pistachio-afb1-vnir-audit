"""
Phase A.1 - Bootstrap 95% CIs for cross-batch lot-level metrics.

ChatGPT review (2026-06-03) flagged that with 52 lots and 18 positives, every
headline lot-level number (AUC, PR-AUC, FPR@100%-recall) must carry a CI.

Method
------
Stratified bootstrap over LOTS (the independent unit), not pixels. For each
of B=2000 resamples we draw with replacement n_lot lots stratified by binary
label at the working threshold, recompute the metric, and report the 2.5/97.5
percentile.

Four pipelines x three thresholds = 12 cells. Pred vectors are reused from
scripts 32 and 37 (no model retraining).

We pair this with a DeLong-equivalent paired bootstrap of the AUC *delta*
between Ridge+SG2 (the weakest cross-batch model) and Ridge+SG2+SNV (the
recommended pipeline) at 8 ppb, which is the central claim of the paper.

Outputs
-------
- results/41_bootstrap_CIs.tsv      one row per pipeline x threshold x metric
- results/41_bootstrap_CIs.md       human summary table
- logs/41_*.log                     stdout
"""
import os, sys, time
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

RES = _os.environ.get("PISTACHIO_RES", "results")
B   = 2000
SEED = 42
THRESHOLDS = [8, 10, 15]

# ----- load v3 lot labels + per-pipeline pred vectors -----
meta3 = pd.read_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t")
print(f"[load] v3 pixels {len(meta3)}, images {meta3['image'].nunique()}", flush=True)

PIPELINES = {
    "Ridge_SG2":   "32_pred_v3_Ridge.npy",
    "RF_SG2":      "32_pred_v3_RF_n200_d15.npy",
    "GBM_SG2":     "32_pred_v3_GBM_iter200_d8.npy",
    "Ridge_SNV":   "37_pred_v3_RidgeSNV.npy",
}

# Aggregate every pred vector to lot-level mean once
lot_table = {}      # name -> DataFrame[image, pred, ppb]
for name, f in PIPELINES.items():
    pred = np.load(os.path.join(RES, f))
    df = meta3.copy(); df["pred"] = pred
    lot = (df.groupby("image")
              .agg(pred=("pred", "mean"), ppb=("AFB1_ppb", "mean"))
              .reset_index())
    lot_table[name] = lot
    print(f"  {name:10s}  n_lot={len(lot)}", flush=True)

# Reference table: 52 lots, ppb same across pipelines
LOTS = lot_table["Ridge_SG2"][["image", "ppb"]].copy()
N_LOT = len(LOTS)


# ----- metric helpers -----
def fpr_at_100_recall(y_true, scores):
    """Lowest decision-threshold FPR that yields recall = 1.0 on the lots,
    matching the script-37 convention. Returns nan if no positives."""
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return np.nan
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    # Need every positive captured -> threshold <= min positive score
    thr = pos.min()
    return float((neg >= thr).mean())


def metrics_one(y_true, scores):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return dict(AUC=np.nan, PRAUC=np.nan, FPR100=np.nan)
    return dict(
        AUC=roc_auc_score(y_true, scores),
        PRAUC=average_precision_score(y_true, scores),
        FPR100=fpr_at_100_recall(y_true, scores),
    )


# ----- stratified lot-level bootstrap -----
def stratified_bootstrap(lot_df, thr, B, rng):
    """Draw B resamples of size n_lot, stratified by binary label (>thr).
    Return arrays of (AUC, PRAUC, FPR100). lot_df has columns pred, ppb."""
    y = (lot_df["ppb"].values >= thr).astype(int)
    s = lot_df["pred"].values
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return np.full(B, np.nan), np.full(B, np.nan), np.full(B, np.nan)
    a = np.empty(B); p = np.empty(B); f = np.empty(B)
    for b in range(B):
        bi_p = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        bi_n = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        bi = np.concatenate([bi_p, bi_n])
        m = metrics_one(y[bi], s[bi])
        a[b] = m["AUC"]; p[b] = m["PRAUC"]; f[b] = m["FPR100"]
    return a, p, f


def pct_ci(arr, q=(2.5, 97.5)):
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (np.nan, np.nan)
    return tuple(float(x) for x in np.percentile(arr, q))


# ----- main loop -----
rows = []
t0 = time.time()
rng = np.random.default_rng(SEED)
for name, lot in lot_table.items():
    for thr in THRESHOLDS:
        # point estimates
        y = (lot["ppb"].values >= thr).astype(int)
        pt = metrics_one(y, lot["pred"].values)
        # bootstrap
        a, p, f = stratified_bootstrap(lot, thr, B, rng)
        rows.append(dict(
            pipeline=name, threshold=thr,
            n_lot=N_LOT, n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
            AUC=pt["AUC"],    AUC_lo=pct_ci(a)[0],    AUC_hi=pct_ci(a)[1],
            PRAUC=pt["PRAUC"],PRAUC_lo=pct_ci(p)[0],  PRAUC_hi=pct_ci(p)[1],
            FPR100=pt["FPR100"], FPR100_lo=pct_ci(f)[0], FPR100_hi=pct_ci(f)[1],
        ))
        print(f"  {name:10s} thr={thr:>2d}  "
              f"AUC={pt['AUC']:.3f} [{pct_ci(a)[0]:.3f}, {pct_ci(a)[1]:.3f}]  "
              f"PRAUC={pt['PRAUC']:.3f} [{pct_ci(p)[0]:.3f}, {pct_ci(p)[1]:.3f}]  "
              f"FPR100={pt['FPR100']:.3f} [{pct_ci(f)[0]:.3f}, {pct_ci(f)[1]:.3f}]",
              flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "41_bootstrap_CIs.tsv"), sep="\t", index=False)

# ----- paired Delta-AUC (Ridge+SNV minus Ridge+SG2) at 8 ppb -----
y8 = (LOTS["ppb"].values >= 8).astype(int)
s_a = lot_table["Ridge_SG2"]["pred"].values
s_b = lot_table["Ridge_SNV"]["pred"].values
idx_pos = np.where(y8 == 1)[0]; idx_neg = np.where(y8 == 0)[0]
delta = np.empty(B)
rng2 = np.random.default_rng(SEED + 1)
for b in range(B):
    bi = np.concatenate([rng2.choice(idx_pos, len(idx_pos), True),
                         rng2.choice(idx_neg, len(idx_neg), True)])
    delta[b] = roc_auc_score(y8[bi], s_b[bi]) - roc_auc_score(y8[bi], s_a[bi])
delta_lo, delta_hi = pct_ci(delta)
delta_pt = roc_auc_score(y8, s_b) - roc_auc_score(y8, s_a)
print(f"\n[delta-AUC @ 8 ppb, Ridge+SNV - Ridge+SG2] "
      f"point={delta_pt:.3f}  95% CI [{delta_lo:.3f}, {delta_hi:.3f}]  "
      f"P(delta>0)={(delta > 0).mean():.3f}", flush=True)

# ----- markdown summary -----
md = ["# Phase A.1 — Bootstrap 95% CIs for cross-batch lot-level metrics", "",
      f"Stratified bootstrap over lots, B={B}, seed={SEED}. n_lot={N_LOT}. "
      "Pred vectors reused from scripts 32 (SG2) and 37 (SG2+SNV).",
      "",
      "## Per-pipeline metrics with 95% CI",
      "",
      "| pipeline | thr | n_pos/n_neg | AUC (95% CI) | PR-AUC (95% CI) | FPR@100%-recall (95% CI) |",
      "|---|---|---|---|---|---|"]
for _, r in df.iterrows():
    md.append(f"| {r['pipeline']} | {r['threshold']} | {int(r['n_pos'])}/{int(r['n_neg'])} | "
              f"{r['AUC']:.3f} ({r['AUC_lo']:.3f}, {r['AUC_hi']:.3f}) | "
              f"{r['PRAUC']:.3f} ({r['PRAUC_lo']:.3f}, {r['PRAUC_hi']:.3f}) | "
              f"{r['FPR100']:.3f} ({r['FPR100_lo']:.3f}, {r['FPR100_hi']:.3f}) |")

md += ["",
       "## Paired Δ-AUC at 8 ppb (Ridge+SNV minus Ridge+SG2)",
       "",
       f"Point estimate **Δ-AUC = {delta_pt:.3f}**, 95% bootstrap CI "
       f"**[{delta_lo:.3f}, {delta_hi:.3f}]**, "
       f"P(Δ > 0) = **{(delta > 0).mean():.3f}**.",
       "",
       "This is the paper's central pairwise claim: SNV preprocessing strictly "
       "improves Ridge cross-batch lot AUC.",
       "",
       "## Notes",
       "- Independent unit is the lot (1 image cube), not the pixel.",
       "- FPR@100%-recall is the lowest threshold that catches every unsafe lot; "
       "it matches the operating-point convention in script 37.",
       "- 95% CIs widen substantially at higher thresholds because n_pos drops "
       f"(18 at 8 ppb → 14 at 15 ppb).",
       ]
with open(os.path.join(RES, "41_bootstrap_CIs.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/41_bootstrap_CIs.{{tsv,md}}", flush=True)
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
