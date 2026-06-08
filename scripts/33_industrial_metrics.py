"""
Phase 4.1 - Industrial metrics package (post-hoc on saved predictions).

Four analyses, no model retraining:
  (a) Multi-threshold classification: AUC + Sens/Spec/PPV/NPV/FPR at 8/10/12/15 ppb
  (b) Recall@95 operating point: decision_threshold giving recall>=0.95 -> FPR
  (c) Three-tier risk matrix: low/mid/high risk decisions with business rules
  (d) Healthy control: mean/percentiles of pred ppb on v3 Level 01 (true 0 ppb)

Inputs (already saved on disk):
  - v1 in-domain RF/GBM preds from script 27
  - v3 cross-batch Ridge/RF/GBM preds from script 32
  - v1 Ridge in-domain: re-run here (3 sec, no .npy saved earlier)
"""
import os, sys, warnings
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
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
print(f"  v1 ppb range {y1.min():.2f}-{y1.max():.2f}  n={len(y1)}", flush=True)
print(f"  v3 ppb range {y3.min():.2f}-{y3.max():.2f}  n={len(y3)}", flush=True)


# ----- rebuild v1 Ridge in-domain pred (not saved before) ---------------
print("\n[rebuild] v1 Ridge GKF predictions", flush=True)
gkf = GroupKFold(n_splits=5)
pred_v1_ridge = np.full(len(y1), np.nan)
for fold_i, (tr, te) in enumerate(gkf.split(X1, y1, groups=images1)):
    pipe = Pipeline([("scale", StandardScaler()),
                     ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
    pipe.fit(X1[tr], y1[tr])
    pred_v1_ridge[te] = pipe.predict(X1[te])
print(f"  Ridge v1 AUC@8 = {roc_auc_score((y1>8).astype(int), pred_v1_ridge):.3f} (sanity vs 0.796)",
      flush=True)


# ============================================================
# load all preds into a dict
# ============================================================
PRED = {
    ("Ridge", "in-domain"):    (y1, pred_v1_ridge),
    ("RF",    "in-domain"):    (y1, np.load(os.path.join(RES, "27_pred_RF_seed42.npy"))),
    ("GBM",   "in-domain"):    (y1, np.load(os.path.join(RES, "27_pred_GBM.npy"))),
    ("Ridge", "cross-batch"):  (y3, np.load(os.path.join(RES, "32_pred_v3_Ridge.npy"))),
    ("RF",    "cross-batch"):  (y3, np.load(os.path.join(RES, "32_pred_v3_RF_n200_d15.npy"))),
    ("GBM",   "cross-batch"):  (y3, np.load(os.path.join(RES, "32_pred_v3_GBM_iter200_d8.npy"))),
}

MODELS = ["Ridge", "RF", "GBM"]
SCOPES = ["in-domain", "cross-batch"]


# ============================================================
# (a) Multi-threshold classification metrics
# ============================================================
def class_metrics(y_true_bin, pred_scores, decision_thr):
    pred_bin = (pred_scores > decision_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_bin, pred_bin, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv  = tp / (tp + fp) if (tp + fp) else np.nan
    npv  = tn / (tn + fn) if (tn + fn) else np.nan
    fpr  = fp / (fp + tn) if (fp + tn) else np.nan
    return dict(n=len(y_true_bin), n_pos=int(np.sum(y_true_bin)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv, FPR=fpr)


def auc_safe(y_bin, pred):
    if 0 < y_bin.sum() < len(y_bin):
        return roc_auc_score(y_bin, pred)
    return np.nan


rows_a = []
for model in MODELS:
    for scope in SCOPES:
        y_true, pred = PRED[(model, scope)]
        for thr in THRESHOLDS:
            y_bin = (y_true > thr).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            auc = auc_safe(y_bin, pred)
            # default operating point: decision_threshold = thr (pred > thr -> unsafe)
            m = class_metrics(y_bin, pred, decision_thr=thr)
            rows_a.append(dict(model=model, scope=scope, threshold_ppb=thr,
                               operating="default (decision_thr = threshold)",
                               AUC=auc, **m))

df_a = pd.DataFrame(rows_a)
df_a.to_csv(os.path.join(RES, "33a_multi_threshold_default.tsv"), sep="\t", index=False)


# ============================================================
# (b) Recall@95% FPR (per model x scope x threshold)
# ============================================================
def find_decision_thr_for_recall(y_bin, pred_scores, target_recall):
    """Find lowest decision threshold s.t. recall >= target_recall on the (y_bin, pred) pair.
    Returns (decision_thr, recall_achieved, fpr_at_that_point)."""
    fpr_curve, tpr_curve, thresholds = roc_curve(y_bin, pred_scores)
    # tpr_curve is monotonically non-decreasing; find first index with tpr>=target
    sel = np.where(tpr_curve >= target_recall)[0]
    if len(sel) == 0:
        return np.nan, np.nan, np.nan
    idx = sel[0]
    return float(thresholds[idx]), float(tpr_curve[idx]), float(fpr_curve[idx])


rows_b = []
for model in MODELS:
    for scope in SCOPES:
        y_true, pred = PRED[(model, scope)]
        for thr in THRESHOLDS:
            y_bin = (y_true > thr).astype(int)
            if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
                continue
            dec_thr, rec, fpr = find_decision_thr_for_recall(y_bin, pred, TARGET_RECALL)
            if not np.isnan(dec_thr):
                m = class_metrics(y_bin, pred, decision_thr=dec_thr)
            else:
                m = dict(n=len(y_true), n_pos=int(y_bin.sum()), tn=0, fp=0, fn=0, tp=0,
                         sensitivity=np.nan, specificity=np.nan, PPV=np.nan,
                         NPV=np.nan, FPR=np.nan)
            rows_b.append(dict(model=model, scope=scope, threshold_ppb=thr,
                               target_recall=TARGET_RECALL,
                               decision_thr=dec_thr,
                               achieved_recall=rec,
                               FPR_at_recall95=fpr,
                               **m))

df_b = pd.DataFrame(rows_b)
df_b.to_csv(os.path.join(RES, "33b_recall95_fpr.tsv"), sep="\t", index=False)


# ============================================================
# (c) Three-tier risk matrix (focus on 8 ppb regulatory threshold)
# ============================================================
# Business rules at the 8 ppb threshold:
#   LOW risk:  pred < L_thr  (must have NPV >= 0.95 = at most 5% truly unsafe)
#   HIGH risk: pred >= H_thr (must have PPV >= 0.90 = at least 90% truly unsafe)
#   MID risk:  in between    (sent to lab for confirmation)
# Find L_thr and H_thr per (model, scope) by sweeping decision thresholds.
def find_tier_thresholds(y_bin, pred, npv_target=0.95, ppv_target=0.90):
    """Sweep decision thresholds; return L_thr (highest that keeps NPV>=npv_target)
    and H_thr (lowest that keeps PPV>=ppv_target)."""
    cands = np.linspace(pred.min(), pred.max(), 200)
    L = None; H = None
    for c in cands:
        m = class_metrics(y_bin, pred, c)
        # NPV: P(true safe | predicted safe).  pred < c == predicted safe.
        if not np.isnan(m["NPV"]) and m["NPV"] >= npv_target:
            L = c
    for c in cands[::-1]:
        m = class_metrics(y_bin, pred, c)
        if not np.isnan(m["PPV"]) and m["PPV"] >= ppv_target:
            H = c
    return L, H


def tier_breakdown(y_true, pred, threshold_ppb, L_thr, H_thr):
    """Assign each sample to low/mid/high. Report per-tier n and actual unsafe rate."""
    if L_thr is None or H_thr is None:
        return None
    if L_thr > H_thr:  # L>H means no tier separation possible
        return None
    low_sel  = pred <  L_thr
    high_sel = pred >= H_thr
    mid_sel  = ~low_sel & ~high_sel
    actual_unsafe = (y_true > threshold_ppb).astype(int)
    rows = []
    for name, sel in [("low", low_sel), ("mid", mid_sel), ("high", high_sel)]:
        n = int(sel.sum())
        if n == 0:
            rows.append(dict(tier=name, n=0, pct=0.0,
                             true_unsafe_n=0, true_unsafe_rate=np.nan,
                             mean_true_ppb=np.nan))
        else:
            rows.append(dict(tier=name, n=n, pct=100*n/len(pred),
                             true_unsafe_n=int(actual_unsafe[sel].sum()),
                             true_unsafe_rate=float(actual_unsafe[sel].mean()),
                             mean_true_ppb=float(y_true[sel].mean())))
    return rows


rows_c = []
threshold_for_tiers = 8  # EU AFB1 ready-to-eat
for model in MODELS:
    for scope in SCOPES:
        y_true, pred = PRED[(model, scope)]
        y_bin = (y_true > threshold_for_tiers).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        L_thr, H_thr = find_tier_thresholds(y_bin, pred)
        if L_thr is None or H_thr is None or L_thr > H_thr:
            print(f"  [skip-tier] {model}/{scope}: cannot satisfy NPV/PPV targets "
                  f"(L_thr={L_thr}, H_thr={H_thr})", flush=True)
            continue
        breakdown = tier_breakdown(y_true, pred, threshold_for_tiers, L_thr, H_thr)
        for row in breakdown:
            rows_c.append(dict(model=model, scope=scope,
                               threshold_ppb=threshold_for_tiers,
                               L_thr=L_thr, H_thr=H_thr, **row))

df_c = pd.DataFrame(rows_c)
df_c.to_csv(os.path.join(RES, "33c_three_tier_matrix.tsv"), sep="\t", index=False)


# ============================================================
# (d) Healthy control: v3 Level 01 = 0.00 ppb true
# ============================================================
healthy3 = (y3 == 0.0)
print(f"\n[healthy] v3 Level 01 (0.00 ppb) pixels: {int(healthy3.sum())}", flush=True)

# Also v1 lowest level (0.40 ppb, Level 01) as comparison
low1_thr = y1.min()
low1 = (y1 == low1_thr)
print(f"[healthy-proxy] v1 Level 01 ({low1_thr} ppb) pixels: {int(low1.sum())}",
      flush=True)


def desc(arr):
    return dict(n=len(arr), mean=float(arr.mean()), median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)),
                p75=float(np.percentile(arr, 75)),
                p95=float(np.percentile(arr, 95)),
                std=float(arr.std()),
                min=float(arr.min()), max=float(arr.max()))


rows_d = []
for model in MODELS:
    # v3 healthy (true 0 ppb)
    _, pred_v3 = PRED[(model, "cross-batch")]
    d = desc(pred_v3[healthy3])
    rows_d.append(dict(model=model, source="v3 Level 01 = 0.00 ppb (true healthy)",
                       true_ppb=0.0, **d))
    # v1 lowest (true 0.40 ppb)
    _, pred_v1 = PRED[(model, "in-domain")]
    d = desc(pred_v1[low1])
    rows_d.append(dict(model=model, source="v1 Level 01 = 0.40 ppb (proxy lowest)",
                       true_ppb=float(low1_thr), **d))

df_d = pd.DataFrame(rows_d)
df_d.to_csv(os.path.join(RES, "33d_healthy_control.tsv"), sep="\t", index=False)


# ============================================================
# Markdown report
# ============================================================
md = ["# Phase 4.1 - Industrial metrics package",
      "",
      "Post-hoc analyses on the saved prediction vectors from scripts 27 (in-domain "
      "v1 GKF) and 32 (cross-batch v1 -> v3). No model retraining. Four sub-analyses "
      "framed for industrial decision support.", ""]

# section (a)
md += ["## (a) Multi-threshold classification (default operating point)",
       "",
       "Decision rule: `pred > threshold -> unsafe`. Reports AUC, Sens, Spec, PPV, FPR.",
       "",
       "| model | scope | threshold | AUC | Sens | Spec | PPV | NPV | FPR | n / n_pos |",
       "|---|---|---|---|---|---|---|---|---|---|"]
for _, r in df_a.iterrows():
    md.append(f"| {r['model']} | {r['scope']} | {r['threshold_ppb']} ppb | "
              f"{r['AUC']:.3f} | {r['sensitivity']:.3f} | {r['specificity']:.3f} | "
              f"{r['PPV']:.3f} | {r['NPV']:.3f} | {r['FPR']:.3f} | "
              f"{r['n']} / {r['n_pos']} |")

# section (b)
md += ["",
       "## (b) Operating point at Recall = 0.95 -> FPR cost",
       "",
       f"For each (model, scope, threshold) find the *lowest* decision threshold that "
       f"achieves recall >= {TARGET_RECALL:.2f}. Report the resulting FPR — this is "
       "the false-alarm rate the user pays to guarantee 95% capture of unsafe lots.",
       "",
       "| model | scope | threshold | decision_thr | achieved_recall | FPR@95recall | Spec | PPV | n_pos |",
       "|---|---|---|---|---|---|---|---|---|"]
for _, r in df_b.iterrows():
    md.append(f"| {r['model']} | {r['scope']} | {r['threshold_ppb']} ppb | "
              f"{r['decision_thr']:.2f} | {r['achieved_recall']:.3f} | "
              f"**{r['FPR_at_recall95']:.3f}** | {r['specificity']:.3f} | "
              f"{r['PPV']:.3f} | {r['n_pos']} |")

# section (c)
md += ["",
       "## (c) Three-tier risk matrix @ 8 ppb threshold",
       "",
       "Business rules: LOW tier requires NPV >= 0.95 (only 5% of low-risk are "
       "actually unsafe); HIGH tier requires PPV >= 0.90 (90% of high-risk truly "
       "unsafe); MID tier = grey-area lots sent to lab confirmation.",
       "",
       "| model | scope | tier | n | % | actual unsafe rate | mean true ppb | L_thr | H_thr |",
       "|---|---|---|---|---|---|---|---|---|"]
for _, r in df_c.iterrows():
    md.append(f"| {r['model']} | {r['scope']} | **{r['tier']}** | {r['n']} | "
              f"{r['pct']:.1f}% | {r['true_unsafe_rate']:.3f} | "
              f"{r['mean_true_ppb']:.2f} | {r['L_thr']:.2f} | {r['H_thr']:.2f} |")

# section (d)
md += ["",
       "## (d) Healthy control: model prediction on 0 ppb samples",
       "",
       "v3 Level 01 (n=600 pixels, true AFB1 = 0.00 ppb, healthy control) lets us "
       "measure each model's regression-to-the-mean bias on a truly clean sample. "
       "v1 Level 01 (true 0.40 ppb) as a proxy comparison.",
       "",
       "| model | source | true_ppb | n | mean pred | median | p75 | p95 | max |",
       "|---|---|---|---|---|---|---|---|---|"]
for _, r in df_d.iterrows():
    md.append(f"| {r['model']} | {r['source']} | {r['true_ppb']:.2f} | {r['n']} | "
              f"{r['mean']:.2f} | {r['median']:.2f} | {r['p75']:.2f} | "
              f"{r['p95']:.2f} | {r['max']:.2f} |")

md += ["",
       "## Industrial decision read-out",
       "",
       "Use these tables together:",
       "- Section (b) tells you the false-alarm cost of high-recall operation per threshold per model.",
       "- Section (c) tells you what fraction of lots fall into the grey area that needs lab confirmation.",
       "- Section (d) calibrates how much to discount low-end model predictions for a truly healthy sample.",
       "",
       "## Outputs",
       "- `results/33a_multi_threshold_default.tsv`",
       "- `results/33b_recall95_fpr.tsv`",
       "- `results/33c_three_tier_matrix.tsv`",
       "- `results/33d_healthy_control.tsv`"]

with open(os.path.join(RES, "33_industrial_metrics.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/33_industrial_metrics.md + 4 tsv files", flush=True)
print("DONE", flush=True)
