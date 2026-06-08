"""
Phase 3.10 - Band-count sweep for industrial decision support.

Two industrial questions:
  G3 (band count): How many wavelengths does a multispectral camera need
                   to match HSI performance on AFB1 at the EU 8 ppb threshold?
  G4 (low-conc):   What is the discrimination capacity at the sub-detection-limit
                   level (we use 2 ppb as the threshold)?

Design (single experiment answering both):
  For k in [1, 3, 5, 10, 20, 50, 100, 200, 462]:
    GroupKFold(5) by image
      In each train fold: fit RidgeCV on 462 bands -> |beta| top-k bands chosen
      Refit Ridge on top-k bands (no leakage) -> predict test fold
    Report R^2, r, RMSE, AUC@8ppb, AUC@2ppb, zone-stratified r.

Comparison anchors at k=5 (re-running here, expected to match script 24):
  - Diaper 5 unsupervised bands (Case C)
  - Uniform-spaced 5 bands (every 92nd band)

Outputs:
  results/26_band_sweep.tsv
  results/26_band_sweep.md
  results/26_band_sweep.png   - AUC vs k, R^2 vs k
"""
import os, sys, time, warnings
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
from pistachio_io import EU_AFB1_THRESHOLD_PPB
warnings.filterwarnings("ignore")

RES = _os.environ.get("PISTACHIO_RES", "results")
SEED = 42
LOW_CONC_THRESHOLD = 2.0   # sub-detection-limit boundary

# --- load ---------------------------------------------------------------
X_raw = np.load(os.path.join(RES, "pistachio_spectra.npy"))
meta = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
y = meta["AFB1_ppb"].values
zones = meta["zone"].values
images = meta["image"].values
print(f"[load] {X_raw.shape[0]} pixels x {X_raw.shape[1]} bands", flush=True)

# wavelength axis
DATA_ROOT = _os.environ["PISTACHIO_V1_DATA"]  # unzipped Zenodo v1 cubes
hdr_path = os.path.join(DATA_ROOT, "Level 01", "L01_0001.hdr")
text = open(hdr_path, "r", encoding="utf-8").read()
s = text.find("wavelength = {") + len("wavelength = {")
e = text.find("}", s)
WAVELENGTHS = np.array([float(v.strip()) for v in text[s:e].split(",")])
N_BANDS = 462

# Diaper unsupervised 5 (for comparison anchor at k=5)
DIAPER_NM = [399.98, 584.64, 704.64, 866.21, 1002.23]
DIAPER_IDX = [int(np.argmin(np.abs(WAVELENGTHS - nm))) for nm in DIAPER_NM]
# Uniform-spaced 5 bands across the spectrum (every ~92 bands)
UNIFORM_IDX = list(np.linspace(0, N_BANDS-1, 5).astype(int))

# --- SG2 once -----------------------------------------------------------
X = _sg(X_raw.astype(float), 2)
print(f"[prep] SG2 done", flush=True)


def auc_at(y_true, pred, thr):
    truth = (y_true > thr).astype(int)
    if 0 < truth.sum() < len(truth):
        return roc_auc_score(truth, pred)
    return np.nan


def metrics(y_true, pred, label=""):
    resid = y_true - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y_true - y_true.mean())**2)
    rmse = np.sqrt(np.mean(resid**2))
    r = pearsonr(y_true, pred)[0]
    return dict(label=label, n=len(y_true), R2=r2, RMSE=rmse,
                bias=float(np.mean(pred-y_true)), pearson_r=r,
                AUC_8ppb=auc_at(y_true, pred, EU_AFB1_THRESHOLD_PPB),
                AUC_2ppb=auc_at(y_true, pred, LOW_CONC_THRESHOLD))


def run_sweep_k(k, mode="supervised"):
    """
    mode:
      'supervised' - Ridge |beta| top-k chosen in train fold (no leakage)
      'diaper'     - fixed Diaper 5 bands (only valid for k=5)
      'uniform'    - uniform-spaced k bands across spectrum
      'all'        - k must equal 462; use all bands
    Returns (pred vector, per-fold top-k indices list).
    """
    gkf = GroupKFold(n_splits=5)
    pred = np.full(len(y), np.nan)
    per_fold_idx = []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups=images)):
        scaler = StandardScaler().fit(X[tr])
        Xs_tr = scaler.transform(X[tr])
        Xs_te = scaler.transform(X[te])
        if mode == "supervised":
            r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs_tr, y[tr])
            order = np.argsort(np.abs(r.coef_))[::-1][:k]
            r2 = Ridge(alpha=r.alpha_).fit(Xs_tr[:, order], y[tr])
            pred[te] = r2.predict(Xs_te[:, order])
            per_fold_idx.append(order.tolist())
        elif mode == "diaper":
            order = np.array(DIAPER_IDX)
            r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs_tr[:, order], y[tr])
            pred[te] = r.predict(Xs_te[:, order])
            per_fold_idx.append(order.tolist())
        elif mode == "uniform":
            order = np.array(list(np.linspace(0, N_BANDS-1, k).astype(int)))
            r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs_tr[:, order], y[tr])
            pred[te] = r.predict(Xs_te[:, order])
            per_fold_idx.append(order.tolist())
        elif mode == "all":
            r = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs_tr, y[tr])
            pred[te] = r.predict(Xs_te)
            per_fold_idx.append(list(range(N_BANDS)))
        else:
            raise ValueError(mode)
    return pred, per_fold_idx


# --- run sweep ----------------------------------------------------------
K_LIST = [1, 3, 5, 10, 20, 50, 100, 200, 462]
rows = []
fold_idx_log = []

t0 = time.time()
for k in K_LIST:
    mode = "all" if k == N_BANDS else "supervised"
    label = f"sup_top{k}" if mode == "supervised" else f"all_{k}"
    print(f"\n[sweep] k={k:>3d} mode={mode}", flush=True)
    pred, fi = run_sweep_k(k, mode=mode)
    m = metrics(y, pred, label)
    row = dict(scheme=label, k=k, mode=mode, n=m["n"],
               R2=m["R2"], RMSE=m["RMSE"], r=m["pearson_r"],
               bias=m["bias"], AUC_8ppb=m["AUC_8ppb"], AUC_2ppb=m["AUC_2ppb"])
    for z in ("low","mid","high"):
        sel = zones == z
        if sel.sum() < 5:
            row[f"r_{z}"] = np.nan; continue
        m_z = metrics(y[sel], pred[sel])
        row[f"r_{z}"] = m_z["pearson_r"]
    rows.append(row)
    for fold_i, idx in enumerate(fi):
        for rank, b in enumerate(idx[:min(k,10)], 1):
            fold_idx_log.append(dict(scheme=label, k=k, fold=fold_i+1,
                                     rank=rank, band_idx=int(b),
                                     wavelength_nm=float(WAVELENGTHS[b])))
    print(f"  ==> R^2={m['R2']:+.3f}  r={m['pearson_r']:+.3f}  "
          f"AUC@8={m['AUC_8ppb']:.3f}  AUC@2={m['AUC_2ppb']:.3f}  "
          f"({time.time()-t0:.1f}s cumulative)", flush=True)

# anchors at k=5 -----------------------------------------------------------
print(f"\n[sweep] k=5 mode=diaper (anchor)", flush=True)
pred_d, fi_d = run_sweep_k(5, mode="diaper")
m_d = metrics(y, pred_d, "diaper_5")
rows.append(dict(scheme="diaper_5", k=5, mode="diaper", n=m_d["n"],
                 R2=m_d["R2"], RMSE=m_d["RMSE"], r=m_d["pearson_r"],
                 bias=m_d["bias"], AUC_8ppb=m_d["AUC_8ppb"], AUC_2ppb=m_d["AUC_2ppb"],
                 **{f"r_{z}": metrics(y[zones==z], pred_d[zones==z])["pearson_r"]
                    for z in ("low","mid","high")}))
print(f"  ==> R^2={m_d['R2']:+.3f}  AUC@8={m_d['AUC_8ppb']:.3f}  AUC@2={m_d['AUC_2ppb']:.3f}", flush=True)

print(f"\n[sweep] k=5 mode=uniform (anchor)", flush=True)
pred_u, fi_u = run_sweep_k(5, mode="uniform")
m_u = metrics(y, pred_u, "uniform_5")
rows.append(dict(scheme="uniform_5", k=5, mode="uniform", n=m_u["n"],
                 R2=m_u["R2"], RMSE=m_u["RMSE"], r=m_u["pearson_r"],
                 bias=m_u["bias"], AUC_8ppb=m_u["AUC_8ppb"], AUC_2ppb=m_u["AUC_2ppb"],
                 **{f"r_{z}": metrics(y[zones==z], pred_u[zones==z])["pearson_r"]
                    for z in ("low","mid","high")}))
print(f"  ==> R^2={m_u['R2']:+.3f}  AUC@8={m_u['AUC_8ppb']:.3f}  AUC@2={m_u['AUC_2ppb']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "26_band_sweep.tsv"), sep="\t", index=False)
pd.DataFrame(fold_idx_log).to_csv(os.path.join(RES, "26_band_sweep_fold_bands.tsv"),
                                  sep="\t", index=False)

# --- plot ---------------------------------------------------------------
sup_df = df[df["mode"].isin(["supervised", "all"])].sort_values("k")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].plot(sup_df["k"], sup_df["AUC_8ppb"], "o-", color="#4C72B0", label="supervised top-k")
axes[0].axhline(m_d["AUC_8ppb"], color="#DD8452", ls=":", label=f"Diaper 5 unsup ({m_d['AUC_8ppb']:.3f})")
axes[0].axhline(m_u["AUC_8ppb"], color="#55A868", ls=":", label=f"uniform 5 ({m_u['AUC_8ppb']:.3f})")
axes[0].axhline(0.5, color="black", ls="--", lw=0.6, alpha=0.5, label="chance")
axes[0].set_xscale("log")
axes[0].set_xlabel("number of bands k")
axes[0].set_ylabel("AUC at 8 ppb (regulatory)")
axes[0].set_title("G1: regulatory-threshold classification")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=8)

axes[1].plot(sup_df["k"], sup_df["AUC_2ppb"], "o-", color="#4C72B0", label="supervised top-k")
axes[1].axhline(m_d["AUC_2ppb"], color="#DD8452", ls=":", label=f"Diaper 5 unsup ({m_d['AUC_2ppb']:.3f})")
axes[1].axhline(m_u["AUC_2ppb"], color="#55A868", ls=":", label=f"uniform 5 ({m_u['AUC_2ppb']:.3f})")
axes[1].axhline(0.5, color="black", ls="--", lw=0.6, alpha=0.5, label="chance")
axes[1].set_xscale("log")
axes[1].set_xlabel("number of bands k")
axes[1].set_ylabel("AUC at 2 ppb (sub-detection-limit)")
axes[1].set_title("G4: low-concentration discrimination")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=8)

axes[2].plot(sup_df["k"], sup_df["R2"], "o-", color="#4C72B0", label="supervised top-k")
axes[2].axhline(m_d["R2"], color="#DD8452", ls=":", label=f"Diaper 5 unsup")
axes[2].axhline(m_u["R2"], color="#55A868", ls=":", label=f"uniform 5")
axes[2].axhline(0, color="black", ls="--", lw=0.6, alpha=0.5)
axes[2].set_xscale("log")
axes[2].set_xlabel("number of bands k")
axes[2].set_ylabel("R^2 (held-out)")
axes[2].set_title("Continuous regression quality")
axes[2].grid(alpha=0.3)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RES, "26_band_sweep.png"), dpi=140)
plt.close()

# --- engineering read-out -----------------------------------------------
# find smallest k that crosses several AUC@8 thresholds
def first_k_at_least(thr):
    sel = sup_df[sup_df["AUC_8ppb"] >= thr]
    return int(sel["k"].min()) if len(sel) else None

k50 = first_k_at_least(0.5)
k70 = first_k_at_least(0.7)
k75 = first_k_at_least(0.75)
k80 = first_k_at_least(0.80)

md = [
    "# Phase 3.10 - Band-count sweep (industrial decision support)",
    "",
    "**Question** (industrial reframing):",
    "- **G1**: at the EU 8 ppb regulatory threshold, how many wavelengths does a "
    "multispectral camera need to match the HSI baseline (AUC = 0.80)?",
    "- **G4**: what is the discrimination capacity at the sub-detection-limit zone "
    f"(threshold = {LOW_CONC_THRESHOLD} ppb)? Is HSI capable, even at full 462 bands?",
    "",
    "**Design**: GroupKFold(5)-by-image + Ridge regression on supervised |beta| top-k "
    "bands, with k chosen *inside* each train fold (no leakage). Anchors at k=5 from "
    "(a) Diaper unsupervised OSP selection, (b) uniformly spaced 5 bands.",
    "",
    f"**Samples**: {X.shape[0]} pixels from {meta['image'].nunique()} images, "
    f"AFB1 {y.min():.2f}-{y.max():.2f} ppb.",
    "",
    "## Sweep results",
    "",
    "| scheme | k | R^2 | r | RMSE | AUC@8ppb | AUC@2ppb | r_low | r_mid | r_high |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for _, r_ in df.iterrows():
    md.append(f"| {r_['scheme']} | {int(r_['k'])} | {r_['R2']:+.3f} | "
              f"{r_['r']:+.3f} | {r_['RMSE']:.2f} | {r_['AUC_8ppb']:.3f} | "
              f"{r_['AUC_2ppb']:.3f} | {r_['r_low']:+.3f} | {r_['r_mid']:+.3f} | "
              f"{r_['r_high']:+.3f} |")

md += [
    "",
    "## Industrial read-out",
    "",
    "### G1 (regulatory): minimum k to cross AUC@8 ppb thresholds",
    "",
    f"- AUC >= 0.50 (above chance)  : k >= {k50 if k50 else 'never within tested range'}",
    f"- AUC >= 0.70                 : k >= {k70 if k70 else 'never within tested range'}",
    f"- AUC >= 0.75                 : k >= {k75 if k75 else 'never within tested range'}",
    f"- AUC >= 0.80 (HSI baseline)  : k >= {k80 if k80 else 'never within tested range'}",
    "",
    "### G4 (sub-detection-limit): AUC at 2 ppb across all k",
    f"- Maximum AUC@2ppb in sweep   : {sup_df['AUC_2ppb'].max():.3f} at k={int(sup_df.loc[sup_df['AUC_2ppb'].idxmax(), 'k'])}",
    f"- Minimum AUC@2ppb            : {sup_df['AUC_2ppb'].min():.3f}",
    ("- **Interpretation** depends on actual values. If AUC@2 stays near 0.5 across "
     "all k, the 2 ppb sub-detection-limit zone is not separable by HSI on this "
     "dataset under any band count — a fundamental sensitivity limit, not a band-"
     "selection problem. Report this as a `<2 ppb is HSI-indistinguishable` zone."),
    "",
    "### Outputs",
    "- `results/26_band_sweep.tsv`           - sweep metrics per k + anchors",
    "- `results/26_band_sweep_fold_bands.tsv`- which bands each fold picked, per k",
    "- `results/26_band_sweep.png`            - 3-panel: AUC@8 / AUC@2 / R^2 vs k",
]
with open(os.path.join(RES, "26_band_sweep.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))

print(f"\n[write] results/26_band_sweep.{{tsv,md,png,fold_bands.tsv}}", flush=True)
print("DONE", flush=True)
