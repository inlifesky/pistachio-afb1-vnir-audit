"""
Phase A.3 - Sampling sensitivity: does the cross-batch headline depend on
            mask threshold, pixel count, or random seed?

ChatGPT review (2026-06-03) flagged two specific risks:
  (a) the 55th-percentile foreground mask at ~650 nm could itself be
      label-dependent if AFB1 changes kernel brightness;
  (b) 300 pixels per cube might be too few (or too noise-sensitive) for the
      cross-batch ridge under SG2+SNV.

Design: pin the model (Ridge+SG2+SNV; matches script 37 = paper's recommended
pipeline), and vary only:
  - mask_quantile in {0.45, 0.55, 0.65}   # baseline = 0.55
  - pixels_per_cube in {100, 300, 1000}   # baseline = 300
  - seed in {0, 1, 2, 3, 4}               # baseline = 42

Report cross-batch lot AUC@8 mean and std across seeds for each
(quantile, pixels) cell. If std < ~0.02 and means cluster within a 0.03 band
the recommended pipeline is robust to sampling choices.
"""
import os, sys, time, glob, warnings
import os as _os
# Env-driven paths; defaults work when scripts are run from the repo root.
# Override via PISTACHIO_RES / PISTACHIO_V1_DATA / PISTACHIO_V3_DATA env vars.
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
from pistachio_io import read_bil, AFB1_PPB, REFLECTANCE_SCALE

V1_ROOT = _os.environ["PISTACHIO_V1_DATA"]  # unzipped Zenodo v1 cubes
V3_ROOT = _os.environ["PISTACHIO_V3_DATA"]  # unzipped Zenodo v3 cubes
RES     = _os.environ.get("PISTACHIO_RES", "results")

V3_PPB = {
    "Level 01": 0.00, "Level 02": 0.40, "Level 03": 0.67, "Level 04": 0.88,
    "Level 05": 1.13, "Level 06": 1.66, "Level 07": 2.15, "Level 08": 2.30,
    "Level 09": 2.48, "Level 10": 2.82, "Level 11": 3.01, "Level 12": 3.05,
    "Level 13": 3.85, "Level 14": 4.43, "Level 15": 5.12, "Level 16": 5.30,
    "Level 17": 6.37, "Level 18": 8.93, "Level 19": 12.16, "Level 20": 17.12,
    "Level 21": 24.03, "Level 22": 26.14, "Level 23": 33.17, "Level 24": 56.06,
    "Level 25": 57.29, "Level 26": 114.67,
}
# Excluded v1 cubes (per script 19 logs)
V1_SKIP = set()  # empty by default; original 57/66 exclusion was at read-time

WAVELENGTHS = np.linspace(386.88, 1003.60, 462)
MASK_BAND_IDX = int(np.argmin(np.abs(WAVELENGTHS - 650.0)))


def discover_cubes(root, ppb_map):
    rows = []
    for level in sorted(ppb_map):
        d = os.path.join(root, level)
        if not os.path.isdir(d):
            continue
        for bil in sorted(glob.glob(os.path.join(d, "*.bil"))):
            rows.append((level, os.path.basename(bil), bil, ppb_map[level]))
    return pd.DataFrame(rows, columns=["level", "image", "path", "ppb"])


def sample_from_cube(bil_path, mask_q, n_pix, rng):
    """Return n_pix x 462 spectra (float32 reflectance) sampled from foreground."""
    try:
        cube = read_bil(bil_path).astype(np.float32) / REFLECTANCE_SCALE
    except Exception as e:
        return None
    band = cube[..., MASK_BAND_IDX]
    thr = np.quantile(band, mask_q)
    mask = band > thr
    fg_pix = cube[mask]  # (N_fg, 462)
    if len(fg_pix) < n_pix:
        return None
    idx = rng.choice(len(fg_pix), size=n_pix, replace=False)
    return fg_pix[idx]


def snv(X):
    X = np.asarray(X, float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True); sd[sd == 0] = 1.0
    return (X - mu) / sd


def build_pool(df, mask_q, n_pix, seed):
    """Extract n_pix per cube, return X (N, 462), y (N,), images (N,)."""
    rng = np.random.default_rng(seed)
    Xs, ys, imgs = [], [], []
    for _, r in df.iterrows():
        S = sample_from_cube(r["path"], mask_q, n_pix, rng)
        if S is None:
            continue
        Xs.append(S)
        ys.append(np.full(len(S), r["ppb"]))
        imgs.append(np.full(len(S), r["image"]))
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(imgs)


def lot_auc(pred, y, images, thr):
    df = pd.DataFrame(dict(pred=pred, y=y, img=images))
    lot = df.groupby("img").agg(p=("pred", "mean"), y=("y", "first")).reset_index()
    yb = (lot["y"].values >= thr).astype(int)
    if yb.sum() == 0 or yb.sum() == len(yb):
        return np.nan, len(lot)
    return roc_auc_score(yb, lot["p"].values), len(lot)


# --- discover cubes ---
v1_df = discover_cubes(V1_ROOT, AFB1_PPB)
v3_df = discover_cubes(V3_ROOT, V3_PPB)
print(f"[discover] v1 cubes {len(v1_df)}  v3 cubes {len(v3_df)}", flush=True)

# --- sweep ---
MASK_QS = [0.45, 0.55, 0.65]
N_PIXS  = [100, 300, 1000]
SEEDS   = [0, 1, 2, 3, 4]
rows = []
t0 = time.time()
for mq in MASK_QS:
    for npx in N_PIXS:
        for sd in SEEDS:
            t1 = time.time()
            X1, y1, I1 = build_pool(v1_df, mq, npx, sd)
            X3, y3, I3 = build_pool(v3_df, mq, npx, sd + 100)  # different seed for v3
            # Ridge + SG2 + SNV pipeline (matches script 37)
            X1_t = snv(_sg(X1, 2))
            X3_t = snv(_sg(X3, 2))
            m = Pipeline([("scale", StandardScaler()),
                          ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])
            m.fit(X1_t, y1)
            pred3 = m.predict(X3_t)
            auc, n_lot = lot_auc(pred3, y3, I3, 8)
            rows.append(dict(mask_q=mq, n_pix=npx, seed=sd, n_lot=n_lot,
                              v1_n=len(X1), v3_n=len(X3), AUC8=auc))
            print(f"  q={mq} npix={npx} seed={sd}  "
                  f"v1={len(X1)} v3={len(X3)} n_lot={n_lot}  "
                  f"AUC@8={auc:.3f}  ({time.time()-t1:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "43_sampling_sensitivity.tsv"), sep="\t", index=False)

# --- aggregate ---
agg = df.groupby(["mask_q", "n_pix"]).agg(
    AUC8_mean=("AUC8", "mean"),
    AUC8_std =("AUC8", "std"),
    AUC8_min =("AUC8", "min"),
    AUC8_max =("AUC8", "max"),
    n_seeds  =("AUC8", "count")).reset_index()
agg.to_csv(os.path.join(RES, "43_sampling_sensitivity_agg.tsv"), sep="\t", index=False)

# --- markdown ---
md = ["# Phase A.3 — Sampling sensitivity of the recommended pipeline", "",
      "Pipeline = Ridge + SG2 + SNV (paper's recommended), as in script 37. "
      "Cross-batch v1->v3, metric = lot AUC at 8 µg/kg.",
      "",
      "Vary only the pixel-sampling design: foreground-mask quantile, "
      "pixels per cube, and seed.",
      "",
      "## Mean ± std AUC@8 across seeds",
      "",
      "| mask_quantile | n_pix/cube | mean AUC@8 | std | min | max | n_seeds |",
      "|---|---|---|---|---|---|---|"]
for _, r in agg.iterrows():
    md.append(f"| {r['mask_q']} | {int(r['n_pix'])} | "
              f"{r['AUC8_mean']:.3f} | {r['AUC8_std']:.3f} | "
              f"{r['AUC8_min']:.3f} | {r['AUC8_max']:.3f} | "
              f"{int(r['n_seeds'])} |")

base = agg[(agg.mask_q == 0.55) & (agg.n_pix == 300)].iloc[0]
md += ["",
       "## Read-out",
       "",
       f"- Baseline cell (q=0.55, n_pix=300) used in scripts 19/32/36/37: "
       f"mean AUC@8 = **{base['AUC8_mean']:.3f}** (std = {base['AUC8_std']:.3f}).",
       f"- Range across all 9 sampling cells: "
       f"**{agg['AUC8_mean'].min():.3f}** to **{agg['AUC8_mean'].max():.3f}**.",
       f"- Max within-cell seed-std: **{agg['AUC8_std'].max():.3f}**.",
       "",
       "If the range across cells stays within ~0.03 of the baseline and "
       "seed-std is < 0.02, the headline cross-batch AUC@8 = 0.971 is robust to "
       "the specific sampling choices made in the main pipeline.",
       "",
       "## Outputs",
       "- `results/43_sampling_sensitivity.tsv` per-cell rows",
       "- `results/43_sampling_sensitivity_agg.tsv` (mask_q, n_pix) summary",
       ]
with open(os.path.join(RES, "43_sampling_sensitivity.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/43_sampling_sensitivity.{{tsv,md}}", flush=True)
print(f"DONE in {time.time()-t0:.0f}s", flush=True)
