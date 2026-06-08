"""
Phase 3.4 — Load HyperPistachio, build sample table, baseline + zone analysis.

Pipeline:
  1. Iterate over 57 .bil cubes (19 levels x 3 images, Level 01-19)
  2. Apply pistachio-vs-background mask (kernels are bright on dark backdrop)
  3. From each cube, randomly sample N pistachio pixels (controls dataset size)
  4. Build sample table: (level, image, pixel_id, AFB1_ppb, is_unsafe, zone, spectrum)
  5. sg2 preprocessing
  6. Baseline ElasticNet regression on AFB1 ppb
  7. Zone-stratified evaluation:
       low  = AFB1 < 2 ppb        (Levels 01-05)
       mid  = 2 <= AFB1 < 8 ppb   (Levels 06-15, plus Level 17 by ppb)
       high = AFB1 >= 8 ppb       (Levels 16, 18, 19)
  8. LOIO evaluation (cross-image generalization = pistachio analog of cross-year)
"""
import os, sys, glob, warnings
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
from pistachio_io import (read_bil, AFB1_PPB, EU_AFB1_THRESHOLD_PPB,
                          REFLECTANCE_SCALE, ROWS, COLS, BANDS)
warnings.filterwarnings("ignore")

DATA_ROOT = r"D:\bioinformatics\project_pistachio_AFB1\data\pistachio\extracted\Dataset"
RES = r"D:\bioinformatics\project_pistachio_AFB1\results"
SEED = 42
RNG = np.random.default_rng(SEED)

# Sampling parameters
PIX_PER_IMAGE = 300       # random pistachio pixels per image
MASK_BAND_NM = 650        # band used for foreground/background separation (red)
MASK_QUANTILE = 0.55      # keep top fraction by reflectance (pistachios bright)

# Wavelength table (from .hdr, copied verbatim into preprocessing)
def load_wavelengths():
    hdr_path = os.path.join(DATA_ROOT, "Level 01", "L01_0001.hdr")
    text = open(hdr_path, "r", encoding="utf-8").read()
    s = text.find("wavelength = {") + len("wavelength = {")
    e = text.find("}", s)
    return np.array([float(v.strip()) for v in text[s:e].split(",")])


WAVELENGTHS = load_wavelengths()
print(f"[setup] {len(WAVELENGTHS)} wavelengths, range {WAVELENGTHS[0]:.1f}-{WAVELENGTHS[-1]:.1f} nm")

# Find the band closest to MASK_BAND_NM
mask_band_idx = int(np.argmin(np.abs(WAVELENGTHS - MASK_BAND_NM)))
print(f"        mask band idx={mask_band_idx} ({WAVELENGTHS[mask_band_idx]:.1f} nm)")


def zone_of(ppb):
    if ppb < 2.0: return "low"
    if ppb < EU_AFB1_THRESHOLD_PPB: return "mid"
    return "high"


def load_cube(bil_path):
    """Return cube as float32 reflectance (rows, cols, bands)."""
    cube = read_bil(bil_path)
    return cube.astype(np.float32) / REFLECTANCE_SCALE


def pistachio_mask(cube, band_idx=mask_band_idx, q=MASK_QUANTILE):
    """Simple intensity-threshold foreground mask."""
    band = cube[..., band_idx]
    thr = np.quantile(band, q)
    return band > thr


# ─── build sample table ─────────────────────────────────────────────────
print("\n[load] iterating cubes...")
rows = []
spectra = []
n_total = 0
for level_name in sorted(AFB1_PPB.keys()):
    level_dir = os.path.join(DATA_ROOT, level_name)
    if not os.path.isdir(level_dir):
        continue
    bils = sorted(glob.glob(os.path.join(level_dir, "*.bil")))
    if not bils:
        continue
    ppb = AFB1_PPB[level_name]
    zone = zone_of(ppb)
    is_unsafe = int(ppb > EU_AFB1_THRESHOLD_PPB)
    for bp in bils:
        cube = load_cube(bp)
        mask = pistachio_mask(cube)
        n_pistachio_pix = int(mask.sum())
        if n_pistachio_pix < 50:
            print(f"  [skip] {os.path.basename(bp)}: only {n_pistachio_pix} pistachio pix")
            continue
        # Random sample
        ys, xs = np.where(mask)
        n = min(PIX_PER_IMAGE, len(ys))
        idx = RNG.choice(len(ys), n, replace=False)
        spec = cube[ys[idx], xs[idx], :].astype(np.float32)
        for k in range(n):
            rows.append(dict(level=level_name, image=os.path.basename(bp),
                             AFB1_ppb=ppb, is_unsafe=is_unsafe, zone=zone))
        spectra.append(spec)
        n_total += n
    print(f"  {level_name}  AFB1={ppb:5.2f} ppb  zone={zone:4s}  "
          f"{len(bils)} images -> sampled {len([r for r in rows if r['level']==level_name])} pixels")

meta = pd.DataFrame(rows)
X = np.vstack(spectra)  # (n_total, 462)
print(f"\n[sample table] {X.shape[0]} pixel-samples, {X.shape[1]} bands")
print(f"               levels = {meta['level'].nunique()}, images = {meta['image'].nunique()}")
print(f"               zones: " + ", ".join(f"{z}={int((meta['zone']==z).sum())}" for z in ("low","mid","high")))
print(f"               unsafe (>{EU_AFB1_THRESHOLD_PPB} ppb): {meta['is_unsafe'].sum()}/{len(meta)}")

os.makedirs(RES, exist_ok=True)
np.save(os.path.join(RES, "pistachio_spectra.npy"), X)
meta.to_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t", index=False)

# ─── sg2 preprocess ─────────────────────────────────────────────────────
X_sg = _sg(X, 2)


def evaluate(y_true, pred, label=""):
    if len(y_true) < 5: return None
    resid = y_true - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y_true - y_true.mean())**2)
    rmse = np.sqrt(np.mean(resid**2))
    r = pearsonr(y_true, pred)[0]
    rho = spearmanr(y_true, pred)[0]
    truth = (y_true > np.quantile(y_true, 2/3)).astype(int) if len(np.unique(y_true)) > 3 else None
    try:
        auc = roc_auc_score(truth, pred) if truth is not None else np.nan
    except Exception:
        auc = np.nan
    return dict(label=label, n=len(y_true), R2=r2, RMSE=rmse,
                bias=float(np.mean(pred-y_true)), pearson_r=r,
                spearman_rho=rho, AUC_topT=auc)


def train_en(X_tr, y_tr):
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", ElasticNet(max_iter=20000))])
    gs = GridSearchCV(pipe,
                      {"model__alpha": np.logspace(-3, 1, 7),
                       "model__l1_ratio":[0.2, 0.5, 0.8]},
                      cv=KFold(5, shuffle=True, random_state=SEED),
                      scoring="r2", n_jobs=-1).fit(X_tr, y_tr)
    return gs


# ════════════════════════════════════════════════════════════════════════
# [A] Image-level random CV (in-sample upper bound)
# ════════════════════════════════════════════════════════════════════════
print("\n=== [A] Random 5-fold CV on full sample table ===")
y_all = meta["AFB1_ppb"].values
zones_all = meta["zone"].values
unsafe_all = meta["is_unsafe"].values

# random 5-fold (per-pixel, no group): this is the OPTIMISTIC ceiling
gs = train_en(X_sg, y_all)
pred_in = np.asarray(gs.predict(X_sg)).ravel()
m_in = evaluate(y_all, pred_in, "in-sample (optimistic)")
print(f"  in-sample (full)       : R2={m_in['R2']:+.3f} RMSE={m_in['RMSE']:.2f} r={m_in['pearson_r']:.3f} AUC={m_in['AUC_topT']:.3f}")
# zone breakdown
for z in ("low","mid","high"):
    sel = zones_all == z
    m = evaluate(y_all[sel], pred_in[sel], f"in-sample / {z} zone")
    print(f"  in-sample / {z:4s} zone   : n={m['n']}  R2={m['R2']:+.3f}  RMSE={m['RMSE']:.2f}  r={m['pearson_r']:.3f}  bias={m['bias']:+.2f}")


# ════════════════════════════════════════════════════════════════════════
# [B] LOIO — Leave-One-Image-Out (real cross-batch generalization)
# ════════════════════════════════════════════════════════════════════════
print("\n=== [B] LOIO: train on 56 images, predict held-out 57th ===")
loio_rows = []
images = meta["image"].unique()
print(f"  {len(images)} unique images")
preds_loio = np.full(len(meta), np.nan)
for img in images:
    te = (meta["image"].values == img)
    tr = ~te
    if te.sum() < 5: continue
    gs = train_en(X_sg[tr], y_all[tr])
    pred = np.asarray(gs.predict(X_sg[te])).ravel()
    preds_loio[te] = pred
# overall LOIO
valid = ~np.isnan(preds_loio)
m_loio = evaluate(y_all[valid], preds_loio[valid], "LOIO overall")
print(f"  LOIO overall           : n={m_loio['n']}  R2={m_loio['R2']:+.3f}  RMSE={m_loio['RMSE']:.2f}  r={m_loio['pearson_r']:.3f}  AUC={m_loio['AUC_topT']:.3f}")

# Zone breakdown for LOIO
for z in ("low","mid","high"):
    sel = valid & (zones_all == z)
    if sel.sum() < 5: continue
    m = evaluate(y_all[sel], preds_loio[sel], f"LOIO / {z}")
    print(f"  LOIO / {z:4s} zone        : n={m['n']}  R2={m['R2']:+.3f}  RMSE={m['RMSE']:.2f}  r={m['pearson_r']:.3f}  bias={m['bias']:+.2f}")


# ════════════════════════════════════════════════════════════════════════
# [C] Zone-restricted training: train only on one zone, predict whole range
# ════════════════════════════════════════════════════════════════════════
print("\n=== [C] Zone-restricted training (LOIO within zone, predict all) ===")
zone_rows = []
for train_zone in ("low","mid","high"):
    tr_zone_mask = (zones_all == train_zone)
    if tr_zone_mask.sum() < 50: continue
    # LOIO within the train zone
    preds_zone = np.full(len(meta), np.nan)
    images_in_zone = meta.loc[tr_zone_mask, "image"].unique()
    for img in images_in_zone:
        te_img = (meta["image"].values == img)
        tr = tr_zone_mask & (~te_img)
        te = te_img
        if tr.sum() < 20 or te.sum() < 5: continue
        gs = train_en(X_sg[tr], y_all[tr])
        pred = np.asarray(gs.predict(X_sg[te])).ravel()
        preds_zone[te] = pred
    # Also predict OTHER zones (without seeing them in training)
    other_imgs = meta.loc[~tr_zone_mask, "image"].unique()
    if len(other_imgs) > 0:
        gs_all = train_en(X_sg[tr_zone_mask], y_all[tr_zone_mask])
        for img in other_imgs:
            te = (meta["image"].values == img)
            preds_zone[te] = np.asarray(gs_all.predict(X_sg[te])).ravel()
    valid = ~np.isnan(preds_zone)
    for eval_zone in ("low","mid","high","all"):
        sel = valid if eval_zone == "all" else valid & (zones_all == eval_zone)
        if sel.sum() < 5: continue
        m = evaluate(y_all[sel], preds_zone[sel], f"train {train_zone} -> test {eval_zone}")
        m.update(train_zone=train_zone, test_zone=eval_zone)
        zone_rows.append(m)
        print(f"  train {train_zone:4s} -> test {eval_zone:4s}  n={m['n']:5d}  R2={m['R2']:+.3f}  RMSE={m['RMSE']:.2f}  r={m['pearson_r']:.3f}  bias={m['bias']:+.2f}")

zone_df = pd.DataFrame(zone_rows)
zone_df.to_csv(os.path.join(RES, "19_pistachio_zone.tsv"), sep="\t", index=False, float_format="%.4f")


# ════════════════════════════════════════════════════════════════════════
# Figures
# ════════════════════════════════════════════════════════════════════════
# Fig 1: AFB1 distribution + zone definitions
fig, ax = plt.subplots(1, 1, figsize=(9, 4))
ax.scatter(np.arange(len(WAVELENGTHS)), X_sg.mean(0), alpha=0)  # dummy
levels_sorted = sorted(AFB1_PPB.items(), key=lambda x: x[1])
xs = [name for name, _ in levels_sorted]; ys = [v for _, v in levels_sorted]
zones_color = ["#55A868" if zone_of(v) == "low" else ("#DD8452" if zone_of(v) == "mid" else "#C44E52")
               for v in ys]
ax2 = ax.twinx(); ax.clear()
ax.bar(range(len(xs)), ys, color=zones_color)
ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=45, fontsize=8)
ax.axhline(EU_AFB1_THRESHOLD_PPB, ls="--", color="black", label=f"EU threshold {EU_AFB1_THRESHOLD_PPB} ppb")
ax.set_ylabel("AFB1 (ppb / µg/kg)"); ax.legend(fontsize=9)
ax.set_title("HyperPistachio levels by AFB1 ppb (green=low, orange=mid, red=high)")
fig.tight_layout(); fig.savefig(os.path.join(RES, "FIG_pistachio_levels.png"), dpi=130)

# Fig 2: predicted vs true (LOIO), colored by zone
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
ax2[0].scatter(y_all, pred_in, c=zones_all == "high", cmap="coolwarm", s=4, alpha=0.4)
ax2[0].plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], "k--")
ax2[0].set_xlabel("true AFB1 (ppb)"); ax2[0].set_ylabel("predicted (in-sample)")
ax2[0].set_title(f"In-sample fit  R^2={m_in['R2']:.2f}")
ax2[0].set_xscale("symlog", linthresh=1)
valid = ~np.isnan(preds_loio)
zone_to_color = {"low":"#55A868","mid":"#DD8452","high":"#C44E52"}
for z in ("low","mid","high"):
    sel = valid & (zones_all == z)
    ax2[1].scatter(y_all[sel], preds_loio[sel], color=zone_to_color[z], s=4, alpha=0.4, label=z)
ax2[1].plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], "k--")
ax2[1].set_xlabel("true AFB1 (ppb)"); ax2[1].set_ylabel("predicted (LOIO)")
ax2[1].set_title(f"LOIO cross-image  R^2={m_loio['R2']:.2f}")
ax2[1].set_xscale("symlog", linthresh=1)
ax2[1].legend(fontsize=9)
fig2.tight_layout(); fig2.savefig(os.path.join(RES, "FIG_pistachio_LOIO_scatter.png"), dpi=130)


# ─── markdown summary ─────────────────────────────────────────────────────
md = ["# Phase 3.4 — HyperPistachio baseline + zone analysis", "",
      f"Samples: {X.shape[0]} pixel-spectra from {meta['image'].nunique()} images "
      f"({meta['level'].nunique()} AFB1 levels). {PIX_PER_IMAGE} random pistachio pixels per image.",
      f"AFB1 range: {y_all.min():.2f}-{y_all.max():.2f} ppb.",
      f"EU threshold: {EU_AFB1_THRESHOLD_PPB} ppb. Zones: low<2, mid<{EU_AFB1_THRESHOLD_PPB}, high>={EU_AFB1_THRESHOLD_PPB} ppb.", "",
      "## [A] In-sample (optimistic ceiling)",
      f"- overall: R^2={m_in['R2']:.3f}  r={m_in['pearson_r']:.3f}  RMSE={m_in['RMSE']:.2f} ppb", "",
      "## [B] LOIO — cross-image generalization (the fair test)",
      f"- overall LOIO: R^2={m_loio['R2']:+.3f}  r={m_loio['pearson_r']:.3f}  RMSE={m_loio['RMSE']:.2f} ppb  AUC={m_loio['AUC_topT']:.3f}", "",
      "## [C] Zone-restricted training (does training only on one concentration zone generalise?)",
      "| train zone | test zone | n | R^2 | RMSE | r | bias |",
      "|---|---|---|---|---|---|---|"]
for _, r in zone_df.iterrows():
    md.append(f"| {r['train_zone']} | {r['test_zone']} | {int(r['n'])} | {r['R2']:+.3f} | {r['RMSE']:.2f} | {r['pearson_r']:.3f} | {r['bias']:+.2f} |")
md += ["", "Raw: results/19_pistachio_zone.tsv, FIG_pistachio_LOIO_scatter.png, FIG_pistachio_levels.png"]
open(os.path.join(RES, "19_pistachio_results.md"), "w", encoding="utf-8").write("\n".join(md))
print("\n[written] 19_pistachio_results.md, 19_pistachio_zone.tsv, FIG_pistachio_*.png")
