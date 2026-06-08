"""
Phase 4.0 - Cross-batch generalization: train on v1 (Feb-2025), test on v3 (May-2026).

Same group, same camera, but different batch/contamination round. This is the
real generalization question for paper 2: do RF/GBM, which learn within-image
features (per script 29/30), actually transfer across acquisitions?

Pipeline:
  1. Load v1 sample table (already on disk: pistachio_spectra.npy + meta.tsv)
  2. Process v3 cubes (Zenodo doi 20027441) -> v3 spectra/meta with SG2
  3. Train Ridge / RF / GBM on FULL v1 (17,100 px) -> predict v3 pixels
  4. Report AUC@8, AUC@2 cross-batch vs in-domain GroupKFold (from scripts 24-30)

v3 specifics (from Zenodo metadata):
  - 26 levels (Level 01 = 0.00 healthy control through Level 26 = 114.67 ppb)
  - 2 samples per level => 26x2 = 52 .bil cubes
  - Naming: L##_####.bil, same Pika XC2 instrument, 462 bands 386.88-1003.60 nm
  - .hdr / .bil same ENVI BIL uint16 format
"""
import os, sys, glob, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import _sg
from pistachio_io import read_bil, REFLECTANCE_SCALE, EU_AFB1_THRESHOLD_PPB
warnings.filterwarnings("ignore")

RES = r"D:\bioinformatics\project_pistachio_AFB1\results"
V3_ROOT = r"D:\bioinformatics\project_pistachio_AFB1\data\pistachio_v3\Dataset"
PIX_PER_IMAGE = 300
MASK_BAND_NM = 650
MASK_QUANTILE = 0.55
SEED = 42

V3_AFB1_PPB = {
    "Level 01": 0.00, "Level 02": 0.40, "Level 03": 0.67, "Level 04": 0.88,
    "Level 05": 1.13, "Level 06": 1.66, "Level 07": 2.15, "Level 08": 2.30,
    "Level 09": 2.48, "Level 10": 2.82, "Level 11": 3.01, "Level 12": 3.05,
    "Level 13": 3.85, "Level 14": 4.43, "Level 15": 5.12, "Level 16": 5.30,
    "Level 17": 6.37, "Level 18": 8.93, "Level 19": 12.16, "Level 20": 17.12,
    "Level 21": 24.03, "Level 22": 26.14, "Level 23": 33.17, "Level 24": 56.06,
    "Level 25": 57.29, "Level 26": 114.67,
}


def auc_at(y_true, pred, thr):
    truth = (y_true > thr).astype(int)
    return roc_auc_score(truth, pred) if 0 < truth.sum() < len(truth) else np.nan


def score_pred(y_true, pred):
    resid = y_true - pred
    r2 = 1 - np.sum(resid**2) / np.sum((y_true - y_true.mean())**2)
    rmse = float(np.sqrt(np.mean(resid**2)))
    return dict(R2=r2, RMSE=rmse, r=pearsonr(y_true, pred)[0],
                AUC_8ppb=auc_at(y_true, pred, EU_AFB1_THRESHOLD_PPB),
                AUC_2ppb=auc_at(y_true, pred, 2.0))


print("[v1] loading saved sample table", flush=True)
X1_raw = np.load(os.path.join(RES, "pistachio_spectra.npy"))
meta1 = pd.read_csv(os.path.join(RES, "pistachio_meta.tsv"), sep="\t")
y1 = meta1["AFB1_ppb"].values
X1 = _sg(X1_raw.astype(float), 2)
print(f"[v1] {X1.shape} from {meta1['image'].nunique()} images, "
      f"{meta1['level'].nunique()} levels, ppb {y1.min():.2f}-{y1.max():.2f}",
      flush=True)


def find_v3_root():
    if os.path.isdir(V3_ROOT):
        return V3_ROOT
    candidates = [
        os.path.join(os.path.dirname(V3_ROOT), "extracted", "Dataset"),
        os.path.join(os.path.dirname(V3_ROOT), "Dataset"),
        os.path.dirname(V3_ROOT),
    ]
    for c in candidates:
        if os.path.isdir(c) and any(f.endswith(".bil") for root, _, files in os.walk(c) for f in files):
            return c
    raise FileNotFoundError(f"v3 cubes not found near {V3_ROOT}")


def first_bil(root):
    matches = glob.glob(os.path.join(root, "**", "*.bil"), recursive=True)
    if not matches:
        raise FileNotFoundError("no .bil under v3 root")
    return matches[0]


def parse_hdr_text(text):
    out = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def parse_hdr(hdr_path):
    text = open(hdr_path, "r", encoding="utf-8", errors="ignore").read()
    hdr = parse_hdr_text(text)
    return int(hdr["lines"]), int(hdr["samples"]), int(hdr["bands"])


def read_wavelengths(hdr_path):
    text = open(hdr_path, "r", encoding="utf-8", errors="ignore").read()
    s = text.find("wavelength = {") + len("wavelength = {")
    e = text.find("}", s)
    return np.array([float(v.strip()) for v in text[s:e].split(",")])


def process_v3():
    v3_root = find_v3_root()
    print(f"[v3] root = {v3_root}", flush=True)
    sample = first_bil(v3_root)
    wl = read_wavelengths(sample.replace(".bil", ".hdr"))
    print(f"[v3] wavelengths {wl[0]:.2f}-{wl[-1]:.2f} nm, n={len(wl)}", flush=True)
    assert len(wl) == 462, f"unexpected band count {len(wl)}"
    mask_band_idx = int(np.argmin(np.abs(wl - MASK_BAND_NM)))

    rng = np.random.default_rng(SEED)
    rows, spectra = [], []
    for level_name, ppb in sorted(V3_AFB1_PPB.items()):
        # try both "Level 01" and "Level_01"
        for variant in (level_name, level_name.replace(" ", "_")):
            lvl_dir = os.path.join(v3_root, variant)
            if os.path.isdir(lvl_dir): break
        bils = sorted(glob.glob(os.path.join(lvl_dir, "*.bil")))
        if not bils:
            print(f"  [skip] {level_name}: no bil under {lvl_dir}", flush=True)
            continue
        zone = ("low" if ppb < 2 else ("mid" if ppb < EU_AFB1_THRESHOLD_PPB else "high"))
        for bp in bils:
            hdr_path = bp.replace(".bil", ".hdr")
            try:
                rows_, cols_, bands_ = parse_hdr(hdr_path)
                cube = read_bil(bp, rows=rows_, cols=cols_, bands=bands_)
            except Exception as e:
                print(f"  [error] {os.path.basename(bp)}: {e}", flush=True)
                continue
            cube = cube.astype(np.float32) / REFLECTANCE_SCALE
            band = cube[..., mask_band_idx]
            thr = float(np.quantile(band, MASK_QUANTILE))
            mask = band > thr
            if mask.sum() < 50:
                continue
            ys, xs = np.where(mask)
            n = min(PIX_PER_IMAGE, len(ys))
            idx = rng.choice(len(ys), n, replace=False)
            spec = cube[ys[idx], xs[idx], :].astype(np.float32)
            for _ in range(n):
                rows.append(dict(level=level_name,
                                 image=os.path.basename(bp),
                                 AFB1_ppb=float(ppb),
                                 is_unsafe=int(ppb > EU_AFB1_THRESHOLD_PPB),
                                 zone=zone))
            spectra.append(spec)
        print(f"  [{level_name}] AFB1={ppb:7.2f}  imgs={len(bils)}", flush=True)
    return np.vstack(spectra), pd.DataFrame(rows)


print("[v3] processing cubes", flush=True)
t0 = time.time()
X3_raw, meta3 = process_v3()
print(f"[v3] {X3_raw.shape} from {meta3['image'].nunique()} images, "
      f"{meta3['level'].nunique()} levels in {time.time()-t0:.0f}s", flush=True)
np.save(os.path.join(RES, "pistachio_v3_spectra.npy"), X3_raw)
meta3.to_csv(os.path.join(RES, "pistachio_v3_meta.tsv"), sep="\t", index=False)
X3 = _sg(X3_raw.astype(float), 2)
y3 = meta3["AFB1_ppb"].values
zones3 = meta3["zone"].values
print(f"[v3] ppb range {y3.min():.2f}-{y3.max():.2f}", flush=True)


def fit_then_predict(model_factory, tag):
    print(f"\n=== {tag}: fit v1, score v3 ===", flush=True)
    t0 = time.time()
    m = model_factory()
    m.fit(X1, y1)
    pred3 = m.predict(X3)
    print(f"  [{tag}] fit+predict in {time.time()-t0:.1f}s", flush=True)
    return pred3


MODELS = [
    ("Ridge",
     lambda: Pipeline([("scale", StandardScaler()),
                       ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13)))])),
    ("RF_n200_d15",
     lambda: RandomForestRegressor(n_estimators=200, max_depth=15,
                                   min_samples_leaf=10, max_features="sqrt",
                                   n_jobs=4, random_state=42)),
    ("GBM_iter200_d8",
     lambda: HistGradientBoostingRegressor(max_iter=200, max_depth=8,
                                            learning_rate=0.1,
                                            min_samples_leaf=20, random_state=42)),
]

rows = []
preds_keep = {}
for tag, factory in MODELS:
    pred3 = fit_then_predict(factory, tag)
    preds_keep[tag] = pred3
    m_full = score_pred(y3, pred3)
    row = dict(case=tag, scope="v3 ALL (incl v3-only high)", n=len(y3), **m_full)
    for z in ("low","mid","high"):
        sel = zones3 == z
        row[f"r_{z}"] = score_pred(y3[sel], pred3[sel])["r"] if sel.sum() >= 5 else np.nan
    rows.append(row)
    print(f"  [{tag}] v3 ALL: R^2={m_full['R2']:+.3f} "
          f"AUC@8={m_full['AUC_8ppb']:.3f} AUC@2={m_full['AUC_2ppb']:.3f}",
          flush=True)
    overlap = y3 <= 33.17
    m_o = score_pred(y3[overlap], pred3[overlap])
    row_o = dict(case=tag, scope="v3 OVERLAP (<=33 ppb)", n=int(overlap.sum()), **m_o)
    for z in ("low","mid","high"):
        sel = (zones3 == z) & overlap
        row_o[f"r_{z}"] = score_pred(y3[sel], pred3[sel])["r"] if sel.sum() >= 5 else np.nan
    rows.append(row_o)
    print(f"  [{tag}] v3 OVERLAP: R^2={m_o['R2']:+.3f} "
          f"AUC@8={m_o['AUC_8ppb']:.3f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(RES, "32_cross_batch_v1_to_v3.tsv"), sep="\t", index=False)
for tag, pred in preds_keep.items():
    np.save(os.path.join(RES, f"32_pred_v3_{tag}.npy"), pred)

in_domain = {
    "Ridge":           dict(R2=0.337, AUC8=0.796, AUC2=0.570),
    "RF_n200_d15":     dict(R2=0.455, AUC8=0.899, AUC2=0.576),
    "GBM_iter200_d8":  dict(R2=0.539, AUC8=0.925, AUC2=0.572),
}

md = [
    "# Phase 4.0 - Cross-batch test: train v1, predict v3",
    "",
    "**The paper-2 watershed test**. Same team, same Pika XC2, same protocol, "
    "different batch (Zenodo v3, May 2026), concentrations extended to 114.67 ppb. "
    "If RF/GBM truly learn within-image AFB1 features (per scripts 29/30), AUC@8 "
    "should hold up across batch. Collapse vs hold-up determines paper-2 framing.",
    "",
    f"**v1 train**: {X1.shape[0]} pixels, ppb 0.40-33.17. "
    f"**v3 test**: {X3.shape[0]} pixels, ppb {y3.min():.2f}-{y3.max():.2f}.",
    "",
    "## In-domain (v1 GroupKFold) vs cross-batch (v1 -> v3)",
    "",
    "| model | scope | n | R^2 | r | RMSE | AUC@8 | AUC@2 | r_low | r_mid | r_high |",
    "|---|---|---|---|---|---|---|---|---|---|---|",
]
for tag in ["Ridge", "RF_n200_d15", "GBM_iter200_d8"]:
    ref = in_domain[tag]
    md.append(f"| {tag} | v1 GKF (in-domain) | 17100 | {ref['R2']:+.3f} | - | - | "
              f"{ref['AUC8']:.3f} | {ref['AUC2']:.3f} | - | - | - |")
    for _, r_ in df[df["case"]==tag].iterrows():
        md.append(f"| {tag} | {r_['scope']} | {int(r_['n'])} | "
                  f"{r_['R2']:+.3f} | {r_['r']:+.3f} | {r_['RMSE']:.2f} | "
                  f"{r_['AUC_8ppb']:.3f} | {r_['AUC_2ppb']:.3f} | "
                  f"{r_['r_low']:+.3f} | {r_['r_mid']:+.3f} | {r_['r_high']:+.3f} |")

md += [
    "",
    "## Verdict guide",
    "",
    "- AUC@8 cross-batch within 0.05 of in-domain -> within-image AFB1 signal transfers; paper claim of generalisable HSI prediction holds.",
    "- AUC@8 cross-batch 0.6-0.7 (drop -0.2 to -0.3) -> partial transfer; paper must report a 'limited cross-batch capability' caveat.",
    "- AUC@8 cross-batch ~0.5 (collapse) -> within-image features themselves are batch-specific; deployment needs domain adaptation / per-batch recalibration.",
    "",
    "Compare Ridge vs RF/GBM cross-batch: if Ridge collapses while RF/GBM transfer, the residual-regression diagnosis is fully confirmed for the cross-batch setting. If both collapse equally, 'Ridge learns baseline, RF learns kernel features' framing needs revision.",
    "",
    "## Outputs",
    "- `results/32_cross_batch_v1_to_v3.tsv`",
    "- `results/pistachio_v3_spectra.npy` + `pistachio_v3_meta.tsv`",
    "- `results/32_pred_v3_*.npy` (per model)",
]
with open(os.path.join(RES, "32_cross_batch_v1_to_v3.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print(f"\n[write] results/32_cross_batch_v1_to_v3.{{tsv,md}}", flush=True)
print(f"[write] results/pistachio_v3_{{spectra.npy,meta.tsv}} (reusable)", flush=True)
print("DONE", flush=True)
