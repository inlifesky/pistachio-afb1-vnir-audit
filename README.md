# Cross-batch validation of VNIR hyperspectral pre-screening for pistachio aflatoxin B1

Reproducibility repository for a manuscript under review.

This repository contains the analysis pipeline that reproduces every number, table, and figure in the manuscript and its supplement. The underlying hyperspectral data are openly deposited at Zenodo under a no-derivatives licence and are **not** redistributed here; see [Data](#data) below for download instructions.

---

## What this paper audits

VNIR (~400–1000 nm) hyperspectral imaging combined with machine learning is widely proposed for non-destructive aflatoxin pre-screening of tree nuts. Reported accuracies on coarse multi-class tasks are high, but two questions central to deployment are rarely answered: (a) whether such models generalise across acquisition batches at the regulatory threshold; (b) whether a low-cost 5–10 band multispectral camera can substitute for full HSI. This study answers both on the HyperPistachio dataset using a deliberately held-out, same-instrument second batch as a cross-batch test set.

The audit reports three layers of evidence, each with 95 % bootstrap confidence intervals over the 52 cross-batch cubes:

1. **Preprocessing-driven cross-batch transfer.** Ridge + SG2 achieves in-domain per-cube AUC 0.984 at the 8 µg/kg EU threshold but falls cross-batch to 0.698 (95 % CI 0.515–0.860); an image-mean baseline reaches a comparable 0.639, showing that under SG2 ridge transfers little beyond per-image baseline information which itself drifts batch-to-batch. Adding SNV lifts the same ridge to a cross-batch per-cube AUC of **0.971 (95 % CI 0.920–1.000)**, a paired Δ-AUC = 0.273 (95 % CI 0.118–0.453, *P*(Δ > 0) = 1.000). Gradient boosting *fails to transfer* under SNV (per-cube AUC 0.289, 95 % CI 0.111–0.477); the partial-dependence direction flips on 5 of 10 top-|β| bands.
2. **Boundary finding on sparse multispectral substitution.** Five-to-ten band VNIR multispectral substitution is **not supported** at the regulatory threshold on this dataset under any of three selection strategies (supervised in-fold |β|, unsupervised orthogonal-subspace, uniform). AUC@8 ≥ 0.50 requires *k* ≥ 100 bands; AUC@8 ≥ 0.80 requires the full 462.
3. **Per-cube analytical metrics and cost analysis.** Pixel outputs are aggregated to per-cube decisions (FPR 0.088 at 100 % per-cube recall, 95 % CI 0.000–0.176); a three-tier risk matrix under explicit NPV ≥ 0.95 / PPV ≥ 0.90 rules; and a scenario-analysis cost model that is favourable across most plausible HPLC and recall-liability cost cells but **not all**.

The methodological contribution is an **image-mean transfer-risk diagnostic** — a residual-regression test paired with a direct image-mean baseline — that flags cross-batch failure before any new-batch test is run. The diagnostic is reusable for any label-uniform-within-image food-safety acquisition.

---

## Data

The spectral data are openly deposited at Zenodo:

- **Training batch (v1).** Sheikh-Akbari, A., & Mehrabinejad, H. (2025). *HyperPistachio: A Hyperspectral Image Dataset of Aflatoxin B1 Contaminated Pistachio Nuts* (Version 1.0) [Data set]. Zenodo. doi:[10.5281/zenodo.16920712](https://doi.org/10.5281/zenodo.16920712).
- **Cross-batch test batch (v3).** Sheikh-Akbari, A., & Mehrabinejad, H. (2026). *HyperPistachio: A Hyperspectral Image Dataset of Aflatoxin B1 Contaminated Pistachio Nuts* (Version 3) [Data set]. Zenodo. doi:[10.5281/zenodo.20027441](https://doi.org/10.5281/zenodo.20027441).

Both releases are distributed under **Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)**. CC BY-ND permits reuse with attribution but **explicitly prohibits redistribution of modified or derivative datasets** — including extracted pixel arrays. Accordingly, this repository does not ship either the raw cubes or the extracted pixel TSVs. Run the pipeline locally after downloading from Zenodo.

Each release contains a set of reflectance-calibrated hyperspectral cubes (band-interleaved-by-line, uint16, 256 × 384 pixels × 462 bands, 386.88–1003.60 nm, reflectance scale factor 10,000), with AFB1 ground truth labelled per concentration level. Two v1 cubes whose files could not be read are excluded, leaving **57 v1 cubes (training) and 52 v3 cubes (cross-batch test); n = 109 cubes total**.

Each archive unzips to a `Dataset/` folder containing one subfolder per contamination level, with `.bil` cubes inside. A suggested layout is:

```
data/
├── v1/                              # unzipped doi:10.5281/zenodo.16920712
│   └── Dataset/
│       ├── Level_01/ ... Level_22/  # .bil cubes per level (22 levels × 3 imaging replicates)
└── v3/                              # unzipped doi:10.5281/zenodo.20027441
    └── Dataset/
        ├── Level_01/ ... Level_26/  # .bil cubes per level (26 levels × 2 imaging replicates)
```

AFB1 ground-truth concentrations are not in the archives — they are hard-coded as a dictionary (`AFB1_PPB`) in `scripts/pistachio_io.py`, keyed by level number. The 8 µg/kg EU regulatory threshold is the constant `EU_AFB1_THRESHOLD_PPB` in the same file.

The PISTACHIO_V1_DATA and PISTACHIO_V3_DATA environment variables (see [Environment variables](#3-environment-variables) below) should point at the two `Dataset/` folders, wherever you put them on disk.

---

## Reproduction

### 1. Environment

```bash
# Option A: conda
conda env create -f environment.yml
conda activate pistachio-afb1-audit

# Option B: pip + venv
python -m venv venv
source venv/bin/activate                # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data

Download the two Zenodo archives (links above) and unzip them somewhere on disk. Each archive contains a `Dataset/` folder with one subfolder per contamination level holding the `.bil` cubes.

### 3. Environment variables

The scripts read three paths from environment variables so the repo works without modification on any machine. Set them once before running the pipeline:

| Variable | Required for | What it points at |
|---|---|---|
| `PISTACHIO_V1_DATA` | scripts 19, 26, 32, 38, 40, 43 (anything that reads v1 cubes) | the unzipped v1 `Dataset/` folder (Zenodo doi:10.5281/zenodo.16920712) |
| `PISTACHIO_V3_DATA` | scripts 32, 38, 43 (anything that reads v3 cubes) | the unzipped v3 `Dataset/` folder (Zenodo doi:10.5281/zenodo.20027441) |
| `PISTACHIO_RES` (optional) | all scripts | output directory for TSVs and figures. Defaults to `./results` when unset, so the cloned repo's `results/` folder is used. |
| `PISTACHIO_FIG` (optional) | scripts 49 (figure generation) | figure output directory. Defaults to `$PISTACHIO_RES/figures`. |

Example (bash / WSL):

```bash
export PISTACHIO_V1_DATA="/path/to/unzipped/zenodo_v1/Dataset"
export PISTACHIO_V3_DATA="/path/to/unzipped/zenodo_v3/Dataset"
# PISTACHIO_RES defaults to ./results
```

Example (PowerShell):

```powershell
$env:PISTACHIO_V1_DATA = "D:\path\to\unzipped\zenodo_v1\Dataset"
$env:PISTACHIO_V3_DATA = "D:\path\to\unzipped\zenodo_v3\Dataset"
```

A script will raise `KeyError: 'PISTACHIO_V1_DATA'` (or `PISTACHIO_V3_DATA`) if the required variable is missing — set the variable and re-run.

### 4. Run

Each script writes a Markdown summary alongside its TSV outputs in `$PISTACHIO_RES` (default `./results`). Scripts are independent and idempotent; re-running a single script does not require re-running upstream ones.

```bash
# Cube I/O + foreground masking + per-cube random pixel draw (deterministic, seed=42)
# Produces v1_pixels.tsv and v3_pixels.tsv (17,100 and 15,600 rows respectively).
python scripts/19_pistachio_load_and_baseline.py

# §3.2 band-count sweep (Table S5; Figure 2)
python scripts/26_pistachio_band_sweep.py

# §3.3 residual-collapse diagnostic + image-mean baseline (Figure 3)
python scripts/29_confound_regression.py
python scripts/42_image_mean_baseline.py

# §3.4 forward cross-batch v1 → v3 under SG2
python scripts/32_cross_batch_v1_to_v3.py

# §3.5 recommended pipeline (Ridge + SG2+SNV cross-batch) — main text headline (0.971)
python scripts/37_ridge_snv_industrial.py

# §3.5 alternative-transform comparison (per-image, CORAL)
python scripts/36_domain_adaptation.py

# §3.6 industrial metrics (three-tier matrix) + calibration (S1, S2)
python scripts/33_industrial_metrics.py
python scripts/34_pr_auc_calibration_lot.py

# §3.7 healthy-bias mitigation grid (Table S3, Figure S3)
python scripts/39_healthy_bias_mitigation.py

# §3.8 reverse cross-batch v3 → v1
python scripts/38_reverse_cross_batch.py

# §3.9 cost-benefit model (Figure 7, Table S9)
python scripts/35_roi_model.py

# §3.10 sampling-sensitivity sweep (Table S6)
python scripts/43_sampling_sensitivity.py

# §4.2 GBM PDP diagnostic under SG2 vs SG2+SNV (Figure 5, Table S8)
python scripts/40_gbm_snv_pdp_diagnostic.py

# Bootstrap 95 % CIs for cube-level metrics (main text Table 2, Table S4)
python scripts/41_bootstrap_lot_CIs.py
python scripts/45_gbm_snv_bootstrap.py

# Figure generation (publication-style PNG + PDF; figures/*.png and figures/*.pdf)
python scripts/49_figures_paper2_v4.py
```

---

## Repository structure

```
.
├── README.md                          ← this file
├── LICENSE                            ← MIT for code; data are not redistributed
├── environment.yml                    ← conda environment lock
├── requirements.txt                   ← pip alternative
├── .gitignore                         ← excludes data/*, results/*, __pycache__
├── data/                              ← (empty; user downloads from Zenodo)
├── scripts/                           ← analysis pipeline
│   ├── preprocessing.py               ← SG / SNV / Standard-scale + utilities
│   ├── pistachio_io.py                ← reflectance-calibrated BIL cube reader
│   ├── 19_pistachio_load_and_baseline.py
│   ├── 26_pistachio_band_sweep.py
│   ├── 29_confound_regression.py
│   ├── 32_cross_batch_v1_to_v3.py
│   ├── 33_industrial_metrics.py
│   ├── 34_pr_auc_calibration_lot.py
│   ├── 35_roi_model.py
│   ├── 36_domain_adaptation.py
│   ├── 37_ridge_snv_industrial.py
│   ├── 38_reverse_cross_batch.py
│   ├── 39_healthy_bias_mitigation.py
│   ├── 40_gbm_snv_pdp_diagnostic.py
│   ├── 41_bootstrap_lot_CIs.py
│   ├── 42_image_mean_baseline.py
│   ├── 43_sampling_sensitivity.py
│   ├── 45_gbm_snv_bootstrap.py
│   └── 49_figures_paper2_v4.py
├── results/                           ← outputs (regenerated by re-running scripts)
└── logs/                              ← number verification log + download provenance
```

**This repository contains only the scripts that reproduce the manuscript.** Many exploratory analyses (alternative model architectures, multitask attempts, mixture models, an external candidate-dataset comparison) were run during development and are omitted here to keep the reproduction pipeline self-contained. Every number, table, and figure in the manuscript and its supplement is produced by the scripts listed above.

---

## What you should see when reproduction succeeds

Headline numbers to match (within rounding from SEED = 42):

| Script | Headline | Expected value |
|---|---|---|
| `19_pistachio_load_and_baseline.py` | v1 / v3 pixel count | 17,100 / 15,600 |
| `19_pistachio_load_and_baseline.py` | v1 / v3 cube count | 57 / 52 |
| `26_pistachio_band_sweep.py` | All-462 AUC@8 in-domain | 0.796 |
| `26_pistachio_band_sweep.py` | top-5 supervised AUC@8 | 0.449 |
| `32_cross_batch_v1_to_v3.py` | Ridge SG2 cross-batch cube AUC@8 | 0.698 |
| `32_cross_batch_v1_to_v3.py` | GBM SG2 cross-batch cube AUC@8 | 0.935 |
| `37_ridge_snv_industrial.py` | Ridge SNV cross-batch cube AUC@8 | 0.971 |
| `41_bootstrap_lot_CIs.py` | Ridge SNV cross-batch cube AUC@8 95 % CI | [0.920, 1.000] |
| `41_bootstrap_lot_CIs.py` | Paired Δ-AUC (SNV − SG2) | 0.273, [0.118, 0.453] |
| `42_image_mean_baseline.py` | Image-mean Ridge SG2 cross-batch | 0.639 |
| `45_gbm_snv_bootstrap.py` | GBM SNV cross-batch cube AUC@8 95 % CI | [0.111, 0.477] |
| `43_sampling_sensitivity.py` | Baseline cell seed-averaged AUC@8 | 0.962 |

If any of these are off by more than the bootstrap variance reported in the manuscript, please open an issue.

---

## Citation

If you use this code, please cite the source dataset:

- Sheikh-Akbari, A., & Mehrabinejad, H. (2025). *HyperPistachio: A Hyperspectral Image Dataset of Aflatoxin B1 Contaminated Pistachio Nuts* (Version 1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.16920712
- Sheikh-Akbari, A., & Mehrabinejad, H. (2026). *HyperPistachio: A Hyperspectral Image Dataset of Aflatoxin B1 Contaminated Pistachio Nuts* (Version 3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.20027441

A full citation for this work will be added on acceptance.

---

## License

Code is released under the [MIT License](LICENSE). The underlying HyperPistachio dataset is released under Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0); users should download it directly from Zenodo and comply with that licence.

---

## Contact

Wei Yuan — rita.w.yuan@gmail.com — ORCID: [0009-0009-4139-7802](https://orcid.org/0009-0009-4139-7802)
