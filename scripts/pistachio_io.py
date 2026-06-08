"""
HyperPistachio dataset I/O.

Each sample is a hyperspectral cube stored as a .bil file (Band Interleaved by Line),
uint16, with shape (rows=2400, cols=1600, bands=462). Per the dataset's MATLAB script:
    img = multibandread('1a.bil', [2400 1600 462], 'uint16', 0, 'bil', 'ieee-le')
which is little-endian uint16, no header offset, BIL interleave.

We read these directly with numpy memmap to keep memory bounded (each cube is ~3.5 GB
uncompressed). For modelling we don't need the full pixel grid — we extract the ROI
and downsample to a manageable representation.

Conventions:
    rows × cols × bands  -> reshape to (rows*cols, bands) for sklearn-style modelling
    ROI from the MATLAB script: rows 291..1314, cols 360..1383  (1024 × 1024)
"""
import numpy as np
import os

# Real specs read from .hdr files (the MATLAB script's 2400x1600 was the un-
# downsampled ROI grid; the deposited .bil files are already at 256x384).
ROWS = 256        # ENVI 'lines'
COLS = 384        # ENVI 'samples'
BANDS = 462
REFLECTANCE_SCALE = 10000.0   # uint16 value / scale = true reflectance
# Use full image; pistachio kernels occupy most of the frame at this size.
ROI_ROWS = slice(0, ROWS)
ROI_COLS = slice(0, COLS)

# wavelengths in nm (copied from Read_HS_Image.m)
WAVELENGTHS = np.array([386.88, 388.19, 389.5, 390.81, 392.12, 393.43, 394.74, 396.05])  # placeholder — full list filled in load()


def load_wavelengths_from_script(script_path):
    """Parse the wavelength array out of Read_HS_Image.m once, for reference."""
    text = open(script_path, "r", encoding="utf-8").read()
    # find the bracketed list after 'wavelength ='
    start = text.find("wavelength = [")
    end = text.find("];", start)
    body = text[start + len("wavelength = ["):end]
    return np.array([float(v) for v in body.split()])


def read_bil(bil_path, rows=ROWS, cols=COLS, bands=BANDS, dtype=np.uint16):
    """Load a BIL hyperspectral cube as (rows, cols, bands) numpy array (memmap)."""
    # BIL layout: for each row, store all bands consecutively (band 1 row, band 2 row, ...).
    # Element count per row = cols * bands.
    # numpy memmap order: (rows, bands, cols)  then transpose to (rows, cols, bands).
    mm = np.memmap(bil_path, dtype=dtype, mode="r", shape=(rows, bands, cols))
    # transpose to (rows, cols, bands)
    return np.transpose(mm, (0, 2, 1))


def extract_mean_spectrum(cube_rcb, roi_mask=None):
    """
    Compute the mean spectrum over a ROI (or over all pixels if roi_mask is None).

    cube_rcb : (rows, cols, bands) array
    roi_mask : (rows, cols) bool array, True = include pixel.

    Returns a length-bands float64 spectrum.
    """
    if roi_mask is None:
        flat = cube_rcb.reshape(-1, cube_rcb.shape[-1]).astype(np.float64)
        return flat.mean(0)
    sel = cube_rcb[roi_mask].astype(np.float64)
    return sel.mean(0)


def simple_pistachio_mask(cube_rcb, ref_band_idx=200, threshold_quantile=0.6):
    """
    Crude pistachio-vs-background mask. Hyperspectral images typically contain
    pistachio kernels on a darker background; pixels with reflectance above a
    threshold (at a chosen band in the visible/NIR) are kept.

    Returns a (rows, cols) bool mask.
    """
    band = cube_rcb[..., ref_band_idx].astype(np.float64)
    thr = np.quantile(band, threshold_quantile)
    return band > thr


# AFB1 contamination level table from Read_Subset_Pistachio.txt
AFB1_PPB = {
    "Level 00": 0.00,
    "Level 01": 0.40,
    "Level 02": 0.67,
    "Level 03": 0.88,
    "Level 04": 1.13,
    "Level 05": 1.66,
    "Level 06": 2.48,
    "Level 07": 2.15,
    "Level 08": 2.30,
    "Level 09": 3.05,
    "Level 10": 2.82,
    "Level 11": 3.01,
    "Level 12": 3.85,
    "Level 13": 4.43,
    "Level 14": 5.30,
    "Level 15": 5.12,
    "Level 16": 8.93,
    "Level 17": 6.37,
    "Level 18": 26.14,
    "Level 19": 33.17,
}

# EU regulatory threshold for AFB1 in pistachios intended for direct consumption: 8 µg/kg
# EU 1881/2006 / Reg (EU) 2023/915
EU_AFB1_THRESHOLD_PPB = 8.0


def get_label(level_name):
    """Return (afb1_ppb, is_unsafe) for a level name like 'Level 03'."""
    ppb = AFB1_PPB[level_name]
    return ppb, ppb > EU_AFB1_THRESHOLD_PPB
