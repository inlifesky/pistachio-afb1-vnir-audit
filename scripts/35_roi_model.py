"""
Phase 4.3 - ROI cost-benefit model for paper 2 industrial framing.

Three deployment scenarios with explicit cost assumptions:

  Scenario A: Status quo, HPLC every lot
    Cost = N_lots * HPLC_cost

  Scenario B: HSI pre-screening with GBM, lab confirmation for high+mid risk
    Cost = HSI_amortised + N_lots * (high_pct + mid_pct) * HPLC_cost
    Risk = N_lots * low_pct * P(unsafe | predicted low) * liability_per_unsafe_lot

  Scenario C: No screening
    Cost = 0 upfront
    Risk = N_lots * base_unsafe_rate * liability_per_unsafe_lot

All cost parameters are USER-CONFIGURABLE; defaults sourced from public industry
references where available, otherwise marked ASSUMPTION. The point is the
framework, not the exact numbers.
"""
import os
import numpy as np, pandas as pd

RES = r"D:\bioinformatics\project_pistachio_AFB1\results"

# ============================================================
# Cost parameters (defaults are illustrative — change per scenario)
# ============================================================
HPLC_COST_PER_LOT_USD = 100.0   # ASSUMPTION; ranges 80-200 USD per HPLC-FLD AFB1 test
HSI_AMORTISED_PER_YEAR_USD = 8000.0  # ASSUMPTION: 35-60k USD camera over 5-7 yr + operator
ANNUAL_LOT_VOLUMES = [1000, 5000, 20000, 100000]   # small, medium, large processor scale
BASE_UNSAFE_RATE = 0.05  # ASSUMPTION: 5% of natural pistachio lots exceed 8 ppb AFB1
LIABILITY_PER_UNSAFE_LOT_USD = 5000.0  # ASSUMPTION: recall/disposal/reputation

# From scripts 33/34, but converted to production-base-rate-agnostic form.
# What we trust: lot-level sensitivity (recall of unsafe) and specificity at the
# chosen operating point. These are intrinsic to the model, not to dataset
# base-rate.
#
# At GBM lot-level mean aggregator @ 8 ppb threshold (script 34c):
#   in-domain   ROC-AUC=0.977, FPR@95recall=0.083  -> sens=0.95, spec=0.917
#   cross-batch ROC-AUC=0.935, FPR@95recall=0.118  -> sens=0.95, spec=0.882

GBM_CROSS_BATCH = dict(
    label="GBM cross-batch lot-level mean @ 8 ppb (script 34c baseline)",
    sensitivity=0.95,     # by construction at recall@95 operating point
    specificity=0.882,    # 1 - FPR
)
GBM_IN_DOMAIN = dict(
    label="GBM in-domain lot-level mean @ 8 ppb (script 34c baseline)",
    sensitivity=0.95,
    specificity=0.917,
)
# Updated 2026-06-03: Ridge+SNV is the new cross-batch champion (script 37b).
# Cross-batch lot @ 8 ppb: achieved recall=1.000, FPR=0.088, spec=0.912
# In-domain lot @ 8 ppb: achieved recall=1.000, FPR=0.083, spec=0.917
RIDGE_SNV_CROSS_BATCH = dict(
    label="Ridge+SNV cross-batch lot-level @ 8 ppb (NEW RECOMMENDED)",
    sensitivity=0.95,
    specificity=0.912,
)
RIDGE_SNV_IN_DOMAIN = dict(
    label="Ridge+SNV in-domain lot-level @ 8 ppb",
    sensitivity=0.95,
    specificity=0.917,
)


def annual_costs(n_lots, scenario_label, params=None, hplc_cost=HPLC_COST_PER_LOT_USD,
                 hsi_amort=HSI_AMORTISED_PER_YEAR_USD,
                 base_unsafe_rate=BASE_UNSAFE_RATE,
                 liability=LIABILITY_PER_UNSAFE_LOT_USD):
    """Return cost dict for one scenario at one annual volume."""
    if scenario_label == "A_status_quo":
        hplc = n_lots * hplc_cost
        risk = 0.0  # assume HPLC catches all
        return dict(scenario="A status-quo (HPLC every lot)",
                    n_lots=n_lots,
                    HSI_amort_usd=0, HPLC_usd=hplc, liability_usd=risk,
                    total_usd=hplc + risk)

    if scenario_label == "B_HSI_screening":
        # Production base-rate-aware decomposition:
        #   true unsafe in 1 yr           = n_lots * base_unsafe_rate
        #   true safe                     = n_lots * (1 - base_unsafe_rate)
        # Model flags as positive (mid+high tiers, sent to HPLC):
        #   true-positive flags           = n_unsafe * sensitivity
        #   false-positive flags          = n_safe * (1 - specificity)
        # Model says LOW (released w/o lab):
        #   false-negative releases       = n_unsafe * (1 - sensitivity)
        #   true-negative releases        = n_safe * specificity
        # HPLC cost: pay for every flagged lot (TP + FP)
        # Liability: only false negatives released
        n_unsafe = n_lots * base_unsafe_rate
        n_safe = n_lots * (1 - base_unsafe_rate)
        n_flagged = n_unsafe * params["sensitivity"] + n_safe * (1 - params["specificity"])
        n_released_unsafe = n_unsafe * (1 - params["sensitivity"])
        hplc = n_flagged * hplc_cost
        risk = n_released_unsafe * liability
        return dict(scenario=f"B HSI screen ({params['label']})",
                    n_lots=n_lots, HSI_amort_usd=hsi_amort, HPLC_usd=hplc,
                    n_lots_sent_to_lab=int(n_flagged),
                    n_unsafe_released=n_released_unsafe,
                    liability_usd=risk,
                    total_usd=hsi_amort + hplc + risk)

    if scenario_label == "C_no_screening":
        unsafe = n_lots * base_unsafe_rate
        risk = unsafe * liability
        return dict(scenario="C no screening (all released)",
                    n_lots=n_lots, HSI_amort_usd=0, HPLC_usd=0,
                    n_unsafe_released=unsafe,
                    liability_usd=risk, total_usd=risk)

    raise ValueError(scenario_label)


# ============================================================
# Scenario sweep
# ============================================================
rows = []
for n_lots in ANNUAL_LOT_VOLUMES:
    rows.append(annual_costs(n_lots, "A_status_quo"))
    rows.append(annual_costs(n_lots, "B_HSI_screening", params=RIDGE_SNV_IN_DOMAIN))
    rows.append(annual_costs(n_lots, "B_HSI_screening", params=RIDGE_SNV_CROSS_BATCH))
    rows.append(annual_costs(n_lots, "B_HSI_screening", params=GBM_IN_DOMAIN))
    rows.append(annual_costs(n_lots, "B_HSI_screening", params=GBM_CROSS_BATCH))
    rows.append(annual_costs(n_lots, "C_no_screening"))

df = pd.DataFrame(rows).fillna(0)
df.to_csv(os.path.join(RES, "35_roi.tsv"), sep="\t", index=False)

# Per-scale comparison table (which scenario is cheapest)
rows_compare = []
for n_lots in ANNUAL_LOT_VOLUMES:
    sub = df[df["n_lots"] == n_lots]
    cheapest_idx = sub["total_usd"].idxmin()
    cheapest = df.loc[cheapest_idx, "scenario"]
    base_a = sub[sub["scenario"].str.startswith("A ")]["total_usd"].iloc[0]
    for _, r in sub.iterrows():
        rows_compare.append(dict(n_lots=n_lots, scenario=r["scenario"],
                                 total_usd=r["total_usd"],
                                 saving_vs_A_usd=base_a - r["total_usd"],
                                 saving_vs_A_pct=(base_a - r["total_usd"])/base_a*100
                                                 if base_a > 0 else 0.0,
                                 cheapest=(r["scenario"] == cheapest)))
df_cmp = pd.DataFrame(rows_compare)
df_cmp.to_csv(os.path.join(RES, "35_roi_comparison.tsv"), sep="\t", index=False)

# ============================================================
# Sensitivity: how does break-even change with hplc_cost / liability?
# ============================================================
sensitivity_rows = []
n_test = 20000
for hplc_c in [50, 100, 150, 200]:
    for liab in [1000, 5000, 20000, 100000]:
        a = annual_costs(n_test, "A_status_quo", hplc_cost=hplc_c, liability=liab)
        # Ridge+SNV (recommended)
        b_rsnv_id = annual_costs(n_test, "B_HSI_screening", params=RIDGE_SNV_IN_DOMAIN,
                                  hplc_cost=hplc_c, liability=liab)
        b_rsnv_xb = annual_costs(n_test, "B_HSI_screening", params=RIDGE_SNV_CROSS_BATCH,
                                  hplc_cost=hplc_c, liability=liab)
        b_gbm_xb = annual_costs(n_test, "B_HSI_screening", params=GBM_CROSS_BATCH,
                                 hplc_cost=hplc_c, liability=liab)
        c = annual_costs(n_test, "C_no_screening", hplc_cost=hplc_c, liability=liab)
        sensitivity_rows.append(dict(
            hplc_per_lot_usd=hplc_c, liability_per_lot_usd=liab,
            A_status_quo=a["total_usd"],
            B_RidgeSNV_in_domain=b_rsnv_id["total_usd"],
            B_RidgeSNV_cross_batch=b_rsnv_xb["total_usd"],
            B_GBM_cross_batch=b_gbm_xb["total_usd"],
            C_no_screen=c["total_usd"],
            saving_RidgeSNV_xb_vs_A=a["total_usd"] - b_rsnv_xb["total_usd"],
        ))
df_s = pd.DataFrame(sensitivity_rows)
df_s.to_csv(os.path.join(RES, "35_roi_sensitivity.tsv"), sep="\t", index=False)


# ============================================================
# Markdown
# ============================================================
md = ["# Phase 4.3 - ROI cost-benefit model",
      "",
      "**Purpose**: translate lot-level metrics (script 33c, 34c) into a cost-benefit "
      "comparison for paper 2 industrial framing. Numbers are illustrative — the "
      "framework is the deliverable, not the specific dollar figures.",
      "",
      "## Cost assumptions (defaults; should be customised by industry partner)",
      "",
      f"- HPLC-FLD cost per lot: **${HPLC_COST_PER_LOT_USD:.0f}** USD (range 80-200)",
      f"- HSI system amortised cost: **${HSI_AMORTISED_PER_YEAR_USD:.0f}** USD/yr (35-60k camera over 5-7 yr + operator)",
      f"- Base AFB1 > 8 ppb rate (natural pistachio): **{BASE_UNSAFE_RATE*100:.0f}%** (illustrative)",
      f"- Liability per unsafe lot released: **${LIABILITY_PER_UNSAFE_LOT_USD:.0f}** USD",
      "",
      "## Model performance plugged in (from earlier scripts)",
      "",
      f"- GBM in-domain lot-level @ 8 ppb: sensitivity = {GBM_IN_DOMAIN['sensitivity']:.2f}, specificity = {GBM_IN_DOMAIN['specificity']:.3f}",
      f"- GBM cross-batch lot-level @ 8 ppb: sensitivity = {GBM_CROSS_BATCH['sensitivity']:.2f}, specificity = {GBM_CROSS_BATCH['specificity']:.3f}",
      "",
      "Sensitivity/specificity are intrinsic to the model and the chosen operating "
      "point. The framework applies them to the realistic production base rate "
      "(natural ~5%, not the dataset-controlled rate of 15-35%).",
      "",
      "## Annual cost sweep across processor scale",
      "",
      "| Annual lots | Scenario | HSI amort | HPLC | Liability | TOTAL | Saving vs A |",
      "|---|---|---|---|---|---|---|"]
for _, r in df_cmp.iterrows():
    full = df[(df["n_lots"]==r["n_lots"]) & (df["scenario"]==r["scenario"])].iloc[0]
    star = " ⭐" if r["cheapest"] else ""
    md.append(f"| {r['n_lots']:,} | {r['scenario']}{star} | "
              f"${full['HSI_amort_usd']:,.0f} | ${full['HPLC_usd']:,.0f} | "
              f"${full['liability_usd']:,.0f} | "
              f"**${r['total_usd']:,.0f}** | "
              f"${r['saving_vs_A_usd']:,.0f} ({r['saving_vs_A_pct']:+.0f}%) |")

md += ["",
       "## Sensitivity to cost parameter assumptions (at 20,000 lots/yr)",
       "",
       "| HPLC/lot | Liability/lot | A status-quo | B Ridge+SNV in-domain | **B Ridge+SNV cross-batch** | B GBM cross-batch | C no-screen | Saving Ridge+SNV vs A |",
       "|---|---|---|---|---|---|---|---|"]
for _, r in df_s.iterrows():
    md.append(f"| ${r['hplc_per_lot_usd']:.0f} | ${r['liability_per_lot_usd']:,.0f} | "
              f"${r['A_status_quo']:,.0f} | ${r['B_RidgeSNV_in_domain']:,.0f} | "
              f"**${r['B_RidgeSNV_cross_batch']:,.0f}** | "
              f"${r['B_GBM_cross_batch']:,.0f} | ${r['C_no_screen']:,.0f} | "
              f"**${r['saving_RidgeSNV_xb_vs_A']:,.0f}** |")

md += ["",
       "## Read-out",
       "- Scenario B (HSI screening + selective HPLC) is cost-attractive when **HPLC cost is high and annual lot volume is large** — both make the HSI amortised cost negligible against saved HPLC checks.",
       "- Scenario C (no screening) is only competitive at low liability assumption — a single unsafe lot recall/disposal at $20k+ wipes out the saving.",
       "- In-domain vs cross-batch difference matters less than scale: GBM cross-batch's higher mid-tier fraction (57% vs 14%) means more HPLC confirmation cost, but liability stays controlled because P(unsafe|low) is similar.",
       "- The framework lets downstream industry partners plug in their own HPLC cost and lot volume to make an informed pre-screening adoption decision.",
       "",
       "## Caveats",
       "- All HPLC and HSI cost numbers are public-source ranges, not committed quotes. Liability per unsafe lot is an order-of-magnitude estimate.",
       "- Base unsafe rate 5% is an illustrative middle ground; actual rate varies by origin, season, processing chain.",
       "- The model ignores recall/return/lawsuit cost asymmetry (a single major recall can dwarf the annual screening cost).",
       "- Does not include HSI false-negative liability in MID tier (which goes to lab anyway and gets caught).",
       "",
       "## Outputs",
       "- `results/35_roi.tsv` — per-scenario per-scale cost breakdown",
       "- `results/35_roi_comparison.tsv` — savings vs status quo",
       "- `results/35_roi_sensitivity.tsv` — sensitivity to HPLC and liability cost"]

with open(os.path.join(RES, "35_roi.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md))
print("[write] results/35_roi.{tsv,md} + 2 derived tsv", flush=True)
print("DONE", flush=True)
