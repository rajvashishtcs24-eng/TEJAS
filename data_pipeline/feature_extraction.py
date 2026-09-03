"""
feature_extraction.py
----------------------
Phase 4: extracts an interpretable engineering feature set from every
processed dynamometer card. Reads:
  - data/processed/processed_cards_shape.npy       [n,200,2]
  - data/processed/processed_cards_magnitude.npy   [n,200,2]
  - data/processed/processed_metadata.csv          (incl. preserved raw
                                                      position[]/load[] JSON)

Does NOT train any ML model. Output target column is condition_label only;
risk_level / recommended_action are carried through as reference columns,
never used as features or targets.

All rows remain tagged data_source=synthetic_physics_v1 (untouched).

Channel convention (from Phase 2): index 0-99 = upstroke, 100-199 =
downstroke; position runs 0->1 on the upstroke and 1->0 on the downstroke.

FEATURE DESIGN NOTES (see PHASE4_README.md for full documentation):
- PPRL/MPRL/load range/mean/std are computed BOTH from the magnitude-
  preserving (well z-score) array (ML-ready, cross-well comparable) AND
  from the preserved RAW load array in real lbf (true engineering units -
  needed for e.g. rod-load-capacity style reasoning later).
- Card area/work is computed BOTH as true physical work (lbf*in, shoelace
  integral over the RAW closed loop) and as a dimensionless shape-only
  area (shoelace over the shape-normalized [0,1]x[0,1] loop). Neither the
  magnitude (z-score) nor a naive use of normalized position alone would
  give a physically meaningful "work" value, hence the split.
- Slope stats report both MEAN slope (overall trend) and PEAK |slope|
  (captures brief impact spikes that a mean would wash out) per branch.
- Two positional shape descriptors record WHERE the peak/min load occurs
  along the normalized stroke (0->1), since spike location itself is
  diagnostic (rod floating: near-bottom; fluid pound: mid-stroke;
  gas interference: no sharp spike at all).
"""

import json
import numpy as np
import pandas as pd

PROC_DIR = "/home/claude/tejas/data/processed"
OUT_DIR = "/home/claude/tejas/data/features"

CONTEXT_COLS = ["SPM", "temperature", "viscosity", "fluid_level", "pump_depth",
                "production_rate", "stroke_length"]


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace formula: signed area of a closed polygon traced by (x,y).
    Returns absolute area (work is always reported positive)."""
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def split_raw_branches(raw_pos: np.ndarray, raw_load: np.ndarray):
    """Split a raw (variable-length) card at its position peak into
    upstroke / downstroke segments. Mirrors the Phase 2 branch-split logic,
    used here only to get per-branch scalar aggregates (no resampling)."""
    peak_idx = int(np.argmax(raw_pos))
    peak_idx = max(1, min(peak_idx, len(raw_pos) - 2))
    up_load = raw_load[: peak_idx + 1]
    down_load = raw_load[peak_idx:]
    return up_load, down_load


def slope_stats(pos: np.ndarray, load: np.ndarray):
    """Mean and peak-absolute slope (dLoad/dPosition) along one branch."""
    d_load = np.diff(load)
    d_pos = np.diff(pos)
    d_pos = np.where(np.abs(d_pos) < 1e-9, 1e-9, d_pos)
    slopes = d_load / d_pos
    return float(np.mean(slopes)), float(np.max(np.abs(slopes)))


def extract_features_for_card(row, mag_card: np.ndarray, shape_card: np.ndarray) -> dict:
    feat = {}

    # ---- context (carried, not derived) ----
    for col in CONTEXT_COLS:
        feat[col] = row[col]

    # ---- raw preserved arrays (real units) ----
    raw_pos = np.array(json.loads(row["position"]), dtype=float)
    raw_load = np.array(json.loads(row["load"]), dtype=float)

    # ---- magnitude-preserving (z-score) headline stats ----
    mag_load = mag_card[:, 1]
    feat["PPRL_z"] = float(np.max(mag_load))
    feat["MPRL_z"] = float(np.min(mag_load))
    feat["load_range_z"] = feat["PPRL_z"] - feat["MPRL_z"]
    feat["mean_load_z"] = float(np.mean(mag_load))
    feat["std_load_z"] = float(np.std(mag_load))

    # ---- raw-physical headline stats (true lbf) ----
    feat["PPRL_raw_lbf"] = float(np.max(raw_load))
    feat["MPRL_raw_lbf"] = float(np.min(raw_load))
    feat["load_range_raw_lbf"] = feat["PPRL_raw_lbf"] - feat["MPRL_raw_lbf"]
    feat["mean_load_raw_lbf"] = float(np.mean(raw_load))
    feat["std_load_raw_lbf"] = float(np.std(raw_load))

    # ---- card area / work ----
    feat["card_work_raw_lbf_in"] = polygon_area(raw_pos, raw_load)
    feat["card_area_shape_norm"] = polygon_area(shape_card[:, 0], shape_card[:, 1])

    # ---- upstroke / downstroke stats (magnitude z-score) ----
    up_z = mag_card[:100, 1]
    down_z = mag_card[100:, 1]
    feat["mean_up_z"] = float(np.mean(up_z))
    feat["std_up_z"] = float(np.std(up_z))
    feat["min_up_z"] = float(np.min(up_z))
    feat["max_up_z"] = float(np.max(up_z))
    feat["mean_down_z"] = float(np.mean(down_z))
    feat["std_down_z"] = float(np.std(down_z))
    feat["min_down_z"] = float(np.min(down_z))
    feat["max_down_z"] = float(np.max(down_z))

    # ---- slope statistics (magnitude z-score, per branch) ----
    up_pos_n = mag_card[:100, 0]
    down_pos_n = mag_card[100:, 0]
    mean_slope_up, max_abs_slope_up = slope_stats(up_pos_n, up_z)
    mean_slope_down, max_abs_slope_down = slope_stats(down_pos_n, down_z)
    feat["mean_slope_up_z"] = mean_slope_up
    feat["max_abs_slope_up_z"] = max_abs_slope_up
    feat["mean_slope_down_z"] = mean_slope_down
    feat["max_abs_slope_down_z"] = max_abs_slope_down

    # ---- geometric / shape descriptors ----
    full_pos = mag_card[:, 0]
    feat["position_of_peak_load"] = float(full_pos[np.argmax(mag_load)])
    feat["position_of_min_load"] = float(full_pos[np.argmin(mag_load)])

    raw_up_load, raw_down_load = split_raw_branches(raw_pos, raw_load)
    feat["up_down_load_ratio_raw"] = float(np.mean(raw_up_load) / np.mean(raw_down_load))

    return feat


def run_extraction():
    shape_arr = np.load(f"{PROC_DIR}/processed_cards_shape.npy")
    mag_arr = np.load(f"{PROC_DIR}/processed_cards_magnitude.npy")
    meta = pd.read_csv(f"{PROC_DIR}/processed_metadata.csv")

    assert len(meta) == shape_arr.shape[0] == mag_arr.shape[0], "row-count mismatch between metadata and arrays"

    records = []
    for i, row in meta.iterrows():
        feat = extract_features_for_card(row, mag_arr[i], shape_arr[i])
        feat["card_id"] = row["card_id"]
        feat["well_id"] = row["well_id"]
        feat["condition_label"] = row["condition_label"]
        feat["risk_level"] = row["risk_level"]  # reference only, NOT a target
        feat["recommended_action"] = row["recommended_action"]  # reference only
        feat["severity"] = row["severity"]
        feat["split"] = row["split"]
        feat["data_source"] = row["data_source"]
        records.append(feat)

    feat_df = pd.DataFrame(records)

    # reorder: identifiers first, then context, then card-derived features
    id_cols = ["card_id", "well_id", "condition_label", "risk_level",
               "recommended_action", "severity", "split", "data_source"]
    ordered_cols = id_cols + CONTEXT_COLS + [
        c for c in feat_df.columns if c not in id_cols + CONTEXT_COLS
    ]
    feat_df = feat_df[ordered_cols]

    feat_df.to_csv(f"{OUT_DIR}/dynacard_features.csv", index=False)
    return feat_df


FEATURE_COLS = [
    "PPRL_z", "MPRL_z", "load_range_z", "mean_load_z", "std_load_z",
    "PPRL_raw_lbf", "MPRL_raw_lbf", "load_range_raw_lbf", "mean_load_raw_lbf", "std_load_raw_lbf",
    "card_work_raw_lbf_in", "card_area_shape_norm",
    "mean_up_z", "std_up_z", "min_up_z", "max_up_z",
    "mean_down_z", "std_down_z", "min_down_z", "max_down_z",
    "mean_slope_up_z", "max_abs_slope_up_z", "mean_slope_down_z", "max_abs_slope_down_z",
    "position_of_peak_load", "position_of_min_load", "up_down_load_ratio_raw",
] + CONTEXT_COLS


if __name__ == "__main__":
    df = run_extraction()
    print(f"Extracted {len(FEATURE_COLS)} features for {len(df)} cards")
    print(f"Wrote {OUT_DIR}/dynacard_features.csv")
    print(df["condition_label"].value_counts())
