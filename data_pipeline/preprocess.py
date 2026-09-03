"""
preprocess.py
-------------
Phase 2 preprocessing pipeline for TEJAS dynamometer-card data.

Reads the raw synthetic dataset (data_source=synthetic_physics_v1), validates
it, resamples every card onto a normalized 0->1 position grid (branch-split,
NOT crank angle), produces two parallel load representations (shape-only and
magnitude-preserving), and writes processed arrays + metadata + QA artifacts.

Does NOT train any model. Does NOT use risk_level/recommended_action as ML
targets - they are carried through metadata only, as placeholders from Phase 1.

RESAMPLING METHOD (important):
A dynamometer card is a closed loop: position rises 0->max on the upstroke
then falls max->0 on the downstroke, so a single position value maps to two
load values. We therefore split each raw card at its position peak into an
upstroke branch and a downstroke branch, normalize position by stroke length
within each branch (0..1), and resample each branch independently onto 100
evenly spaced normalized-position points (100 up + 100 down = 200 total).
This preserves the natural closed-loop trace order and avoids ever treating
load as a single-valued function of position across the full cycle.
"""

import json
import numpy as np
import pandas as pd

RAW_CSV = "/home/claude/tejas/data/synthetic/dynacards_synthetic_v1.csv"
OUT_DIR = "/home/claude/tejas/data/processed"
N_HALF = 100  # points per branch -> 200 total per card

EXPECTED_DATA_SOURCE = "synthetic_physics_v1"


# --------------------------------------------------------------------------
# Loading & validation
# --------------------------------------------------------------------------

def load_raw(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def parse_card(row) -> dict:
    """Parse one row's JSON position/load strings. Returns dict with parsed
    arrays and a validity flag + reason if invalid. Never raises."""
    result = {"valid": True, "reason": None, "position": None, "load": None, "n_raw_points": None}
    try:
        pos = np.array(json.loads(row["position"]), dtype=float)
        load = np.array(json.loads(row["load"]), dtype=float)
    except Exception as e:
        result["valid"] = False
        result["reason"] = f"json_parse_error: {e}"
        return result

    result["n_raw_points"] = len(pos)

    if len(pos) != len(load):
        result["valid"] = False
        result["reason"] = "position_load_length_mismatch"
        return result
    if len(pos) < 10:
        result["valid"] = False
        result["reason"] = "too_few_points"
        return result
    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(load))):
        result["valid"] = False
        result["reason"] = "non_finite_values"
        return result
    if row.get("stroke_length", 0) is None or row["stroke_length"] <= 0:
        result["valid"] = False
        result["reason"] = "invalid_stroke_length"
        return result
    if np.ptp(pos) < 1e-6:
        result["valid"] = False
        result["reason"] = "degenerate_position_no_variation"
        return result
    if row.get("data_source") != EXPECTED_DATA_SOURCE:
        result["valid"] = False
        result["reason"] = f"unexpected_data_source:{row.get('data_source')}"
        return result

    result["position"] = pos
    result["load"] = load
    return result


# --------------------------------------------------------------------------
# Branch-split resampling onto normalized position grid (0->1)
# --------------------------------------------------------------------------

def split_and_resample(position: np.ndarray, load: np.ndarray, stroke_length: float,
                        n_half: int = N_HALF):
    """Split a raw card at its position peak into upstroke/downstroke branches,
    normalize position by stroke_length, resample each branch onto n_half
    evenly spaced normalized-position points. Returns:
      pos_norm (2*n_half,), load_resampled (2*n_half,), split_index (int)
    """
    peak_idx = int(np.argmax(position))
    # guard against a peak at the very start/end (degenerate branch)
    peak_idx = max(1, min(peak_idx, len(position) - 2))

    up_pos = position[: peak_idx + 1]
    up_load = load[: peak_idx + 1]
    down_pos = position[peak_idx:]
    down_load = load[peak_idx:]

    up_pos_n = up_pos / stroke_length
    down_pos_n = down_pos / stroke_length

    grid = np.linspace(0.0, 1.0, n_half)

    # upstroke: position ascending -> np.interp needs ascending x
    order_up = np.argsort(up_pos_n)
    up_load_resampled = np.interp(grid, up_pos_n[order_up], up_load[order_up])

    # downstroke: position descending -> sort ascending for interp, then reverse output
    order_down = np.argsort(down_pos_n)
    down_load_grid_ascending = np.interp(grid, down_pos_n[order_down], down_load[order_down])
    down_pos_out = grid[::-1]                     # 1 -> 0
    down_load_out = down_load_grid_ascending[::-1]

    pos_norm = np.concatenate([grid, down_pos_out])          # 0->1 then 1->0
    load_resampled = np.concatenate([up_load_resampled, down_load_out])

    return pos_norm, load_resampled, peak_idx


# --------------------------------------------------------------------------
# Load normalization variants
# --------------------------------------------------------------------------

def shape_normalize_load(load_resampled: np.ndarray) -> np.ndarray:
    """Per-card min-max normalization -> [0, 1]. Discards absolute magnitude
    by design; intended for shape-based ML/CNN."""
    lo, hi = load_resampled.min(), load_resampled.max()
    if hi - lo < 1e-9:
        return np.zeros_like(load_resampled)
    return (load_resampled - lo) / (hi - lo)


def compute_well_load_stats(resampled_loads: list, well_ids: list) -> dict:
    """Per-well_id mean/std computed across all points of all cards assigned
    to that well_id. CAVEAT (see README): well_id is not currently bound to
    consistent physical parameters in Phase 1, so this does not yet achieve
    genuine well-specific baseline removal - it's a correctly-implemented
    placeholder for when Phase 1's well assignment is fixed."""
    df_tmp = pd.DataFrame({
        "well_id": well_ids,
        "load_flat": [arr for arr in resampled_loads],
    })
    stats = {}
    for wid, group in df_tmp.groupby("well_id"):
        all_vals = np.concatenate(group["load_flat"].values)
        stats[wid] = (float(all_vals.mean()), float(all_vals.std() + 1e-9))
    return stats


def magnitude_normalize_load(load_resampled: np.ndarray, well_id: str, well_stats: dict) -> np.ndarray:
    """Well-level z-score. Preserves relative magnitude information (e.g. PPRL
    separation between conditions) rather than collapsing every card to the
    same [0,1] range."""
    mean, std = well_stats[well_id]
    return (load_resampled - mean) / std


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

def run_pipeline():
    df = load_raw(RAW_CSV)
    n_raw = len(df)

    parsed = [parse_card(row) for _, row in df.iterrows()]
    valid_mask = [p["valid"] for p in parsed]
    n_valid = sum(valid_mask)
    n_invalid = n_raw - n_valid
    invalid_reasons = pd.Series([p["reason"] for p in parsed if not p["valid"]]).value_counts()
    raw_point_counts = pd.Series([p["n_raw_points"] for p in parsed if p["n_raw_points"] is not None])

    # --- resample valid cards, both load representations ---
    pos_arrays, resampled_loads, split_indices = [], [], []
    valid_rows = []
    for (idx, row), p in zip(df.iterrows(), parsed):
        if not p["valid"]:
            continue
        pos_norm, load_resampled, split_idx = split_and_resample(
            p["position"], p["load"], row["stroke_length"]
        )
        pos_arrays.append(pos_norm)
        resampled_loads.append(load_resampled)
        split_indices.append(split_idx)
        valid_rows.append(row)

    meta = pd.DataFrame(valid_rows).reset_index(drop=True)
    meta["n_raw_points"] = [p["n_raw_points"] for p in parsed if p["valid"]]
    meta["branch_split_index"] = split_indices

    # shape-normalized representation
    shape_loads = [shape_normalize_load(l) for l in resampled_loads]

    # magnitude-preserving representation (well-level z-score)
    well_stats = compute_well_load_stats(resampled_loads, meta["well_id"].tolist())
    magnitude_loads = [
        magnitude_normalize_load(l, wid, well_stats)
        for l, wid in zip(resampled_loads, meta["well_id"])
    ]

    n_points_total = 2 * N_HALF
    processed_shape = np.stack(
        [np.stack([p, l], axis=1) for p, l in zip(pos_arrays, shape_loads)]
    )  # [n_valid, 200, 2]
    processed_magnitude = np.stack(
        [np.stack([p, l], axis=1) for p, l in zip(pos_arrays, magnitude_loads)]
    )  # [n_valid, 200, 2]

    assert processed_shape.shape == (n_valid, n_points_total, 2)
    assert processed_magnitude.shape == (n_valid, n_points_total, 2)

    np.save(f"{OUT_DIR}/processed_cards_shape.npy", processed_shape)
    np.save(f"{OUT_DIR}/processed_cards_magnitude.npy", processed_magnitude)
    meta.to_csv(f"{OUT_DIR}/processed_metadata.csv", index=False)

    # --- well-level split reporting (NOT silently created) ---
    wells_summary = meta.groupby("well_id").agg(
        n_cards=("card_id", "count"),
    )
    cond_by_well = pd.crosstab(meta["well_id"], meta["condition_label"])
    n_unique_wells = meta["well_id"].nunique()

    rng = np.random.default_rng(123)
    unique_wells = list(meta["well_id"].unique())
    perm = rng.permutation(len(unique_wells))
    unique_wells = [unique_wells[i] for i in perm]
    n_train_wells = int(round(0.8 * len(unique_wells)))
    train_wells = set(unique_wells[:n_train_wells])
    val_wells = set(unique_wells[n_train_wells:])
    meta["split"] = meta["well_id"].apply(lambda w: "train" if w in train_wells else "val")
    meta.to_csv(f"{OUT_DIR}/processed_metadata.csv", index=False)  # rewrite with split col

    split_cond_dist = pd.crosstab(meta["split"], meta["condition_label"])
    split_well_counts = meta.groupby("split")["well_id"].nunique()

    qa_report = {
        "n_raw_cards": n_raw,
        "n_valid_cards": n_valid,
        "n_invalid_cards": n_invalid,
        "invalid_reasons": invalid_reasons.to_dict(),
        "raw_point_counts_min": int(raw_point_counts.min()) if len(raw_point_counts) else None,
        "raw_point_counts_max": int(raw_point_counts.max()) if len(raw_point_counts) else None,
        "condition_counts_raw": df["condition_label"].value_counts().to_dict(),
        "condition_counts_valid": meta["condition_label"].value_counts().to_dict(),
        "n_unique_wells": int(n_unique_wells),
        "cards_per_well_min": int(wells_summary["n_cards"].min()),
        "cards_per_well_max": int(wells_summary["n_cards"].max()),
        "cards_per_well_mean": float(wells_summary["n_cards"].mean()),
        "train_wells": int(split_well_counts.get("train", 0)),
        "val_wells": int(split_well_counts.get("val", 0)),
        "load_range_raw_min": float(min(l.min() for l in resampled_loads)),
        "load_range_raw_max": float(max(l.max() for l in resampled_loads)),
        "position_norm_min": float(min(p.min() for p in pos_arrays)),
        "position_norm_max": float(max(p.max() for p in pos_arrays)),
    }

    return {
        "meta": meta,
        "processed_shape": processed_shape,
        "processed_magnitude": processed_magnitude,
        "qa_report": qa_report,
        "cond_by_well": cond_by_well,
        "split_cond_dist": split_cond_dist,
        "resampled_loads": resampled_loads,
        "pos_arrays": pos_arrays,
        "well_stats": well_stats,
    }


if __name__ == "__main__":
    out = run_pipeline()
    print("=== QA REPORT ===")
    for k, v in out["qa_report"].items():
        print(f"{k}: {v}")
    print("\n=== CONDITION COUNTS BY WELL (first 10 wells) ===")
    print(out["cond_by_well"].head(10))
    print("\n=== TRAIN/VAL CONDITION DISTRIBUTION ===")
    print(out["split_cond_dist"])
