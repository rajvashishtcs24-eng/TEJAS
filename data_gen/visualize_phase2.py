import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data_pipeline"))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from preprocess import run_pipeline

out = run_pipeline()
meta = out["meta"]
shape_arr = out["processed_shape"]
mag_arr = out["processed_magnitude"]

rng = np.random.default_rng(3)
conditions = ["Normal", "Rod Floating", "Fluid Pound", "Gas Interference"]

# ---------------------------------------------------------------
# Figure 1: raw vs processed (branch-split resampled) - one per condition
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for i, cond in enumerate(conditions):
    idx = meta.index[meta["condition_label"] == cond][0]
    row = meta.loc[idx]
    raw_pos = np.array(json.loads(row["position"]))
    raw_load = np.array(json.loads(row["load"]))

    ax = axes[0, i]
    ax.plot(raw_pos, raw_load, lw=1.4, color="#333333")
    ax.set_title(f"{cond}\nRAW (n={len(raw_pos)} pts)", fontsize=10)
    ax.set_xlabel("Position (in)")
    if i == 0:
        ax.set_ylabel("Load (lbf)")
    ax.grid(alpha=0.25)

    ax2 = axes[1, i]
    pos_n = shape_arr[idx, :, 0]
    load_n = shape_arr[idx, :, 1]
    ax2.plot(pos_n[:100], load_n[:100], lw=1.6, color="#1f7a3d", label="upstroke")
    ax2.plot(pos_n[100:], load_n[100:], lw=1.6, color="#c0392b", label="downstroke")
    ax2.set_title("PROCESSED (branch-split, 0\u21921 pos, 200 pts)", fontsize=10)
    ax2.set_xlabel("Normalized position")
    if i == 0:
        ax2.set_ylabel("Shape-norm load [0,1]")
        ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "qa_plots" / "phase2_raw_vs_processed.png", dpi=130)
plt.close(fig)
print("Saved phase2_raw_vs_processed.png")

# ---------------------------------------------------------------
# Figure 2: shape-normalized vs magnitude-preserving, one per condition
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for i, cond in enumerate(conditions):
    idx = meta.index[meta["condition_label"] == cond][0]

    ax = axes[0, i]
    ax.plot(shape_arr[idx, :100, 0], shape_arr[idx, :100, 1], color="#1f7a3d")
    ax.plot(shape_arr[idx, 100:, 0], shape_arr[idx, 100:, 1], color="#c0392b")
    ax.set_title(f"{cond}\nSHAPE-normalized", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    if i == 0:
        ax.set_ylabel("Load (min-max)")
    ax.grid(alpha=0.25)

    ax2 = axes[1, i]
    ax2.plot(mag_arr[idx, :100, 0], mag_arr[idx, :100, 1], color="#1f7a3d")
    ax2.plot(mag_arr[idx, 100:, 0], mag_arr[idx, 100:, 1], color="#c0392b")
    ax2.set_title("MAGNITUDE-preserving (well z-score)", fontsize=10)
    ax2.set_xlabel("Normalized position")
    if i == 0:
        ax2.set_ylabel("Load (z-score)")
    ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(PROJECT_ROOT / "qa_plots" / "phase2_shape_vs_magnitude.png", dpi=130)
plt.close(fig)
print("Saved phase2_shape_vs_magnitude.png")

# ---------------------------------------------------------------
# Figure 3: several randomly selected processed cards (magnitude repr)
# ---------------------------------------------------------------
sample_idxs = rng.choice(len(meta), size=8, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, idx in zip(axes.flat, sample_idxs):
    row = meta.loc[idx]
    ax.plot(mag_arr[idx, :100, 0], mag_arr[idx, :100, 1], color="#1f7a3d", lw=1.3)
    ax.plot(mag_arr[idx, 100:, 0], mag_arr[idx, 100:, 1], color="#c0392b", lw=1.3)
    ax.set_title(f"{row['condition_label']} | {row['well_id']}", fontsize=9)
    ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(PROJECT_ROOT / "qa_plots" / "phase2_random_samples.png", dpi=130)
plt.close(fig)
print("Saved phase2_random_samples.png")
