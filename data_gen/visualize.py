import sys
sys.path.insert(0, "/home/claude/tejas/data_gen")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from srp_physics import WellConfig
from conditions import GENERATORS

rng = np.random.default_rng(7)

fig, axes = plt.subplots(2, 4, figsize=(18, 9))

conditions = ["Normal", "Rod Floating", "Fluid Pound", "Gas Interference"]

# Row 1: one representative card per condition (fixed well config for fair comparison)
base_well = WellConfig(spm=8.0, viscosity_cp=650, fluid_level_ft=2200)
for i, cond in enumerate(conditions):
    theta, pos, load, meta = GENERATORS[cond](base_well, n_points=200, rng=rng)
    ax = axes[0, i]
    ax.plot(pos, load, lw=1.6, color="#1f3864")
    ax.fill(pos, load, alpha=0.08, color="#1f3864")
    ax.set_title(f"{cond}\n(severity={meta.get('severity', 0):.2f})", fontsize=11)
    ax.set_xlabel("Position (in)")
    if i == 0:
        ax.set_ylabel("Load (lbf)")
    ax.grid(alpha=0.25)

# Row 2: 3 overlaid samples per condition to show natural variation
for i, cond in enumerate(conditions):
    ax = axes[1, i]
    for _ in range(3):
        well = WellConfig(
            spm=rng.uniform(5, 10),
            viscosity_cp=rng.uniform(200, 900),
            fluid_level_ft=rng.uniform(1500, 3500),
        )
        theta, pos, load, meta = GENERATORS[cond](well, n_points=200, rng=rng)
        ax.plot(pos, load, lw=1.2, alpha=0.8)
    ax.set_title(f"{cond} - variation", fontsize=10)
    ax.set_xlabel("Position (in)")
    if i == 0:
        ax.set_ylabel("Load (lbf)")
    ax.grid(alpha=0.25)

plt.tight_layout()
out_path = "/home/claude/tejas/qa_plots/condition_shapes_qa.png"
plt.savefig(out_path, dpi=130)
print(f"Saved QA plot to {out_path}")
