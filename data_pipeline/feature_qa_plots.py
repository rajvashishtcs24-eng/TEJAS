from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
QA_PLOTS_DIR = PROJECT_ROOT / "qa_plots"

df = pd.read_csv(FEATURES_DIR / "dynacard_features.csv")
anova = pd.read_csv(FEATURES_DIR / "feature_separability_anova.csv")
corr = pd.read_csv(FEATURES_DIR / "feature_correlation_matrix.csv", index_col=0)

conditions = ["Normal", "Rod Floating", "Fluid Pound", "Gas Interference"]
palette = {"Normal": "#2E7D32", "Rod Floating": "#C0392B", "Fluid Pound": "#D97B29", "Gas Interference": "#6A4FA0"}

# ---------------------------------------------------------------
# Figure 1: boxplots of top 8 most separating features by condition
# ---------------------------------------------------------------
top_feats = anova.head(8)["feature"].tolist()
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, feat in zip(axes.flat, top_feats):
    data = [df[df["condition_label"] == c][feat].values for c in conditions]
    bp = ax.boxplot(data, labels=[c.replace(" ", "\n") for c in conditions], patch_artist=True)
    for patch, c in zip(bp["boxes"], conditions):
        patch.set_facecolor(palette[c])
        patch.set_alpha(0.6)
    ax.set_title(feat, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(QA_PLOTS_DIR / "phase4_top_features_boxplots.png", dpi=130)
plt.close(fig)
print("Saved phase4_top_features_boxplots.png")

# ---------------------------------------------------------------
# Figure 2: correlation heatmap
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True,
            cbar_kws={"shrink": 0.7}, ax=ax, xticklabels=True, yticklabels=True)
ax.set_title("Feature correlation matrix (report only - nothing removed)", fontsize=13)
plt.xticks(fontsize=7, rotation=90)
plt.yticks(fontsize=7)
plt.tight_layout()
plt.savefig(QA_PLOTS_DIR / "phase4_correlation_heatmap.png", dpi=130)
plt.close(fig)
print("Saved phase4_correlation_heatmap.png")

# ---------------------------------------------------------------
# Figure 3: 2D scatter of two strong discriminators, colored by condition
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
for c in conditions:
    sub = df[df["condition_label"] == c]
    ax.scatter(sub["mean_load_z"], sub["max_abs_slope_down_z"], label=c, alpha=0.6, s=25, color=palette[c])
ax.set_xlabel("mean_load_z")
ax.set_ylabel("max_abs_slope_down_z")
ax.set_title("Top 2 discriminative features")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax2 = axes[1]
for c in conditions:
    sub = df[df["condition_label"] == c]
    ax2.scatter(sub["PPRL_raw_lbf"], sub["position_of_min_load"], label=c, alpha=0.6, s=25, color=palette[c])
ax2.set_xlabel("PPRL_raw_lbf")
ax2.set_ylabel("position_of_min_load")
ax2.set_title("Physical PPRL vs. spike location")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(QA_PLOTS_DIR / "phase4_scatter_separability.png", dpi=130)
plt.close(fig)
print("Saved phase4_scatter_separability.png")
