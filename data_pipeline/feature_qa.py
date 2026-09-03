import sys
sys.path.insert(0, "/home/claude/tejas/data_pipeline")

import numpy as np
import pandas as pd
from scipy import stats

from feature_extraction import FEATURE_COLS

df = pd.read_csv("/home/claude/tejas/data/features/dynacard_features.csv")

report_lines = []


def log(s=""):
    print(s)
    report_lines.append(str(s))


log("=" * 70)
log("PHASE 4 FEATURE QA REPORT")
log("=" * 70)
log(f"\nTotal cards: {len(df)}")
log(f"Total features (excluding identifiers/context passthrough labels): {len(FEATURE_COLS)}")
log(f"\nFeature list:\n" + "\n".join(f"  - {c}" for c in FEATURE_COLS))

# --- NaN / infinite check ---
log("\n" + "-" * 70)
log("NaN / Infinite check")
log("-" * 70)
nan_counts = df[FEATURE_COLS].isna().sum()
inf_counts = df[FEATURE_COLS].apply(lambda c: np.isinf(c).sum())
any_nan = nan_counts[nan_counts > 0]
any_inf = inf_counts[inf_counts > 0]
log(f"Features with NaN values: {dict(any_nan) if len(any_nan) else 'none'}")
log(f"Features with Inf values: {dict(any_inf) if len(any_inf) else 'none'}")

# --- constant / near-constant features ---
log("\n" + "-" * 70)
log("Constant / near-constant feature check")
log("-" * 70)
nunique = df[FEATURE_COLS].nunique()
constant_feats = nunique[nunique <= 1].index.tolist()
log(f"Fully constant features: {constant_feats if constant_feats else 'none'}")

# --- duplicate feature columns (identical values across all rows) ---
log("\n" + "-" * 70)
log("Duplicate feature columns")
log("-" * 70)
dup_report = []
cols = FEATURE_COLS
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        if df[cols[i]].equals(df[cols[j]]):
            dup_report.append((cols[i], cols[j]))
log(f"Exact duplicate columns: {dup_report if dup_report else 'none'}")

# --- unreasonable ranges (basic physical sanity) ---
log("\n" + "-" * 70)
log("Range sanity checks")
log("-" * 70)
checks = {
    "PPRL_raw_lbf > MPRL_raw_lbf (always true)": (df["PPRL_raw_lbf"] > df["MPRL_raw_lbf"]).all(),
    "load_range_raw_lbf > 0": (df["load_range_raw_lbf"] > 0).all(),
    "load_range_z > 0": (df["load_range_z"] > 0).all(),
    "card_work_raw_lbf_in > 0": (df["card_work_raw_lbf_in"] > 0).all(),
    "card_area_shape_norm >= 0": (df["card_area_shape_norm"] >= 0).all(),
    "position_of_peak_load in [0,1]": df["position_of_peak_load"].between(0, 1).all(),
    "position_of_min_load in [0,1]": df["position_of_min_load"].between(0, 1).all(),
    "stroke_length in [50,200] in": df["stroke_length"].between(50, 200).all(),
    "SPM in [1,15]": df["SPM"].between(1, 15).all(),
}
for k, v in checks.items():
    log(f"  {k}: {'OK' if v else 'FAILED'}")

log(f"\nup_down_load_ratio_raw range: [{df['up_down_load_ratio_raw'].min():.2f}, {df['up_down_load_ratio_raw'].max():.2f}]")
outliers = df[(df["up_down_load_ratio_raw"] < 0.5) | (df["up_down_load_ratio_raw"] > 5)]
log(f"up_down_load_ratio_raw extreme outliers (<0.5 or >5): {len(outliers)} cards")

# --- correlation analysis (report only, no removal) ---
log("\n" + "-" * 70)
log("Strong feature correlations (|r| > 0.9) - reported only, NOT removed")
log("-" * 70)
corr = df[FEATURE_COLS].corr(numeric_only=True)
strong_pairs = []
for i in range(len(FEATURE_COLS)):
    for j in range(i + 1, len(FEATURE_COLS)):
        r = corr.iloc[i, j]
        if abs(r) > 0.9:
            strong_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], round(r, 3)))
strong_pairs.sort(key=lambda x: -abs(x[2]))
for a, b, r in strong_pairs:
    log(f"  {a}  <->  {b}   r={r}")
if not strong_pairs:
    log("  none above threshold")
corr.to_csv("/home/claude/tejas/data/features/feature_correlation_matrix.csv")

# --- basic statistics by condition ---
log("\n" + "-" * 70)
log("Basic statistics by condition (mean, selected headline features)")
log("-" * 70)
headline = ["PPRL_raw_lbf", "MPRL_raw_lbf", "load_range_raw_lbf", "card_work_raw_lbf_in",
            "max_abs_slope_down_z", "position_of_peak_load", "card_area_shape_norm"]
by_cond = df.groupby("condition_label")[headline].mean().round(2)
log(by_cond.to_string())

# --- separability check via one-way ANOVA (descriptive statistic, NOT a trained model) ---
log("\n" + "-" * 70)
log("Separability check: one-way ANOVA F-statistic per feature across the")
log("4 conditions (descriptive statistic only - NOT a trained classifier)")
log("-" * 70)
anova_results = []
groups_by_cond = {c: g for c, g in df.groupby("condition_label")}
conditions = list(groups_by_cond.keys())
for feat in FEATURE_COLS:
    samples = [groups_by_cond[c][feat].values for c in conditions]
    try:
        f_stat, p_val = stats.f_oneway(*samples)
    except Exception:
        f_stat, p_val = np.nan, np.nan
    anova_results.append((feat, f_stat, p_val))
anova_df = pd.DataFrame(anova_results, columns=["feature", "F_statistic", "p_value"]).sort_values(
    "F_statistic", ascending=False
)
log(anova_df.to_string(index=False))
anova_df.to_csv("/home/claude/tejas/data/features/feature_separability_anova.csv", index=False)

log(f"\nTop 8 most separating features by F-statistic:")
for _, r in anova_df.head(8).iterrows():
    log(f"  {r['feature']}: F={r['F_statistic']:.1f}, p={r['p_value']:.2e}")

n_significant = (anova_df["p_value"] < 0.01).sum()
log(f"\n{n_significant} / {len(FEATURE_COLS)} features show statistically significant "
    f"(p<0.01) differences across conditions.")

with open("/home/claude/tejas/data/features/PHASE4_QA_REPORT.txt", "w") as f:
    f.write("\n".join(report_lines))

print("\nSaved: feature_correlation_matrix.csv, feature_separability_anova.csv, PHASE4_QA_REPORT.txt")
