"""
train_phase5a.py
-----------------
Phase 5A: Feature-based ML classification of dynamometer-card conditions.

Models trained:
  1. Random Forest (sklearn.ensemble.RandomForestClassifier)
  2. HistGradientBoosting (sklearn.ensemble.HistGradientBoostingClassifier)
     — this is scikit-learn's native gradient-boosting implementation,
     used here as a gradient-boosting baseline. It is NOT XGBoost.
     It was chosen because it is a high-quality, dependency-free
     gradient-boosting classifier available in scikit-learn, avoiding
     the need for an external XGBoost installation.

Target: condition_label only (4 classes).
Split: existing well-level train/val split from the `split` column
       (Phase 1.1 persistent-well identity, Phase 2 well-grouped split).
       Train = 319 cards (16 wells), Val = 81 cards (4 wells).

risk_level and recommended_action are NOT used as features or targets.
data_source = synthetic_physics_v1 on every row (unchanged).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURES_CSV = PROJECT_ROOT / "data" / "features" / "dynacard_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"

FEATURE_COLS = [
    "PPRL_z", "MPRL_z", "load_range_z", "mean_load_z", "std_load_z",
    "PPRL_raw_lbf", "MPRL_raw_lbf", "load_range_raw_lbf", "mean_load_raw_lbf", "std_load_raw_lbf",
    "card_work_raw_lbf_in", "card_area_shape_norm",
    "mean_up_z", "std_up_z", "min_up_z", "max_up_z",
    "mean_down_z", "std_down_z", "min_down_z", "max_down_z",
    "mean_slope_up_z", "max_abs_slope_up_z", "mean_slope_down_z", "max_abs_slope_down_z",
    "position_of_peak_load", "position_of_min_load", "up_down_load_ratio_raw",
    "SPM", "temperature", "viscosity", "fluid_level", "pump_depth",
    "production_rate", "stroke_length",
]

TARGET = "condition_label"
CLASS_ORDER = ["Normal", "Rod Floating", "Fluid Pound", "Gas Interference"]


def load_data():
    """Load features CSV and split into train/val using the existing split column."""
    df = pd.read_csv(FEATURES_CSV)
    assert len(df) == 400, f"Expected 400 rows, got {len(df)}"
    assert TARGET in df.columns
    assert "split" in df.columns
    assert all(c in df.columns for c in FEATURE_COLS), "Missing feature columns"

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()

    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET].values
    X_val = val[FEATURE_COLS].values
    y_val = val[TARGET].values

    print(f"Train: {len(train)} cards, {train['well_id'].nunique()} wells")
    print(f"Val:   {len(val)} cards, {val['well_id'].nunique()} wells")
    print(f"Train condition counts: {train[TARGET].value_counts().to_dict()}")
    print(f"Val   condition counts: {val[TARGET].value_counts().to_dict()}")

    return df, train, val, X_train, y_train, X_val, y_val


def evaluate_model(name, model, X_train, y_train, X_val, y_val, feature_names, val_df):
    """Full evaluation of a trained model. Returns a results dict."""
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    results = {}

    # --- headline metrics ---
    results["train_accuracy"] = accuracy_score(y_train, y_pred_train)
    results["val_accuracy"] = accuracy_score(y_val, y_pred_val)
    results["val_balanced_accuracy"] = balanced_accuracy_score(y_val, y_pred_val)
    results["val_macro_f1"] = f1_score(y_val, y_pred_val, average="macro")

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    print(f"  Train accuracy:         {results['train_accuracy']:.4f}")
    print(f"  Val accuracy:           {results['val_accuracy']:.4f}")
    print(f"  Val balanced accuracy:  {results['val_balanced_accuracy']:.4f}")
    print(f"  Val macro F1:           {results['val_macro_f1']:.4f}")

    # --- per-class report ---
    report_str = classification_report(
        y_val, y_pred_val, labels=CLASS_ORDER, digits=4, zero_division=0
    )
    results["classification_report"] = report_str
    print(f"\n  Per-class report (validation):\n{report_str}")

    # --- confusion matrix ---
    cm = confusion_matrix(y_val, y_pred_val, labels=CLASS_ORDER)
    results["confusion_matrix"] = cm
    print(f"  Confusion matrix (rows=true, cols=pred):")
    print(f"  Labels: {CLASS_ORDER}")
    for i, row in enumerate(cm):
        print(f"    {CLASS_ORDER[i]:20s} {row}")

    # --- most common errors ---
    errors = []
    for true_label, pred_label in zip(y_val, y_pred_val):
        if true_label != pred_label:
            errors.append((true_label, pred_label))

    if errors:
        from collections import Counter
        error_counts = Counter(errors).most_common()
        results["most_common_errors"] = error_counts
        print(f"\n  Most common errors (true -> predicted):")
        for (true_l, pred_l), count in error_counts:
            print(f"    {true_l:20s} -> {pred_l:20s}  ({count} cards)")
    else:
        results["most_common_errors"] = []
        print(f"\n  No errors on validation set!")

    # --- feature importance (model-native, Gini/gain-based) ---
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = None

    if importances is not None:
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        results["feature_importance"] = imp_df
        print(f"\n  Top 10 features (model-native importance):")
        for _, r in imp_df.head(10).iterrows():
            print(f"    {r['feature']:30s}  {r['importance']:.4f}")

    # --- permutation importance (validation set) ---
    print(f"\n  Computing permutation importance (10 repeats)...")
    perm_result = permutation_importance(
        model, X_val, y_val, n_repeats=10, random_state=42, scoring="balanced_accuracy"
    )
    perm_df = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": perm_result.importances_mean,
        "perm_importance_std": perm_result.importances_std,
    }).sort_values("perm_importance_mean", ascending=False)
    results["permutation_importance"] = perm_df
    print(f"  Top 10 features (permutation importance, balanced accuracy):")
    for _, r in perm_df.head(10).iterrows():
        print(f"    {r['feature']:30s}  {r['perm_importance_mean']:.4f} +/- {r['perm_importance_std']:.4f}")

    # --- per-well validation performance ---
    well_results = []
    for wid in sorted(val_df["well_id"].unique()):
        mask = val_df["well_id"].values == wid
        w_true = y_val[mask]
        w_pred = y_pred_val[mask]
        w_acc = accuracy_score(w_true, w_pred)
        w_n = mask.sum()
        w_correct = (w_true == w_pred).sum()
        w_conditions = list(pd.Series(w_true).value_counts().to_dict().items())
        well_results.append({
            "well_id": wid,
            "n_cards": w_n,
            "accuracy": w_acc,
            "correct": w_correct,
            "conditions": w_conditions,
        })

    results["per_well_val"] = well_results
    print(f"\n  Per-well validation performance:")
    print(f"    {'Well':<12} {'Cards':>5} {'Acc':>8} {'Correct':>8}  Conditions")
    for wr in well_results:
        cond_str = ", ".join(f"{c}:{n}" for c, n in wr["conditions"])
        print(f"    {wr['well_id']:<12} {wr['n_cards']:>5} {wr['accuracy']:>8.4f} {wr['correct']:>5}/{wr['n_cards']:<3}  {cond_str}")

    return results


def save_results(name, model, results, feature_names):
    """Save model and evaluation results."""
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    # save model
    model_path = MODELS_DIR / f"{slug}.joblib"
    joblib.dump(model, model_path)
    print(f"\n  Model saved: {model_path}")

    # save classification report
    report_path = RESULTS_DIR / f"{slug}_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Train accuracy: {results['train_accuracy']:.4f}\n")
        f.write(f"Val accuracy: {results['val_accuracy']:.4f}\n")
        f.write(f"Val balanced accuracy: {results['val_balanced_accuracy']:.4f}\n")
        f.write(f"Val macro F1: {results['val_macro_f1']:.4f}\n\n")
        f.write(results["classification_report"])
        f.write(f"\nConfusion matrix (rows=true, cols=pred):\n")
        f.write(f"Labels: {CLASS_ORDER}\n")
        for i, row in enumerate(results["confusion_matrix"]):
            f.write(f"  {CLASS_ORDER[i]:20s} {list(row)}\n")
        f.write(f"\nMost common errors (true -> predicted):\n")
        for (true_l, pred_l), count in results.get("most_common_errors", []):
            f.write(f"  {true_l:20s} -> {pred_l:20s}  ({count} cards)\n")
        if not results.get("most_common_errors"):
            f.write("  None\n")
        f.write(f"\nPer-well validation:\n")
        for wr in results["per_well_val"]:
            cond_str = ", ".join(f"{c}:{n}" for c, n in wr["conditions"])
            f.write(f"  {wr['well_id']:<12} {wr['n_cards']:>3} cards  acc={wr['accuracy']:.4f}  {cond_str}\n")
    print(f"  Report saved: {report_path}")

    # save feature importance
    if "feature_importance" in results:
        imp_path = RESULTS_DIR / f"{slug}_feature_importance.csv"
        results["feature_importance"].to_csv(imp_path, index=False)
        print(f"  Feature importance saved: {imp_path}")

    # save permutation importance
    perm_path = RESULTS_DIR / f"{slug}_permutation_importance.csv"
    results["permutation_importance"].to_csv(perm_path, index=False)
    print(f"  Permutation importance saved: {perm_path}")

    # save confusion matrix
    cm_path = RESULTS_DIR / f"{slug}_confusion_matrix.csv"
    cm_df = pd.DataFrame(results["confusion_matrix"], index=CLASS_ORDER, columns=CLASS_ORDER)
    cm_df.to_csv(cm_path)
    print(f"  Confusion matrix saved: {cm_path}")


def main():
    print("=" * 70)
    print("TEJAS Phase 5A — Feature-Based ML Classification")
    print("=" * 70)
    print()
    print("NOTE: HistGradientBoostingClassifier is scikit-learn's native")
    print("gradient-boosting implementation. It is NOT XGBoost. It is used")
    print("here as a gradient-boosting baseline without requiring an")
    print("external XGBoost installation.")
    print()

    # create output dirs
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # load data
    df, train_df, val_df, X_train, y_train, X_val, y_val = load_data()

    # ---- Model 1: Random Forest ----
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,        # fully grown trees
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",  # handle class imbalance
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_results = evaluate_model(
        "Random Forest", rf, X_train, y_train, X_val, y_val, FEATURE_COLS, val_df
    )
    save_results("Random Forest", rf, rf_results, FEATURE_COLS)

    # ---- Model 2: HistGradientBoosting ----
    print("\n--- Training HistGradientBoosting ---")
    print("(scikit-learn native gradient boosting — NOT XGBoost)")
    hgb = HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.1,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    hgb.fit(X_train, y_train)
    hgb_results = evaluate_model(
        "HistGradientBoosting (not XGBoost)", hgb,
        X_train, y_train, X_val, y_val, FEATURE_COLS, val_df
    )
    save_results("HistGradientBoosting (not XGBoost)", hgb, hgb_results, FEATURE_COLS)

    # ---- Summary comparison ----
    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<30s} {'Random Forest':>15s} {'HGB':>15s}")
    print(f"  {'-'*60}")
    print(f"  {'Train accuracy':<30s} {rf_results['train_accuracy']:>15.4f} {hgb_results['train_accuracy']:>15.4f}")
    print(f"  {'Val accuracy':<30s} {rf_results['val_accuracy']:>15.4f} {hgb_results['val_accuracy']:>15.4f}")
    print(f"  {'Val balanced accuracy':<30s} {rf_results['val_balanced_accuracy']:>15.4f} {hgb_results['val_balanced_accuracy']:>15.4f}")
    print(f"  {'Val macro F1':<30s} {rf_results['val_macro_f1']:>15.4f} {hgb_results['val_macro_f1']:>15.4f}")

    # save summary
    summary_path = RESULTS_DIR / "phase5a_summary.txt"
    with open(summary_path, "w") as f:
        f.write("TEJAS Phase 5A — Feature-Based ML Classification Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write("NOTE: HistGradientBoostingClassifier is scikit-learn's native\n")
        f.write("gradient-boosting implementation. It is NOT XGBoost.\n\n")
        f.write(f"Data: {FEATURES_CSV}\n")
        f.write(f"Features: {len(FEATURE_COLS)} (34 total)\n")
        f.write(f"Target: condition_label (4 classes)\n")
        f.write(f"Train: {len(X_train)} cards, Val: {len(X_val)} cards\n")
        f.write(f"Split: well-level (existing split column)\n\n")
        f.write(f"{'Metric':<30s} {'Random Forest':>15s} {'HGB':>15s}\n")
        f.write(f"{'-'*60}\n")
        f.write(f"{'Train accuracy':<30s} {rf_results['train_accuracy']:>15.4f} {hgb_results['train_accuracy']:>15.4f}\n")
        f.write(f"{'Val accuracy':<30s} {rf_results['val_accuracy']:>15.4f} {hgb_results['val_accuracy']:>15.4f}\n")
        f.write(f"{'Val balanced accuracy':<30s} {rf_results['val_balanced_accuracy']:>15.4f} {hgb_results['val_balanced_accuracy']:>15.4f}\n")
        f.write(f"{'Val macro F1':<30s} {rf_results['val_macro_f1']:>15.4f} {hgb_results['val_macro_f1']:>15.4f}\n")
    print(f"\n  Summary saved: {summary_path}")

    print("\n" + "=" * 70)
    print("  Phase 5A complete. Models and results saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()
