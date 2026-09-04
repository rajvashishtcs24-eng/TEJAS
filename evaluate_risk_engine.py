"""
evaluate_risk_engine.py
-----------------------
Evaluates the Phase 6 Risk & Engineering Assessment Engine on the
400 dynamometer cards using predictions from the Phase 5B CNN model.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_phase5b_cnn import DynaCardCNN, CLASS_ORDER, LABEL2IDX
from data_pipeline.risk_engine import TEJASRiskEngine, AssessmentResult

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "phase5"
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    cards_shape = np.load(DATA_DIR / "processed" / "processed_cards_shape.npy")
    features_df = pd.read_csv(DATA_DIR / "features" / "dynacard_features.csv")

    assert len(cards_shape) == len(features_df) == 400

    # 2. Load trained CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynaCardCNN(num_classes=4).to(device)
    model_path = MODELS_DIR / "cnn_phase5b.pt"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Compute CNN probabilities
    # cards_shape is [400, 200, 2] -> transpose to [400, 2, 200]
    tensor_cards = torch.tensor(cards_shape.transpose(0, 2, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(tensor_cards)
        probs = F.softmax(logits, dim=1).cpu().numpy()

    # 4. Run Risk Engine Assessment
    risk_engine = TEJASRiskEngine()
    results = []

    for i in range(len(features_df)):
        row = features_df.iloc[i]
        card_id = row["card_id"]
        well_id = row["well_id"]

        card_probs = {CLASS_ORDER[j]: float(probs[i, j]) for j in range(4)}
        context = row.to_dict()

        assessment = risk_engine.assess_card(card_id, well_id, card_probs, context)
        results.append(assessment)

    # 5. Format results into DataFrame
    results_records = []
    for a in results:
        results_records.append({
            "card_id": a.card_id,
            "well_id": a.well_id,
            "actual_condition": features_df.loc[features_df["card_id"] == a.card_id, "condition_label"].values[0],
            "predicted_condition": a.predicted_condition,
            "confidence": a.confidence,
            "prob_normal": a.probabilities.get("Normal", 0.0),
            "prob_rod_floating": a.probabilities.get("Rod Floating", 0.0),
            "prob_fluid_pound": a.probabilities.get("Fluid Pound", 0.0),
            "prob_gas_interference": a.probabilities.get("Gas Interference", 0.0),
            "risk_score": a.risk_score,
            "severity": a.severity,
            "split": features_df.loc[features_df["card_id"] == a.card_id, "split"].values[0],
            "engineering_explanation": a.engineering_explanation,
            "recommended_action": a.recommended_action,
            "contributing_factors": " | ".join(a.contributing_factors),
        })

    res_df = pd.DataFrame(results_records)
    out_csv = RESULTS_DIR / "risk_assessment_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(res_df)} risk assessments to {out_csv}")

    # 6. Generate Summary and Breakdown
    val_res = res_df[res_df["split"] == "val"]

    print("\n" + "=" * 70)
    print("TEJAS Phase 6 — Risk Assessment Summary")
    print("=" * 70)
    print("\n--- Overall Severity Distribution by Condition (All 400 cards) ---")
    sev_crosstab = pd.crosstab(res_df["actual_condition"], res_df["severity"], margins=True)
    print(sev_crosstab)

    print("\n--- Mean Risk Score by Condition ---")
    mean_risks = res_df.groupby("actual_condition")["risk_score"].agg(["mean", "min", "max", "std"]).round(2)
    print(mean_risks)

    print("\n--- Validation Set (81 cards) Severity Distribution ---")
    val_sev = pd.crosstab(val_res["actual_condition"], val_res["severity"], margins=True)
    print(val_sev)

    # Save human-readable report
    report_path = RESULTS_DIR / "risk_assessment_report.txt"
    with open(report_path, "w") as f:
        f.write("TEJAS Phase 6 — Risk & Engineering Assessment Report\n")
        f.write("=" * 70 + "\n\n")
        f.write("1. SUMMARY OF SEVERITY TIERS BY CONDITION (400 CARDS)\n")
        f.write(sev_crosstab.to_string() + "\n\n")
        f.write("2. RISK SCORE STATISTICS BY CONDITION\n")
        f.write(mean_risks.to_string() + "\n\n")
        f.write("3. VALIDATION SET SEVERITY DISTRIBUTION (81 CARDS)\n")
        f.write(val_sev.to_string() + "\n\n")
        f.write("4. REPRESENTATIVE EXAMPLE ASSESSMENTS BY CONDITION\n")
        f.write("-" * 70 + "\n")

        for cond in CLASS_ORDER:
            sample = res_df[res_df["actual_condition"] == cond].iloc[0]
            f.write(f"\n[CONDITION: {cond.upper()}] - Card {sample['card_id']} ({sample['well_id']})\n")
            f.write(f"  Predicted:      {sample['predicted_condition']} (confidence: {sample['confidence']:.2%})\n")
            f.write(f"  Risk Score:     {sample['risk_score']} / 100  (Severity: {sample['severity']})\n")
            f.write(f"  Probabilities:  Normal={sample['prob_normal']:.2f}, RF={sample['prob_rod_floating']:.2f}, FP={sample['prob_fluid_pound']:.2f}, GI={sample['prob_gas_interference']:.2f}\n")
            f.write(f"  Explanation:    {sample['engineering_explanation']}\n")
            f.write(f"  Recommendation: {sample['recommended_action']}\n")

    print(f"\nWrote detailed report to {report_path}")


if __name__ == "__main__":
    main()
