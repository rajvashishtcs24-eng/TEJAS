"""
test_phase9_e2e.py
------------------
End-to-end integration test proving the entire TEJAS pipeline:
  Dynacard Input [200 pts] + Well Context
    -> Phase 5B 1D-CNN (Condition Probabilities)
    -> Phase 6 Risk & Engineering Assessment
    -> Phase 7.5 Physics-Informed Prediction
    -> Phase 8 Multi-Objective Optimization
    -> Final Operator Advisory Response

Saves outputs to:
  - results/phase9/e2e_api_test_response.json
  - results/phase9/phase9_integration_report.txt
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import app

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase9"
DATA_DIR = PROJECT_ROOT / "data"

client = TestClient(app)


def test_end_to_end_pipeline():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset sample cards
    cards = np.load(DATA_DIR / "processed" / "processed_cards_shape.npy")
    meta = pd.read_csv(DATA_DIR / "processed" / "processed_metadata.csv")

    test_cases = [
        ("Rod Floating", "WELL-019"),
        ("Fluid Pound", "WELL-005"),
        ("Gas Interference", "WELL-007"),
        ("Normal", "WELL-003"),
    ]

    all_responses = {}
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("TEJAS Phase 9 — End-to-End Pipeline Integration Test Report")
    report_lines.append("=" * 80)
    report_lines.append("\nPipeline: Dynacard Input -> CNN -> Risk -> Physics -> Optimizer -> API Response\n")

    for cond, well_id in test_cases:
        idx = meta.index[meta["condition_label"] == cond][0]
        row = meta.iloc[idx]
        card_points = cards[idx].tolist()

        payload = {
            "well_id": well_id,
            "card_points": card_points,
            "spm": float(row["SPM"]),
            "temperature_f": float(row["temperature"]),
            "viscosity_cp": float(row["viscosity"]),
            "fluid_level_ft": float(row["fluid_level"]),
            "gas_fraction": 0.28 if cond == "Gas Interference" else 0.05,
            "stroke_length_in": float(row["stroke_length"]),
            "pump_depth_ft": float(row["pump_depth"]),
        }

        # Send POST request to /api/analyze
        response = client.post("/api/analyze", json=payload)
        assert response.status_code == 200, f"API failed for {cond}: {response.text}"

        data = response.json()
        all_responses[cond] = data

        # Schema & Structure Assertions
        assert "diagnosis" in data
        assert "risk_assessment" in data
        assert "current_state" in data
        assert "optimization" in data
        assert "actionable_recommendation" in data
        assert "disclaimer" in data

        diag = data["diagnosis"]
        risk = data["risk_assessment"]
        cur = data["current_state"]
        opt = data["optimization"]
        rec = opt["recommended_scenario"]

        assert 0.0 <= diag["confidence"] <= 1.0
        assert sum(diag["probabilities"].values()) > 0.98
        assert 0.0 <= risk["overall_risk_score"] <= 100.0
        assert risk["severity"] in ["Low", "Medium", "High", "Critical"]
        assert cur["net_production_bpd"] > 0
        assert cur["polished_rod_power_kw"] > 0
        assert len(opt["selection_rationale"]) > 10

        # Log to report
        report_lines.append(f"[{cond.upper()} CASE: {well_id}]")
        report_lines.append("-" * 80)
        report_lines.append(f"  1. Diagnosis:       {diag['predicted_condition']} (Confidence: {diag['confidence']*100:.1f}%)")
        report_lines.append(f"     Probabilities:   {diag['probabilities']}")
        report_lines.append(f"  2. Risk Assessment: {risk['overall_risk_score']:.1f}/100 | Severity: {risk['severity']}")
        report_lines.append(f"     Root Causes:     {'; '.join(risk['contributing_factors']) or 'Nominal'}")
        report_lines.append(f"  3. Current State:   Prod: {cur['net_production_bpd']:.1f} bpd | Power: {cur['polished_rod_power_kw']:.2f} kW | SEC: {cur['specific_energy_kwh_per_bbl']:.2f} kWh/bbl")
        report_lines.append(f"  4. Recommended Opt: [{rec['scenario_id']}] {rec['modifications']}")
        report_lines.append(f"     Expected Delta:  Delta Prod: {rec['delta_production_bpd']:+.1f} bpd | Delta Power: {rec['delta_power_kw']:+.2f} kW | Delta Risk: {rec['delta_risk']:+.1f} pts")
        report_lines.append(f"     Target State:    Prod: {rec['net_production_bpd']:.1f} bpd | Power: {rec['polished_rod_power_kw']:.2f} kW | Risk: {rec['risk_score']:.1f} ({rec['severity']})")
        report_lines.append(f"  5. Selection Reason: {opt['selection_rationale']}")
        report_lines.append(f"  6. Operator Action: {data['actionable_recommendation']}")
        report_lines.append("-" * 80 + "\n")

    # 2. Test Sample Wells endpoint
    res_samples = client.get("/api/sample_wells")
    assert res_samples.status_code == 200
    samples_json = res_samples.json()
    assert len(samples_json["samples"]) == 4

    # 3. Test Root Dashboard HTML endpoint
    res_dash = client.get("/")
    assert res_dash.status_code == 200
    assert "TEJAS — Thermal-Enabled Well Optimization" in res_dash.text

    # 4. Save JSON Artifact
    json_out = RESULTS_DIR / "e2e_api_test_response.json"
    with open(json_out, "w") as f:
        json.dump(all_responses, f, indent=2)
    print(f"Saved E2E API response JSON to {json_out}")

    # 5. Save Report Text
    report_out = RESULTS_DIR / "phase9_integration_report.txt"
    with open(report_out, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Saved E2E integration report to {report_out}")

    print("\n" + "=" * 80)
    print("ALL PHASE 9 END-TO-END INTEGRATION TESTS PASSED (100% SUCCESS)")
    print("=" * 80)


if __name__ == "__main__":
    test_end_to_end_pipeline()
