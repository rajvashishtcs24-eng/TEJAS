"""
app.py
------
TEJAS Phase 9: End-to-End Integrated FastAPI Backend & Operator Decision Dashboard.

Integrates:
  - Phase 5B 1D-CNN (Card Classification)
  - Phase 6 Risk & Engineering Assessment Engine
  - Phase 7.5 Physics-Informed Well Prediction
  - Phase 8 Multi-Objective Optimization & Digital Twin Engine

Endpoints:
  - POST /api/analyze        : Full end-to-end well diagnosis & optimization
  - GET  /api/sample_wells   : Preset real synthetic cases for quick operator demo
  - GET  /                   : Operator Decision Dashboard
"""

from pathlib import Path
import json
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from train_phase5b_cnn import DynaCardCNN, CLASS_ORDER
from data_pipeline.risk_engine import TEJASRiskEngine
from data_pipeline.well_prediction import (
    PhysicsWellPredictor,
    WellOperatingState,
    WellPredictionResult,
)
from data_pipeline.optimizer import TEJASWellOptimizer, OptimizationRecommendation

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

app = FastAPI(
    title="TEJAS — Thermal-Enabled Well Optimization & Decision Support",
    description="End-to-End Sucker-Rod Pumping Diagnostic & Digital Twin Engine",
    version="9.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------
# Model & Engine Initialization (Loaded Once at Startup)
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Phase 5B CNN
cnn_model = DynaCardCNN(num_classes=4).to(device)
cnn_path = MODELS_DIR / "cnn_phase5b.pt"
if cnn_path.exists():
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_model.eval()
    print(f"[TEJAS] Loaded Phase 5B CNN weights from {cnn_path}")
else:
    print(f"[TEJAS WARNING] Weights not found at {cnn_path}")

# 2. Phase 6 Risk Engine
risk_engine = TEJASRiskEngine()

# 3. Phase 7.5 Physics Predictor
physics_predictor = PhysicsWellPredictor()

# 4. Phase 8 Optimizer
optimizer = TEJASWellOptimizer(predictor=physics_predictor)


# -----------------------------------------------------------------
# Request & Response Schemas
# -----------------------------------------------------------------
class WellAnalysisRequest(BaseModel):
    well_id: str = Field(default="WELL-019", description="Well identifier")
    card_points: List[List[float]] = Field(
        ...,
        description="200-point dynamometer card [[norm_pos, norm_load], ...], shape [200, 2]",
    )
    spm: float = Field(default=8.5, description="Pumping speed in strokes per minute")
    temperature_f: float = Field(default=125.0, description="Wellbore temperature in °F")
    viscosity_cp: float = Field(default=850.0, description="Crude oil viscosity in cP")
    fluid_level_ft: float = Field(default=3400.0, description="Annular fluid level in ft above pump")
    gas_fraction: float = Field(default=0.05, description="Free gas fraction in pump intake [0-1]")
    stroke_length_in: float = Field(default=120.0, description="Polished rod stroke length in inches")
    pump_depth_ft: float = Field(default=5000.0, description="Pump setting depth in ft")
    rod_weight_per_ft: float = Field(default=2.20, description="Rod string weight per foot in lb/ft")
    plunger_diameter_in: float = Field(default=2.0, description="Pump plunger diameter in inches")
    fluid_specific_gravity: float = Field(default=0.92, description="Produced fluid specific gravity")
    friction_mechanical_lbf: float = Field(default=400.0, description="Mechanical friction load in lbf")


class DiagnosisOutput(BaseModel):
    predicted_condition: str
    confidence: float
    probabilities: Dict[str, float]


class RiskAssessmentOutput(BaseModel):
    rod_floating_risk: float
    fluid_pound_risk: float
    gas_interference_risk: float
    overall_risk_score: float
    severity: str
    engineering_explanation: str
    contributing_factors: List[str]


class CurrentStateOutput(BaseModel):
    net_production_bpd: float
    theoretical_displacement_bpd: float
    volumetric_efficiency_pct: float
    polished_rod_power_kw: float
    specific_energy_kwh_per_bbl: float
    pprl_lbf: float
    mprl_lbf: float
    load_range_lbf: float


class ScenarioOption(BaseModel):
    scenario_id: str
    scenario_type: str
    modifications: Dict[str, Any]
    net_production_bpd: float
    polished_rod_power_kw: float
    specific_energy_kwh_per_bbl: float
    risk_score: float
    severity: str
    delta_production_bpd: float
    delta_power_kw: float
    delta_risk: float


class OptimizationOutput(BaseModel):
    recommended_scenario: ScenarioOption
    max_production_scenario: ScenarioOption
    max_efficiency_scenario: ScenarioOption
    utility_score: float
    selection_rationale: str
    operational_warnings: List[str]


class WellAnalysisResponse(BaseModel):
    well_id: str
    diagnosis: DiagnosisOutput
    risk_assessment: RiskAssessmentOutput
    current_state: CurrentStateOutput
    optimization: OptimizationOutput
    actionable_recommendation: str
    disclaimer: str = (
        "ADVISORY NOTICE: All assessments, predictions, and recommendations are "
        "decision support outputs generated from physics models and neural network pattern recognition. "
        "They do NOT constitute autonomous control. Review operational changes with engineering personnel."
    )


# -----------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------
@app.post("/api/analyze", response_model=WellAnalysisResponse)
def analyze_well(request: WellAnalysisRequest):
    """
    End-to-End Pipeline:
    Dynacard -> Phase 5B CNN -> Phase 6 Risk -> Phase 7.5 Physics -> Phase 8 Optimizer -> Advisory Output
    """
    card_arr = np.array(request.card_points, dtype=np.float32)
    if card_arr.shape != (200, 2):
        raise HTTPException(
            status_code=400,
            detail=f"Expected card_points shape (200, 2), got {card_arr.shape}",
        )

    # 1. Phase 5B CNN Inference
    # Input tensor shape: (1, 2, 200)
    card_tensor = torch.tensor(
        card_arr.transpose(1, 0)[np.newaxis, :, :], dtype=torch.float32
    ).to(device)

    with torch.no_grad():
        logits = cnn_model(card_tensor)
        probs_arr = F.softmax(logits, dim=1).cpu().numpy()[0]

    probabilities = {CLASS_ORDER[i]: float(probs_arr[i]) for i in range(4)}
    pred_condition = max(probabilities, key=probabilities.get)
    confidence = float(probabilities[pred_condition])

    # 2. Phase 6 Risk Engine Assessment
    operating_ctx = {
        "SPM": request.spm,
        "temperature": request.temperature_f,
        "viscosity": request.viscosity_cp,
        "fluid_level": request.fluid_level_ft,
        "pump_depth": request.pump_depth_ft,
        "gas_fraction": request.gas_fraction,
        "stroke_length": request.stroke_length_in,
    }

    risk_eval = risk_engine.assess_card(
        card_id="ONLINE_INPUT",
        well_id=request.well_id,
        probabilities=probabilities,
        context=operating_ctx,
    )

    r_rf, f_rf = risk_engine.evaluate_rod_floating_risk(probabilities.get("Rod Floating", 0.0), operating_ctx)
    r_fp, f_fp = risk_engine.evaluate_fluid_pound_risk(probabilities.get("Fluid Pound", 0.0), operating_ctx)
    r_gi, f_gi = risk_engine.evaluate_gas_interference_risk(probabilities.get("Gas Interference", 0.0), operating_ctx)

    # 3. Phase 7.5 Physics Evaluation of Current State
    state_obj = WellOperatingState(
        well_id=request.well_id,
        spm=request.spm,
        temperature_f=request.temperature_f,
        viscosity_cp=request.viscosity_cp,
        fluid_level_ft=request.fluid_level_ft,
        gas_fraction=request.gas_fraction,
        stroke_length_in=request.stroke_length_in,
        pump_depth_ft=request.pump_depth_ft,
        rod_weight_per_ft=request.rod_weight_per_ft,
        plunger_diameter_in=request.plunger_diameter_in,
        fluid_specific_gravity=request.fluid_specific_gravity,
        friction_mechanical_lbf=request.friction_mechanical_lbf,
    )

    current_physics = physics_predictor.evaluate_well(state_obj)

    # 4. Phase 8 Multi-Objective Optimization
    opt_rec: OptimizationRecommendation = optimizer.optimize_well(state_obj)

    def to_scenario_option(c) -> ScenarioOption:
        return ScenarioOption(
            scenario_id=c.scenario_id,
            scenario_type=c.scenario_type,
            modifications=c.modifications,
            net_production_bpd=c.state.net_production_bpd,
            polished_rod_power_kw=c.state.polished_rod_power_kw,
            specific_energy_kwh_per_bbl=c.state.specific_energy_kwh_per_bbl,
            risk_score=c.state.composite_risk_score,
            severity=c.state.severity,
            delta_production_bpd=c.delta.delta_production_bpd,
            delta_power_kw=c.delta.delta_power_kw,
            delta_risk=c.delta.delta_composite_risk,
        )

    best_opt = to_scenario_option(opt_rec.recommended_scenario)
    max_p_opt = to_scenario_option(opt_rec.max_production_scenario)
    max_e_opt = to_scenario_option(opt_rec.max_efficiency_scenario)

    return WellAnalysisResponse(
        well_id=request.well_id,
        diagnosis=DiagnosisOutput(
            predicted_condition=pred_condition,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probabilities.items()},
        ),
        risk_assessment=RiskAssessmentOutput(
            rod_floating_risk=round(r_rf, 1),
            fluid_pound_risk=round(r_fp, 1),
            gas_interference_risk=round(r_gi, 1),
            overall_risk_score=round(risk_eval.risk_score, 1),
            severity=risk_eval.severity,
            engineering_explanation=risk_eval.engineering_explanation,
            contributing_factors=risk_eval.contributing_factors,
        ),
        current_state=CurrentStateOutput(
            net_production_bpd=round(current_physics.net_production_bpd, 1),
            theoretical_displacement_bpd=round(current_physics.theoretical_displacement_bpd, 1),
            volumetric_efficiency_pct=round(current_physics.volumetric_efficiency * 100.0, 1),
            polished_rod_power_kw=round(current_physics.polished_rod_power_kw, 2),
            specific_energy_kwh_per_bbl=round(current_physics.specific_energy_kwh_per_bbl, 2),
            pprl_lbf=round(current_physics.pprl_lbf, 1),
            mprl_lbf=round(current_physics.mprl_lbf, 1),
            load_range_lbf=round(current_physics.load_range_lbf, 1),
        ),
        optimization=OptimizationOutput(
            recommended_scenario=best_opt,
            max_production_scenario=max_p_opt,
            max_efficiency_scenario=max_e_opt,
            utility_score=round(opt_rec.recommended_scenario.utility_score, 4),
            selection_rationale=opt_rec.selection_rationale,
            operational_warnings=opt_rec.operational_warnings,
        ),
        actionable_recommendation=risk_eval.recommended_action,
    )


@app.get("/api/sample_wells")
def get_sample_wells():
    """
    Returns representative sample wells from our dataset with real 200-point cards.
    """
    shape_path = DATA_DIR / "processed" / "processed_cards_shape.npy"
    meta_path = DATA_DIR / "processed" / "processed_metadata.csv"

    if not (shape_path.exists() and meta_path.exists()):
        raise HTTPException(status_code=404, detail="Dataset files not found on disk")

    cards = np.load(shape_path)
    meta = pd.read_csv(meta_path)

    samples = []
    conditions = ["Rod Floating", "Fluid Pound", "Gas Interference", "Normal"]

    for cond in conditions:
        idx_matches = meta.index[meta["condition_label"] == cond]
        if len(idx_matches) > 0:
            idx = idx_matches[0]
            row = meta.iloc[idx]
            samples.append({
                "label": cond,
                "well_id": row["well_id"],
                "card_id": row["card_id"],
                "condition_label": row["condition_label"],
                "spm": float(row["SPM"]),
                "temperature_f": float(row["temperature"]),
                "viscosity_cp": float(row["viscosity"]),
                "fluid_level_ft": float(row["fluid_level"]),
                "gas_fraction": 0.30 if cond == "Gas Interference" else 0.05,
                "stroke_length_in": float(row["stroke_length"]),
                "pump_depth_ft": float(row["pump_depth"]),
                "card_points": cards[idx].tolist(),  # [200, 2]
            })

    return JSONResponse(content={"samples": samples})


@app.get("/", response_class=HTMLResponse)
def get_dashboard():
    """
    Serves the TEJAS Operator Decision Dashboard HTML application.
    """
    dashboard_path = PROJECT_ROOT / "app" / "dashboard.html"
    if dashboard_path.exists():
        return HTMLResponse(content=dashboard_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>TEJAS Operator Dashboard</h1><p>Dashboard HTML not found.</p>")
