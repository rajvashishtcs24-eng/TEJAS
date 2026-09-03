"""
generate_dataset.py
--------------------
Batch-generates a SYNTHETIC (physics-simulated, not field-measured)
dynamometer-card dataset for TEJAS Phase 1, following the schema agreed
in the project handoff:

card_id, well_id, timestamp, position[], load[], SPM, stroke_length,
temperature, viscosity, fluid_level, pump_depth, production_rate,
condition_label, risk_level, recommended_action

Notes on scope (Phase 1 only):
- risk_level / recommended_action below are a placeholder heuristic label,
  NOT the actual risk/decision engine (that's Phase 8/9). They exist only
  so the schema is populated end-to-end and downstream phases have
  something to train/validate against; they must be revisited once the
  real risk engine is built.
- All rows are tagged data_source = "synthetic_physics_v1" so this can
  never be confused with real field data later in the pipeline.
"""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from srp_physics import WellConfig
from conditions import GENERATORS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(42)

CONDITION_WEIGHTS = {
    "Normal": 0.40,
    "Rod Floating": 0.22,
    "Fluid Pound": 0.20,
    "Gas Interference": 0.18,
}


def sample_well_baseline(rng) -> dict:
    """PATCH (Phase 1 well-identity fix): one persistent baseline per well_id.

    Mechanical/structural fields are fixed for the life of the well (real
    stroke length, pump depth, rod weight, plunger size, friction don't
    change day-to-day). Dynamic/operational fields get a 'baseline' typical
    operating point that individual cards drift around over time.
    """
    return {
        "stroke_length_in": float(rng.uniform(74, 168)),
        "pump_depth_ft": float(rng.uniform(3000, 6000)),
        "rod_weight_per_ft": float(rng.uniform(1.8, 2.6)),
        "fluid_specific_gravity": float(rng.uniform(0.87, 0.97)),
        "plunger_diameter_in": float(rng.choice([1.5, 1.75, 2.0, 2.25, 2.5])),
        "friction_lbf": float(rng.uniform(250, 600)),
        "spm_baseline": float(rng.uniform(4, 8)),
        "temperature_baseline": float(rng.uniform(120, 170)),
        "viscosity_baseline": float(rng.uniform(200, 600)),
        "fluid_level_baseline": float(rng.uniform(2000, 4000)),
        "production_baseline": float(rng.uniform(20, 100)),
    }


def sample_well_config(baseline: dict, condition: str, rng) -> WellConfig:
    """Build one card's WellConfig from its well's persistent baseline.

    Mechanical fields are copied exactly (persistent, no drift).
    Dynamic fields = baseline + small realistic drift (normal fluctuation),
    plus a condition-informed excursion for fault cards (the same
    directional correlations as before - e.g. rod floating skews toward
    higher SPM + higher viscosity - but now expressed as a deviation from
    THIS well's own baseline rather than an independent random redraw).
    """
    spm = baseline["spm_baseline"] + rng.normal(0, 0.4)
    temperature = baseline["temperature_baseline"] + rng.normal(0, 4)
    viscosity = baseline["viscosity_baseline"] + rng.normal(0, 30)
    fluid_level = baseline["fluid_level_baseline"] + rng.normal(0, 100)
    production = baseline["production_baseline"] + rng.normal(0, 5)

    if condition == "Rod Floating":
        spm += rng.uniform(2, 4)
        viscosity += rng.uniform(200, 500)
        temperature -= rng.uniform(10, 30)  # cooler -> more viscous
    elif condition == "Fluid Pound":
        fluid_level -= rng.uniform(500, 1500)  # drawdown / declining fillage
    elif condition == "Gas Interference":
        pass  # gas_fraction is handled inside conditions.py itself

    spm = float(np.clip(spm, 3, 12))
    temperature = float(np.clip(temperature, 80, 200))
    viscosity = float(np.clip(viscosity, 50, 1200))
    fluid_level = float(np.clip(fluid_level, 200, 5000))
    production = float(np.clip(production, 5, 150))

    return WellConfig(
        stroke_length_in=baseline["stroke_length_in"],
        spm=spm,
        pump_depth_ft=baseline["pump_depth_ft"],
        rod_weight_per_ft=baseline["rod_weight_per_ft"],
        fluid_specific_gravity=baseline["fluid_specific_gravity"],
        plunger_diameter_in=baseline["plunger_diameter_in"],
        friction_lbf=baseline["friction_lbf"],
        fluid_level_ft=fluid_level,
        temperature_F=temperature,
        viscosity_cp=viscosity,
        production_bpd=production,
    )


def heuristic_risk(condition: str, severity: float) -> tuple[str, str]:
    """Placeholder risk/decision heuristic - NOT the Phase 8/9 risk engine."""
    if condition == "Normal":
        return "Low", "Continue monitoring"
    if severity >= 0.6:
        risk = "High"
    elif severity >= 0.3:
        risk = "Medium"
    else:
        risk = "Low"

    action_map = {
        "Rod Floating": "Reduce SPM / adjust VFD; evaluate CSS timing" if risk != "Low" else "Continue monitoring",
        "Fluid Pound": "Reduce SPM to match inflow; check pump-off control" if risk != "Low" else "Continue monitoring",
        "Gas Interference": "Evaluate gas separation / SPM adjustment" if risk != "Low" else "Continue monitoring",
    }
    return risk, action_map.get(condition, "Continue monitoring")


def generate_dataset(n_cards: int = 400, n_points: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    conditions = list(CONDITION_WEIGHTS.keys())
    weights = list(CONDITION_WEIGHTS.values())

    rows = []
    start_time = datetime(2026, 1, 1)
    well_ids = [f"WELL-{i:03d}" for i in range(1, 21)]

    # PATCH: one persistent baseline per well_id, sampled once, reused for
    # every card assigned to that well.
    well_baselines = {wid: sample_well_baseline(rng) for wid in well_ids}

    for i in range(n_cards):
        well_id = rng.choice(well_ids)
        condition = rng.choice(conditions, p=weights)
        well = sample_well_config(well_baselines[well_id], condition, rng)
        gen_fn = GENERATORS[condition]
        theta, pos, load, meta = gen_fn(well, n_points=n_points, rng=rng)

        risk_level, action = heuristic_risk(condition, meta.get("severity", 0.0))

        rows.append({
            "card_id": str(uuid.uuid4())[:8],
            "well_id": well_id,
            "timestamp": (start_time + timedelta(minutes=15 * i)).isoformat(),
            "position": json.dumps(np.round(pos, 2).tolist()),
            "load": json.dumps(np.round(load, 2).tolist()),
            "SPM": round(well.spm, 2),
            "stroke_length": round(well.stroke_length_in, 1),
            "temperature": round(well.temperature_F, 1),
            "viscosity": round(well.viscosity_cp, 1),
            "fluid_level": round(well.fluid_level_ft, 1),
            "pump_depth": round(well.pump_depth_ft, 1),
            "production_rate": round(well.production_bpd, 1),
            "condition_label": condition,
            "risk_level": risk_level,
            "recommended_action": action,
            "severity": round(meta.get("severity", 0.0), 3),
            "data_source": "synthetic_physics_v1",
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(n_cards=400, n_points=200, seed=42)
    out_path = PROJECT_ROOT / "data" / "raw" / "dynacards_synthetic_v1.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic cards to {out_path}")
    print(df["condition_label"].value_counts())
    print(df["risk_level"].value_counts())
