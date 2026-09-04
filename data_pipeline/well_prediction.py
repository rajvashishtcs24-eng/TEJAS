"""
well_prediction.py
------------------
TEJAS Phase 7.5: Physics-Informed Well Prediction & What-If Scenario Layer.
(Audited & Calibrated for physical consistency and transparent coupling).

Governing Physics Chain:
  Temperature (T)
    -> [HEURISTIC_ASSUMP: Andrade/Arrhenius Rheology] -> Viscosity (mu)
    -> [PHYSICS_DERIVED] -> Fluid Mobility Index (1/mu)
    -> [PHYSICS_DERIVED: Couette Shear & SHM Kinematics] -> Viscous Drag & Inertial Loading
    -> [PHYSICS_DERIVED + HEURISTIC: Inflow-Fillage Coupling] -> Valve Efficiency & Net Production
    -> [PHYSICS_DERIVED: Cycle Work Envelope] -> Power (kW) & Specific Energy (kWh/bbl)
    -> [PHYSICS_INFORMED DIAGNOSTIC] -> Condition Risk Scores (0-100)

Explicit Taxonomy of Quantities:
  - [SYNTHETIC_INPUT]  : Direct measured or synthetic operational input (well geometry, SPM, T, FL)
  - [PHYSICS_DERIVED]   : Derived directly via fundamental physics (statics, kinematics, hydraulics)
  - [HEURISTIC_ASSUMP]  : Empirical engineering coupling parameter (viscosity beta, valve choke, submergence threshold)
  - [PREDICTED_OUTPUT]  : Simulated state, delta, and diagnostic risk score
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

# Physical Constants [PHYSICS_DERIVED]
G_FTPS2 = 32.174          # standard gravity, ft/s^2
STEEL_SG = 7.85           # specific gravity of steel rods
PSI_PER_FT_WATER = 0.433  # hydrostatic pressure gradient of fresh water, psi/ft
API_DISPLACEMENT_CONST = 0.11662  # bpd per (in^2 * in * spm) from API RP 11L


@dataclass
class WellOperatingState:
    """Operating parameters for a sucker-rod pump well."""
    # Mechanical Architecture [SYNTHETIC_INPUT - Fixed Well Properties]
    stroke_length_in: float = 120.0
    pump_depth_ft: float = 4500.0
    rod_weight_per_ft: float = 2.16
    plunger_diameter_in: float = 2.0
    fluid_specific_gravity: float = 0.92
    friction_mechanical_lbf: float = 400.0

    # Operational Variables [SYNTHETIC_INPUT / MODIFIABLE]
    spm: float = 6.0
    temperature_f: float = 135.0
    viscosity_cp: float = 450.0
    fluid_level_ft: float = 3200.0
    gas_fraction: float = 0.05
    well_id: str = "WELL-001"


@dataclass
class WellPredictionResult:
    """Comprehensive physics and diagnostic prediction for a well operating state."""
    # 1. Rheology & Mobility
    temperature_f: float
    viscosity_cp: float
    mobility_index: float  # [PHYSICS_DERIVED]: relative to 400 cP baseline

    # 2. Kinematics & SRP Loading [PHYSICS_DERIVED]
    spm: float
    omega_rad_s: float
    buoyant_rod_weight_lbf: float
    fluid_load_lbf: float
    inertia_amplitude_lbf: float
    viscous_drag_lbf: float
    pprl_lbf: float
    mprl_lbf: float
    load_range_lbf: float

    # 3. Volumetric Delivery & Production [PHYSICS_DERIVED + HEURISTIC]
    theoretical_displacement_bpd: float
    pump_fillage_fraction: float
    valve_viscosity_efficiency: float
    rod_float_stroke_efficiency: float
    volumetric_efficiency: float
    net_production_bpd: float

    # 4. Energy & Power Consumption [PHYSICS_DERIVED]
    polished_rod_power_hp: float
    polished_rod_power_kw: float
    specific_energy_kwh_per_bbl: float
    cycle_work_lbf_in: float

    # 5. Physics-Informed Diagnostic Risk Scores [0-100]
    rod_floating_risk: float
    fluid_pound_risk: float
    gas_interference_risk: float
    composite_risk_score: float
    severity: str

    # Engineering Summary
    diagnostics_summary: str


@dataclass
class ScenarioDelta:
    """Quantitative deltas between Scenario and Baseline."""
    delta_spm: float
    delta_temperature_f: float
    delta_viscosity_cp: float
    pct_viscosity_change: float
    delta_mobility_index: float

    delta_pprl_lbf: float
    delta_mprl_lbf: float
    delta_load_range_lbf: float

    delta_production_bpd: float
    pct_production_change: float

    delta_power_kw: float
    delta_specific_energy_kwh_bbl: float

    delta_rod_floating_risk: float
    delta_fluid_pound_risk: float
    delta_gas_interference_risk: float
    delta_composite_risk: float


@dataclass
class ScenarioComparison:
    """Complete Scenario Analysis containing Baseline, Scenario, and Delta."""
    scenario_name: str
    scenario_description: str
    baseline: WellPredictionResult
    scenario: WellPredictionResult
    delta: ScenarioDelta
    engineering_insights: List[str]


class PhysicsWellPredictor:
    """
    Physics-informed prediction and what-if simulation engine for heavy-oil SRP wells.
    """

    def __init__(
        self,
        temp_viscosity_beta: float = 0.028,
        critical_submergence_ft: float = 2800.0,
        card_fullness_factor: float = 0.72,
    ):
        """
        Args:
            temp_viscosity_beta: [HEURISTIC_ASSUMP] Thermal viscosity exponential decay constant (1/°F).
                                 Heavy oil empirical range: 0.022 - 0.035 /°F.
            critical_submergence_ft: [HEURISTIC_ASSUMP] Annular fluid level required above pump for 100% fillage.
            card_fullness_factor: [HEURISTIC_ASSUMP] Ratio of actual dynacard work loop area to (PPRL-MPRL)*Stroke.
        """
        self.beta = temp_viscosity_beta
        self.fl_crit = critical_submergence_ft
        self.card_fullness = card_fullness_factor

    @staticmethod
    def _clip(val: float, low: float = 0.0, high: float = 1.0) -> float:
        return float(np.clip(val, low, high))

    def predict_viscosity_from_temp(
        self, temp_new_f: float, temp_base_f: float, visc_base_cp: float
    ) -> float:
        """
        [HEURISTIC_ASSUMP] Heavy crude temperature-viscosity relationship:
        mu(T) = mu_0 * exp(-beta * (T - T_0))
        """
        delta_t = temp_new_f - temp_base_f
        visc_new = visc_base_cp * np.exp(-self.beta * delta_t)
        return float(np.clip(visc_new, 20.0, 5000.0))

    def evaluate_well(
        self,
        state: WellOperatingState,
        compute_thermal_viscosity: bool = False,
        base_temp_f: float = 135.0,
        base_visc_cp: float = 450.0,
    ) -> WellPredictionResult:
        """
        Full physics evaluation of a well operating state.
        """
        # 1. Rheology & Mobility [HEURISTIC_ASSUMP + PHYSICS_DERIVED]
        if compute_thermal_viscosity:
            visc = self.predict_viscosity_from_temp(
                state.temperature_f, base_temp_f, base_visc_cp
            )
        else:
            visc = state.viscosity_cp

        # Mobility index relative to nominal 400 cP crude
        mobility_idx = 400.0 / max(visc, 1e-2)

        # 2. Structural & Hydrostatic Forces [PHYSICS_DERIVED]
        buoyancy_factor = 1.0 - (state.fluid_specific_gravity / STEEL_SG)
        w_air = state.rod_weight_per_ft * state.pump_depth_ft
        w_buoyant = w_air * buoyancy_factor

        plunger_area_in2 = (np.pi / 4.0) * (state.plunger_diameter_in ** 2)
        fluid_gradient = PSI_PER_FT_WATER * state.fluid_specific_gravity
        f_fluid = fluid_gradient * state.fluid_level_ft * plunger_area_in2

        # 3. Dynamic Forces & Kinematics [PHYSICS_DERIVED]
        omega = state.spm * 2.0 * np.pi / 60.0
        stroke_ft = state.stroke_length_in / 12.0
        mass_lbm = w_air / G_FTPS2
        f_inertia = mass_lbm * (stroke_ft / 2.0) * (omega ** 2)

        # Couette annular shear viscous drag approximation [PHYSICS_DERIVED]
        k_drag = 0.25 * (state.pump_depth_ft / 4500.0) * (state.stroke_length_in / 120.0)
        f_viscous_drag = k_drag * (visc / 400.0) * (state.spm / 6.0) * 350.0
        total_friction = state.friction_mechanical_lbf + f_viscous_drag

        # Polished Rod Loads [PHYSICS_DERIVED]
        pprl = w_buoyant + f_fluid + f_inertia + total_friction
        mprl = w_buoyant - f_inertia - total_friction
        load_range = pprl - mprl

        # 4. Diagnostic Risk Scores [PHYSICS_INFORMED 0-100]
        # (Evaluated prior to production delivery so rod-float stroke loss can be coupled)
        rf_drag_ratio = (f_viscous_drag + f_inertia) / max(w_buoyant * 0.35, 1.0)
        rf_spm_factor = self._clip((state.spm - 5.0) / 4.5)
        rf_visc_factor = self._clip((visc - 300.0) / 600.0)
        risk_rf = 100.0 * self._clip(
            0.40 * rf_visc_factor + 0.35 * rf_spm_factor + 0.25 * rf_drag_ratio
        )

        # Submergence & fillage factor [PHYSICS_DERIVED + HEURISTIC]
        submergence_factor = self._clip(state.fluid_level_ft / self.fl_crit, 0.20, 1.0)
        gas_cushion_factor = self._clip(1.0 - state.gas_fraction * 1.5, 0.20, 1.0)
        fillage = submergence_factor * gas_cushion_factor

        risk_fp = 100.0 * self._clip((1.0 - fillage) * 1.3)
        risk_gi = 100.0 * self._clip(
            state.gas_fraction * 2.5 * (3000.0 / max(state.fluid_level_ft, 500.0))
        )
        composite_risk = max(risk_rf, risk_fp, risk_gi, 5.0)

        # 5. Volumetric Delivery & Production Coupling [PHYSICS_DERIVED + HEURISTIC]
        # Theoretical volumetric displacement (API RP 11L)
        q_theor = (
            API_DISPLACEMENT_CONST
            * (state.plunger_diameter_in ** 2)
            * state.stroke_length_in
            * state.spm
        )

        # Valve intake throttling loss (high viscosity chokes standing valve entry)
        # [HEURISTIC_ASSUMP grounded in valve flow loss]
        eta_valve = self._clip(1.0 - 0.12 * ((visc - 300.0) / 700.0), 0.76, 1.0)

        # Rod floating effective stroke loss (downstroke float shortens effective plunger travel)
        # [HEURISTIC_ASSUMP grounded in kinematics lag]
        eta_rf_stroke = self._clip(1.0 - 0.20 * ((risk_rf - 30.0) / 70.0), 0.78, 1.0)

        # Total combined volumetric efficiency
        volumetric_eff = fillage * eta_valve * eta_rf_stroke
        q_net = q_theor * volumetric_eff

        # 6. Energy & Power [PHYSICS_DERIVED]
        cycle_work_lbf_in = load_range * state.stroke_length_in * self.card_fullness
        power_hp = (cycle_work_lbf_in * state.spm) / (12.0 * 33000.0)
        power_kw = power_hp * 0.7457
        daily_kwh = power_kw * 24.0
        specific_energy = daily_kwh / max(q_net, 0.1)

        # Severity Classification
        if composite_risk >= 80.0:
            sev = "Critical"
        elif composite_risk >= 55.0:
            sev = "High"
        elif composite_risk >= 25.0:
            sev = "Medium"
        else:
            sev = "Low"

        # Diagnostics Summary
        summary = (
            f"SPM={state.spm:.1f} | Visc={visc:.0f} cP | PPRL={pprl:.0f} lbf | "
            f"Prod={q_net:.1f} bpd (eta={volumetric_eff*100:.1f}%) | "
            f"Power={power_kw:.1f} kW | Risk={composite_risk:.1f} ({sev})"
        )

        return WellPredictionResult(
            temperature_f=state.temperature_f,
            viscosity_cp=round(visc, 1),
            mobility_index=round(mobility_idx, 3),
            spm=state.spm,
            omega_rad_s=round(omega, 3),
            buoyant_rod_weight_lbf=round(w_buoyant, 1),
            fluid_load_lbf=round(f_fluid, 1),
            inertia_amplitude_lbf=round(f_inertia, 1),
            viscous_drag_lbf=round(f_viscous_drag, 1),
            pprl_lbf=round(pprl, 1),
            mprl_lbf=round(mprl, 1),
            load_range_lbf=round(load_range, 1),
            theoretical_displacement_bpd=round(q_theor, 1),
            pump_fillage_fraction=round(fillage, 3),
            valve_viscosity_efficiency=round(eta_valve, 3),
            rod_float_stroke_efficiency=round(eta_rf_stroke, 3),
            volumetric_efficiency=round(volumetric_eff, 3),
            net_production_bpd=round(q_net, 1),
            polished_rod_power_hp=round(power_hp, 2),
            polished_rod_power_kw=round(power_kw, 2),
            specific_energy_kwh_per_bbl=round(specific_energy, 2),
            cycle_work_lbf_in=round(cycle_work_lbf_in, 1),
            rod_floating_risk=round(risk_rf, 1),
            fluid_pound_risk=round(risk_fp, 1),
            gas_interference_risk=round(risk_gi, 1),
            composite_risk_score=round(composite_risk, 1),
            severity=sev,
            diagnostics_summary=summary,
        )

    def simulate_scenario(
        self,
        baseline_state: WellOperatingState,
        modifications: Dict[str, Any],
        scenario_name: str = "Scenario",
        scenario_description: str = "",
        thermal_coupling: bool = True,
    ) -> ScenarioComparison:
        """
        Executes a what-if scenario comparison against a baseline well state.
        """
        # Baseline evaluation
        baseline_res = self.evaluate_well(
            baseline_state, compute_thermal_viscosity=False
        )

        # Build modified state
        mod_dict = asdict(baseline_state)
        for k, v in modifications.items():
            if k in mod_dict:
                mod_dict[k] = v
        mod_state = WellOperatingState(**mod_dict)

        # Scenario evaluation
        scenario_res = self.evaluate_well(
            mod_state,
            compute_thermal_viscosity=thermal_coupling
            and (
                "temperature_f" in modifications
                and "viscosity_cp" not in modifications
            ),
            base_temp_f=baseline_state.temperature_f,
            base_visc_cp=baseline_state.viscosity_cp,
        )

        # Compute Deltas
        pct_visc = (
            (scenario_res.viscosity_cp - baseline_res.viscosity_cp)
            / baseline_res.viscosity_cp
        ) * 100.0
        pct_prod = (
            (scenario_res.net_production_bpd - baseline_res.net_production_bpd)
            / max(baseline_res.net_production_bpd, 0.1)
        ) * 100.0

        delta = ScenarioDelta(
            delta_spm=round(scenario_res.spm - baseline_res.spm, 2),
            delta_temperature_f=round(
                scenario_res.temperature_f - baseline_res.temperature_f, 1
            ),
            delta_viscosity_cp=round(
                scenario_res.viscosity_cp - baseline_res.viscosity_cp, 1
            ),
            pct_viscosity_change=round(pct_visc, 1),
            delta_mobility_index=round(
                scenario_res.mobility_index - baseline_res.mobility_index, 3
            ),
            delta_pprl_lbf=round(scenario_res.pprl_lbf - baseline_res.pprl_lbf, 1),
            delta_mprl_lbf=round(scenario_res.mprl_lbf - baseline_res.mprl_lbf, 1),
            delta_load_range_lbf=round(
                scenario_res.load_range_lbf - baseline_res.load_range_lbf, 1
            ),
            delta_production_bpd=round(
                scenario_res.net_production_bpd - baseline_res.net_production_bpd, 1
            ),
            pct_production_change=round(pct_prod, 1),
            delta_power_kw=round(
                scenario_res.polished_rod_power_kw
                - baseline_res.polished_rod_power_kw,
                2,
            ),
            delta_specific_energy_kwh_bbl=round(
                scenario_res.specific_energy_kwh_per_bbl
                - baseline_res.specific_energy_kwh_per_bbl,
                2,
            ),
            delta_rod_floating_risk=round(
                scenario_res.rod_floating_risk - baseline_res.rod_floating_risk, 1
            ),
            delta_fluid_pound_risk=round(
                scenario_res.fluid_pound_risk - baseline_res.fluid_pound_risk, 1
            ),
            delta_gas_interference_risk=round(
                scenario_res.gas_interference_risk
                - baseline_res.gas_interference_risk,
                1,
            ),
            delta_composite_risk=round(
                scenario_res.composite_risk_score
                - baseline_res.composite_risk_score,
                1,
            ),
        )

        # Generate Engineering Insights
        insights = []
        if abs(delta.delta_viscosity_cp) > 20:
            insights.append(
                f"Viscosity shifted by {delta.delta_viscosity_cp:+.0f} cP ({delta.pct_viscosity_change:+.1f}%), changing mobility index from {baseline_res.mobility_index:.2f} to {scenario_res.mobility_index:.2f}."
            )
        if abs(delta.delta_rod_floating_risk) >= 10:
            direction = "reduced" if delta.delta_rod_floating_risk < 0 else "increased"
            insights.append(
                f"Rod floating risk {direction} by {abs(delta.delta_rod_floating_risk):.1f} pts (now {scenario_res.rod_floating_risk:.1f}/100)."
            )
        if abs(delta.delta_fluid_pound_risk) >= 10:
            direction = "reduced" if delta.delta_fluid_pound_risk < 0 else "increased"
            insights.append(
                f"Fluid pound risk {direction} by {abs(delta.delta_fluid_pound_risk):.1f} pts (now {scenario_res.fluid_pound_risk:.1f}/100)."
            )
        if abs(delta.delta_production_bpd) > 0.5:
            direction = "increased" if delta.delta_production_bpd > 0 else "decreased"
            insights.append(
                f"Net production {direction} by {abs(delta.delta_production_bpd):.1f} bpd ({delta.pct_production_change:+.1f}%), moving volumetric efficiency from {baseline_res.volumetric_efficiency*100:.1f}% to {scenario_res.volumetric_efficiency*100:.1f}%."
            )
        if abs(delta.delta_power_kw) > 0.3:
            insights.append(
                f"Polished rod power changed by {delta.delta_power_kw:+.2f} kW (Specific energy: {scenario_res.specific_energy_kwh_per_bbl:.2f} kWh/bbl)."
            )

        return ScenarioComparison(
            scenario_name=scenario_name,
            scenario_description=scenario_description,
            baseline=baseline_res,
            scenario=scenario_res,
            delta=delta,
            engineering_insights=insights,
        )
