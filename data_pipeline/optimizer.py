"""
optimizer.py
------------
TEJAS Phase 8: Multi-Objective Optimization & Digital Twin Decision Engine.

Searches feasible operating parameter spaces for heavy-oil Sucker-Rod Pumping (SRP) wells
and selects Pareto-optimal operating recommendations.

Objectives:
  1. Maximize Net Liquid Production (bpd)
  2. Minimize Mechanical & Diagnostic Risk (0-100 score, avoid High/Critical tiers)
  3. Minimize Specific Energy Consumption (kWh/bbl) & Polished Rod Power (kW)

Explicitly handles operational constraints:
  - Maximum allowable risk (R < 50.0 for recommended scenarios)
  - Mechanical rod load limits (PPRL, MPRL >= 0)
  - Kinematic SPM bounds (3.0 <= SPM <= 10.0)

Provides:
  - Balanced Operating Optimum (Recommended)
  - Maximum Safe Production Scenario (Alternative)
  - Maximum Energy Efficiency Scenario (Alternative)
  - Actionable Engineering Rationale explaining selection

IMPORTANT:
This is a Digital Twin Decision Support system, NOT an autonomous control loop.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from data_pipeline.well_prediction import (
    PhysicsWellPredictor,
    WellOperatingState,
    WellPredictionResult,
    ScenarioDelta,
    ScenarioComparison,
)


@dataclass
class OptimizationCandidate:
    """Evaluated operating scenario candidate."""
    scenario_id: str
    scenario_type: str  # e.g., 'SPM_Sweep', 'Thermal_CSS', 'POC_Recovery', 'Gas_Separation', 'Co_Optimization'
    modifications: Dict[str, Any]
    state: WellPredictionResult
    delta: ScenarioDelta
    is_feasible: bool
    constraint_violations: List[str]
    is_pareto_optimal: bool = False
    utility_score: float = 0.0


@dataclass
class OptimizationRecommendation:
    """Complete Decision Support Recommendation for a Well."""
    well_id: str
    current_state: WellPredictionResult
    recommended_scenario: OptimizationCandidate
    max_production_scenario: OptimizationCandidate
    max_efficiency_scenario: OptimizationCandidate
    all_pareto_scenarios: List[OptimizationCandidate]
    selection_rationale: str
    operational_warnings: List[str]


class TEJASWellOptimizer:
    """
    Multi-objective Digital Twin optimizer for Sucker-Rod Pumping wells.
    """

    def __init__(
        self,
        predictor: Optional[PhysicsWellPredictor] = None,
        max_acceptable_risk: float = 48.0,
        max_allowable_pprl_lbf: float = 24000.0,
        min_allowable_mprl_lbf: float = 500.0,
        w_production: float = 0.45,
        w_risk: float = 0.35,
        w_energy: float = 0.20,
    ):
        self.predictor = predictor or PhysicsWellPredictor()
        self.max_risk = max_acceptable_risk
        self.max_pprl = max_allowable_pprl_lbf
        self.min_mprl = min_allowable_mprl_lbf

        # Objective weights (normalized sum = 1.0)
        total_w = w_production + w_risk + w_energy
        self.w_prod = w_production / total_w
        self.w_risk = w_risk / total_w
        self.w_energy = w_energy / total_w

    def generate_candidate_grid(
        self, baseline_state: WellOperatingState
    ) -> List[Tuple[str, Dict[str, Any], str]]:
        """
        Generates realistic operational what-if candidates tailored to the well state.
        Returns list of (scenario_id, modifications_dict, scenario_type).
        """
        candidates = []
        base_spm = baseline_state.spm
        base_temp = baseline_state.temperature_f
        base_visc = baseline_state.viscosity_cp
        base_fl = baseline_state.fluid_level_ft
        base_gas = baseline_state.gas_fraction

        # 1. Pure SPM Adjustments (VFD Speed Trimming / Scaling)
        spm_range = np.arange(3.5, 9.5, 0.5)
        for spm in spm_range:
            spm = round(float(spm), 1)
            if abs(spm - base_spm) > 0.1:
                candidates.append(
                    (f"SPM_{spm:.1f}", {"spm": spm}, "VFD_Speed_Adjustment")
                )

        # 2. Thermal Heating Candidates (CSS / Steam Soaking)
        if base_visc > 350.0 or base_temp < 150.0:
            for delta_t in [15.0, 30.0, 45.0]:
                new_temp = min(base_temp + delta_t, 185.0)
                candidates.append(
                    (
                        f"Thermal_+{delta_t:.0f}F",
                        {"temperature_f": new_temp},
                        "Thermal_CSS_Stimulation",
                    )
                )

        # 3. Fluid Level Recovery (POC Duty-Cycle Throttling)
        if base_fl < 2600.0:
            for rec_fl in [2400.0, 3000.0, 3500.0]:
                for throttle_spm in [4.0, 5.0, 6.0]:
                    candidates.append(
                        (
                            f"POC_SPM_{throttle_spm:.1f}_FL_{rec_fl:.0f}",
                            {"spm": throttle_spm, "fluid_level_ft": rec_fl},
                            "POC_Inflow_Recovery",
                        )
                    )

        # 4. Downhole Gas Mitigation
        if base_gas > 0.10:
            candidates.append(
                (
                    "Gas_Separator_Retrofit",
                    {"gas_fraction": 0.04},
                    "Downhole_Gas_Separation",
                )
            )

        # 5. Combined Co-Optimization (Thermal + VFD Tuning)
        if base_visc > 400.0:
            for delta_t in [20.0, 35.0]:
                new_temp = min(base_temp + delta_t, 180.0)
                for opt_spm in [5.0, 5.5, 6.0, 6.5, 7.0]:
                    candidates.append(
                        (
                            f"CoOpt_T+{delta_t:.0f}F_SPM_{opt_spm:.1f}",
                            {"temperature_f": new_temp, "spm": opt_spm},
                            "Thermal_VFD_CoOptimization",
                        )
                    )

        # 6. Combined Gas + Speed / Inflow Tuning
        if base_gas > 0.10:
            for opt_spm in [5.0, 6.0, 7.0]:
                candidates.append(
                    (
                        f"GasSep_SPM_{opt_spm:.1f}",
                        {"gas_fraction": 0.04, "spm": opt_spm},
                        "Gas_Separation_Plus_Speed",
                    )
                )
            if base_fl < 2600.0:
                for opt_spm in [4.5, 5.5]:
                    candidates.append(
                        (
                            f"GasSep_POC_SPM_{opt_spm:.1f}_FL_3000",
                            {"gas_fraction": 0.04, "spm": opt_spm, "fluid_level_ft": 3000.0},
                            "Gas_Separation_Plus_POC_Recovery",
                        )
                    )

        return candidates

    def check_feasibility(
        self, pred: WellPredictionResult
    ) -> Tuple[bool, List[str]]:
        """
        Validates engineering and mechanical constraints.
        """
        violations = []

        if pred.composite_risk_score > self.max_risk:
            violations.append(
                f"Risk score {pred.composite_risk_score:.1f} exceeds threshold ({self.max_risk:.1f}) [Severity: {pred.severity}]"
            )

        if pred.pprl_lbf > self.max_pprl:
            violations.append(
                f"Peak load {pred.pprl_lbf:.0f} lbf exceeds structural limit ({self.max_pprl:.0f} lbf)"
            )

        if pred.mprl_lbf < self.min_mprl:
            violations.append(
                f"Minimum load {pred.mprl_lbf:.0f} lbf below compression threshold ({self.min_mprl:.0f} lbf) [Floating Rods]"
            )

        if pred.spm < 3.0 or pred.spm > 11.0:
            violations.append(f"SPM {pred.spm:.1f} outside mechanical VFD bounds [3.0 - 11.0]")

        return (len(violations) == 0, violations)

    @staticmethod
    def identify_pareto_front(
        candidates: List[OptimizationCandidate],
    ) -> List[OptimizationCandidate]:
        """
        Identifies non-dominated solutions on the 3D objective space:
        Maximize Production, Minimize Composite Risk, Minimize Specific Energy.
        """
        feasible = [c for c in candidates if c.is_feasible]
        if not feasible:
            return []

        pareto_list = []
        for i, c1 in enumerate(feasible):
            dominated = False
            # Points: (Prod, -Risk, -SEC)
            p1 = (
                c1.state.net_production_bpd,
                -c1.state.composite_risk_score,
                -c1.state.specific_energy_kwh_per_bbl,
            )

            for j, c2 in enumerate(feasible):
                if i != j:
                    p2 = (
                        c2.state.net_production_bpd,
                        -c2.state.composite_risk_score,
                        -c2.state.specific_energy_kwh_per_bbl,
                    )
                    # c2 dominates c1 if all objectives are >= and at least one is >
                    if (
                        p2[0] >= p1[0]
                        and p2[1] >= p1[1]
                        and p2[2] >= p1[2]
                        and (p2[0] > p1[0] or p2[1] > p1[1] or p2[2] > p1[2])
                    ):
                        dominated = True
                        break

            if not dominated:
                c1.is_pareto_optimal = True
                pareto_list.append(c1)

        return pareto_list

    def compute_utility_scores(
        self,
        candidates: List[OptimizationCandidate],
        baseline: WellPredictionResult,
    ):
        """
        Computes transparent multi-attribute utility score for ranking candidates:
        U = w_prod * Norm(Prod) + w_risk * (1 - Norm(Risk)) + w_energy * (1 - Norm(SEC))
        """
        feasible = [c for c in candidates if c.is_feasible]
        if not feasible:
            return

        prods = [c.state.net_production_bpd for c in feasible]
        risks = [c.state.composite_risk_score for c in feasible]
        secs = [c.state.specific_energy_kwh_per_bbl for c in feasible]

        min_p, max_p = min(prods), max(prods)
        min_r, max_r = min(risks), max(risks)
        min_e, max_e = min(secs), max(secs)

        range_p = max(max_p - min_p, 1e-3)
        range_r = max(max_r - min_r, 1e-3)
        range_e = max(max_e - min_e, 1e-3)

        for c in feasible:
            norm_prod = (c.state.net_production_bpd - min_p) / range_p
            norm_risk = (c.state.composite_risk_score - min_r) / range_r
            norm_sec = (c.state.specific_energy_kwh_per_bbl - min_e) / range_e

            # Utility in [0, 1]
            u = (
                self.w_prod * norm_prod
                + self.w_risk * (1.0 - norm_risk)
                + self.w_energy * (1.0 - norm_sec)
            )
            c.utility_score = round(float(u), 4)

    def optimize_well(
        self, baseline_state: WellOperatingState
    ) -> OptimizationRecommendation:
        """
        Executes end-to-end multi-objective optimization for a well.
        """
        # 1. Evaluate baseline
        baseline_res = self.predictor.evaluate_well(
            baseline_state, compute_thermal_viscosity=False
        )

        # 2. Generate candidate modifications
        grid = self.generate_candidate_grid(baseline_state)

        # 3. Simulate and evaluate all candidates
        evaluated_candidates: List[OptimizationCandidate] = []

        for scen_id, mods, scen_type in grid:
            sim = self.predictor.simulate_scenario(
                baseline_state=baseline_state,
                modifications=mods,
                scenario_name=scen_id,
                thermal_coupling=True,
            )

            is_feas, violations = self.check_feasibility(sim.scenario)

            candidate = OptimizationCandidate(
                scenario_id=scen_id,
                scenario_type=scen_type,
                modifications=mods,
                state=sim.scenario,
                delta=sim.delta,
                is_feasible=is_feas,
                constraint_violations=violations,
            )
            evaluated_candidates.append(candidate)

        # 4. Identify Pareto Frontier
        pareto_front = self.identify_pareto_front(evaluated_candidates)

        # 5. Compute Utility Scores
        self.compute_utility_scores(evaluated_candidates, baseline_res)

        # 6. Select Representative Operating Scenarios
        feasible_candidates = [c for c in evaluated_candidates if c.is_feasible]

        if not feasible_candidates:
            # Fallback: Pick candidate with lowest risk
            best_cand = min(
                evaluated_candidates, key=lambda c: c.state.composite_risk_score
            )
            max_prod_cand = best_cand
            max_eff_cand = best_cand
            rationale = "No candidates met all strict constraints. Selected safest operating point to de-escalate severity."
            warnings = ["Severe mechanical/fluid constraints present; de-escalation required."]
        else:
            # A. Recommended (Highest Composite Utility)
            best_cand = max(feasible_candidates, key=lambda c: c.utility_score)

            # B. Max Safe Production (Highest Production among Feasible)
            max_prod_cand = max(
                feasible_candidates, key=lambda c: c.state.net_production_bpd
            )

            # C. Max Energy Efficiency (Lowest Specific Energy among Feasible)
            max_eff_cand = min(
                feasible_candidates,
                key=lambda c: c.state.specific_energy_kwh_per_bbl,
            )

            # Rationale generation
            rationale = self._synthesize_rationale(
                baseline_res, best_cand, max_prod_cand, max_eff_cand
            )
            warnings = []
            if baseline_res.composite_risk_score >= 55.0:
                warnings.append(
                    f"Baseline well is in {baseline_res.severity} severity tier ({baseline_res.composite_risk_score:.1f}/100); immediate operating point shift advised."
                )

        return OptimizationRecommendation(
            well_id=baseline_state.well_id,
            current_state=baseline_res,
            recommended_scenario=best_cand,
            max_production_scenario=max_prod_cand,
            max_efficiency_scenario=max_eff_cand,
            all_pareto_scenarios=pareto_front,
            selection_rationale=rationale,
            operational_warnings=warnings,
        )

    def _synthesize_rationale(
        self,
        base: WellPredictionResult,
        rec: OptimizationCandidate,
        max_prod: OptimizationCandidate,
        max_eff: OptimizationCandidate,
    ) -> str:
        s = rec.state
        d = rec.delta
        reasons = []

        if "temperature_f" in rec.modifications:
            reasons.append(
                f"Thermal CSS treatment ({s.temperature_f:.0f}°F) cuts viscosity by {d.delta_viscosity_cp:+.0f} cP ({d.pct_viscosity_change:+.1f}%), unlocking fluid mobility"
            )
        if "spm" in rec.modifications:
            reasons.append(
                f"Adjusting pumping speed from {base.spm:.1f} to {s.spm:.1f} SPM rebalances rod kinematics and avoids downstroke impact"
            )
        if "fluid_level_ft" in rec.modifications:
            rec_fl = rec.modifications["fluid_level_ft"]
            reasons.append(
                f"Allowing annular fluid level recovery to {rec_fl:.0f} ft eliminates pump-off fluid pound"
            )
        if "gas_fraction" in rec.modifications:
            reasons.append(
                "Gas separation retrofit recovers volumetric barrel fullness"
            )

        outcome = (
            f"Achieves {d.delta_production_bpd:+.1f} bpd net liquid ({d.pct_production_change:+.1f}%), "
            f"reduces polished rod power by {d.delta_power_kw:+.2f} kW, and drives composite risk down from "
            f"{base.composite_risk_score:.1f} ({base.severity}) to {s.composite_risk_score:.1f} ({s.severity})."
        )

        return f"{'; '.join(reasons)}. {outcome}"
