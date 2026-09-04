"""
risk_engine.py
--------------
TEJAS Phase 6: Physics-Informed Risk & Engineering Assessment Engine.

Converts:
  - CNN condition probabilities (from Phase 5B)
  - Operating context (SPM, viscosity, temperature, fluid level, pump depth, production rate)
  - Card mechanical features (impact slopes, MPRL, load range, card fullness)
Into:
  - Condition-specific risk scores (0-100)
  - Overall Severity level (Low, Medium, High, Critical)
  - Physics-grounded engineering explanations
  - Actionable operational recommendations

NOTE:
This module contains explainable, rule-based engineering logic grounded in
petroleum engineering physics (API RP 11L / Sucker-Rod Pumping literature).
It is distinct from the ML classifier (CNN) and is not field-calibrated.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


@dataclass
class AssessmentResult:
    card_id: str
    well_id: str
    predicted_condition: str
    confidence: float
    probabilities: Dict[str, float]
    risk_score: float
    severity: str
    primary_fault_risk: float
    engineering_explanation: str
    recommended_action: str
    contributing_factors: List[str]


class TEJASRiskEngine:
    """
    Physics-informed risk and operational decision engine for Sucker-Rod Pumping.
    """

    SEVERITY_THRESHOLDS = {
        "Critical": 80.0,
        "High": 55.0,
        "Medium": 25.0,
        "Low": 0.0,
    }

    def __init__(self):
        pass

    @staticmethod
    def _clip(val: float, low: float = 0.0, high: float = 1.0) -> float:
        return float(np.clip(val, low, high))

    def evaluate_rod_floating_risk(
        self, p_rf: float, ctx: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """
        Evaluate Rod Floating severity.
        Physics mechanism: Heavy viscous drag + high pumping speed prevents rod
        from falling freely on downstroke, causing slack and impact spike near bottom.
        """
        viscosity = ctx.get("viscosity", 400.0)
        spm = ctx.get("SPM", 6.0)
        temp = ctx.get("temperature", 130.0)
        max_slope_down = ctx.get("max_abs_slope_down_z", 0.0)

        factors = []

        # 1. Viscosity contribution (cold/heavy crude increases drag exponentially)
        f_visc = self._clip((viscosity - 350.0) / 650.0)
        if viscosity > 600.0:
            factors.append(f"Elevated viscosity ({viscosity:.1f} cP) increases rod downstroke fluid drag")

        # 2. Pumping speed (higher SPM reduces allowable downstroke fall time)
        f_spm = self._clip((spm - 5.5) / 5.0)
        if spm > 7.5:
            factors.append(f"High pumping speed ({spm:.1f} SPM) exceeds rod free-fall velocity limit")

        # 3. Thermal deficit (cooler temperature elevates viscosity)
        f_temp = self._clip((145.0 - temp) / 45.0)
        if temp < 120.0:
            factors.append(f"Low wellbore temperature ({temp:.1f} °F) exacerbates fluid thickening")

        # 4. Impact spike indicator on downstroke
        f_spike = self._clip((max_slope_down - 15.0) / 90.0)
        if max_slope_down > 40.0:
            factors.append(f"Severe mechanical impact spike detected on downstroke (slope_z={max_slope_down:.1f})")

        # Operating severity index [0..1]
        operating_index = 0.35 * f_visc + 0.30 * f_spm + 0.15 * f_temp + 0.20 * f_spike

        # Risk score (0 to 100)
        risk = p_rf * (35.0 + 65.0 * operating_index)
        return float(np.clip(risk, 0.0, 100.0)), factors

    def evaluate_fluid_pound_risk(
        self, p_fp: float, ctx: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """
        Evaluate Fluid Pound severity.
        Physics mechanism: Pump displacement exceeds reservoir inflow; incomplete
        barrel fillage causes plunger to free-fall through gas/vapor and slam onto fluid.
        """
        fluid_level = ctx.get("fluid_level", 3000.0)
        pump_depth = ctx.get("pump_depth", 4500.0)
        max_slope_down = ctx.get("max_abs_slope_down_z", 0.0)
        mprl_raw = ctx.get("MPRL_raw_lbf", 7000.0)

        factors = []

        # 1. Annulus fluid level deficit (low fluid level above pump)
        f_fl = self._clip((3500.0 - fluid_level) / 2500.0)
        if fluid_level < 2500.0:
            factors.append(f"Depleted fluid column ({fluid_level:.0f} ft above pump) indicates pump-off condition")

        # 2. Submergence ratio
        submergence_ratio = fluid_level / max(pump_depth, 1.0)
        f_sub = self._clip((0.65 - submergence_ratio) / 0.5)
        if submergence_ratio < 0.50:
            factors.append(f"Low pump submergence ratio ({submergence_ratio*100:.1f}%)")

        # 3. Downstroke impact spike intensity
        f_spike = self._clip((max_slope_down - 15.0) / 90.0)
        if max_slope_down > 40.0:
            factors.append(f"Sharp fluid-surface impact shock detected (slope_z={max_slope_down:.1f})")

        # 4. Low minimum load during downstroke free-fall
        f_mprl = self._clip((7500.0 - mprl_raw) / 3500.0)
        if mprl_raw < 5500.0:
            factors.append(f"Depressed minimum load ({mprl_raw:.0f} lbf) flags plunger free-fall inside barrel")

        operating_index = 0.35 * f_fl + 0.20 * f_sub + 0.30 * f_spike + 0.15 * f_mprl
        risk = p_fp * (40.0 + 60.0 * operating_index)
        return float(np.clip(risk, 0.0, 100.0)), factors

    def evaluate_gas_interference_risk(
        self, p_gi: float, ctx: Dict[str, Any]
    ) -> tuple[float, List[str]]:
        """
        Evaluate Gas Interference severity.
        Physics mechanism: Free gas enters pump barrel, delaying traveling valve opening,
        compressing cyclically and reducing net pumped stroke volume and lifting efficiency.
        """
        card_area_norm = ctx.get("card_area_shape_norm", 0.75)
        prod_rate = ctx.get("production_rate", 50.0)
        load_range = ctx.get("load_range_raw_lbf", 6000.0)

        factors = []

        # 1. Dimensionless card area loss (stroke efficiency loss)
        f_area = self._clip((0.78 - card_area_norm) / 0.40)
        if card_area_norm < 0.65:
            factors.append(f"Volumetric card area collapsed to {card_area_norm:.2f} due to gas compression")

        # 2. Production deficit
        f_prod = self._clip((60.0 - prod_rate) / 45.0)
        if prod_rate < 30.0:
            factors.append(f"Suppressed production rate ({prod_rate:.1f} bpd) from gas lock/slippage")

        # 3. Load range compression
        f_range = self._clip((5800.0 - load_range) / 2500.0)
        if load_range < 5000.0:
            factors.append(f"Narrow load swing ({load_range:.0f} lbf) reflects cushioned valve response")

        operating_index = 0.45 * f_area + 0.30 * f_prod + 0.25 * f_range
        risk = p_gi * (30.0 + 70.0 * operating_index)
        return float(np.clip(risk, 0.0, 100.0)), factors

    def evaluate_normal_risk(self, p_norm: float, ctx: Dict[str, Any]) -> tuple[float, List[str]]:
        spm = ctx.get("SPM", 6.0)
        base_stress = 8.0 * self._clip(spm / 10.0, 0.5, 1.5)
        risk = p_norm * base_stress
        factors = ["Operating within nominal mechanical baseline"] if p_norm > 0.7 else []
        return float(np.clip(risk, 0.0, 20.0)), factors

    def assess_card(
        self,
        card_id: str,
        well_id: str,
        probabilities: Dict[str, float],
        context: Dict[str, Any],
    ) -> AssessmentResult:
        """
        Perform complete engineering and risk assessment for a single card.
        """
        p_norm = probabilities.get("Normal", 0.0)
        p_rf = probabilities.get("Rod Floating", 0.0)
        p_fp = probabilities.get("Fluid Pound", 0.0)
        p_gi = probabilities.get("Gas Interference", 0.0)

        # Compute individual risk scores
        r_rf, f_rf = self.evaluate_rod_floating_risk(p_rf, context)
        r_fp, f_fp = self.evaluate_fluid_pound_risk(p_fp, context)
        r_gi, f_gi = self.evaluate_gas_interference_risk(p_gi, context)
        r_norm, f_norm = self.evaluate_normal_risk(p_norm, context)

        # Primary predicted condition by probability
        pred_cond = max(probabilities, key=probabilities.get)
        confidence = probabilities[pred_cond]

        # Overall risk score is dominated by active fault hazards
        fault_risks = {
            "Rod Floating": (r_rf, f_rf),
            "Fluid Pound": (r_fp, f_fp),
            "Gas Interference": (r_gi, f_gi),
            "Normal": (r_norm, f_norm),
        }

        active_risk, active_factors = fault_risks[pred_cond]
        overall_risk = max(r_rf, r_fp, r_gi, r_norm)

        # Map to severity tier
        if overall_risk >= self.SEVERITY_THRESHOLDS["Critical"]:
            severity = "Critical"
        elif overall_risk >= self.SEVERITY_THRESHOLDS["High"]:
            severity = "High"
        elif overall_risk >= self.SEVERITY_THRESHOLDS["Medium"]:
            severity = "Medium"
        else:
            severity = "Low"

        # Generate engineering explanations and recommendations
        explanation, recommendation = self._generate_engineering_guidance(
            pred_cond, severity, overall_risk, active_factors, context
        )

        return AssessmentResult(
            card_id=card_id,
            well_id=well_id,
            predicted_condition=pred_cond,
            confidence=round(confidence, 4),
            probabilities={k: round(v, 4) for k, v in probabilities.items()},
            risk_score=round(overall_risk, 1),
            severity=severity,
            primary_fault_risk=round(active_risk, 1),
            engineering_explanation=explanation,
            recommended_action=recommendation,
            contributing_factors=active_factors,
        )

    def _generate_engineering_guidance(
        self,
        condition: str,
        severity: str,
        risk: float,
        factors: List[str],
        ctx: Dict[str, Any],
    ) -> tuple[str, str]:
        spm = ctx.get("SPM", 6.0)
        visc = ctx.get("viscosity", 400.0)
        fl = ctx.get("fluid_level", 3000.0)

        factor_summary = (
            "; ".join(factors) if factors else "Operating parameters within nominal thresholds"
        )

        if condition == "Normal":
            explanation = (
                f"Sucker-rod card exhibits uniform load profile and normal full-stroke work envelope. {factor_summary}."
            )
            recommendation = (
                "Maintain baseline pumping parameters. Continue regular SCADA monitoring and scheduled lubrication."
            )

        elif condition == "Rod Floating":
            explanation = (
                f"Severe downstroke rod deceleration diagnosed. Downstroke velocity lag caused by heavy crude "
                f"drag ({visc:.0f} cP) and pumping speed ({spm:.1f} SPM) creates compressive rod stress and "
                f"sharp impact on plunger re-engagement. Root factors: {factor_summary}."
            )
            if severity in ["Critical", "High"]:
                recommendation = (
                    f"IMMEDIATE ACTION: Reduce VFD stroke speed from {spm:.1f} SPM to {(spm*0.75):.1f} SPM to restore "
                    f"gravity downstroke tracking. Schedule CSS (Cyclic Steam Stimulation) / diluent injection to "
                    f"reduce crude viscosity. Inspect rod string for compressive buckling."
                )
            else:
                recommendation = (
                    f"ADVISORY: Monitor downstroke load trough. Trim VFD speed by 0.5-1.0 SPM if wellbore temperature "
                    f"drops further. Plan thermal stimulation cycle."
                )

        elif condition == "Fluid Pound":
            explanation = (
                f"Incomplete pump barrel fillage diagnosed due to reservoir inflow deficit ({fl:.0f} ft fluid level). "
                f"Plunger free-falls through vapor pocket before abruptly impacting fluid surface on downstroke, "
                f"transmitting shock waves to polished rod and gearbox. Root factors: {factor_summary}."
            )
            if severity in ["Critical", "High"]:
                recommendation = (
                    f"IMMEDIATE ACTION: Throttle pump speed from {spm:.1f} SPM to {(spm*0.70):.1f} SPM or activate "
                    f"automatic Pump-Off Controller (POC) with intermittent timer (duty cycling) to permit annular "
                    f"fluid recovery. Inspect valves and rod guides."
                )
            else:
                recommendation = (
                    f"ADVISORY: Adjust POC thresholds to trigger earlier pump shutdown on underfill. Verify bottomhole "
                    f"pressure and well inflow performance."
                )

        elif condition == "Gas Interference":
            explanation = (
                f"Compressible free gas inside the pump barrel is delaying traveling valve opening on downstroke. "
                f"Volumetric lifting capacity is degraded due to cyclic gas compression. Root factors: {factor_summary}."
            )
            if severity in ["Critical", "High"]:
                recommendation = (
                    "OPERATIONAL ACTION: Increase casing-tubing annulus venting to relieve gas backpressure; "
                    "lower pump intake setting below perforations to improve natural gas separation, or install "
                    "a downhole gas separator (poor boy / helical separator)."
                )
            else:
                recommendation = (
                    "ADVISORY: Monitor casinghead gas backpressure. Optimize stroke length vs SPM to maximize "
                    "compression ratio in barrel."
                )
        else:
            explanation = f"Condition: {condition}. {factor_summary}."
            recommendation = "Inspect well operating parameters."

        return explanation, recommendation
