"""
run_whatif_scenarios.py
-----------------------
Executes representative what-if operational scenarios for heavy-oil SRP wells
using the TEJAS Phase 7 Physics-Informed Prediction Layer.
"""

from pathlib import Path
import json
import pandas as pd
from data_pipeline.well_prediction import (
    PhysicsWellPredictor,
    WellOperatingState,
    ScenarioComparison,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase7"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    predictor = PhysicsWellPredictor(temp_viscosity_beta=0.028)

    # -------------------------------------------------------------
    # Define Baseline Well Profiles
    # -------------------------------------------------------------
    # Profile 1: Heavy Crude Cold Well (High Rod-Floating Hazard)
    well_heavy_cold = WellOperatingState(
        well_id="WELL-019",
        spm=8.5,
        temperature_f=125.0,
        viscosity_cp=850.0,
        fluid_level_ft=3400.0,
        stroke_length_in=120.0,
        pump_depth_ft=5000.0,
        plunger_diameter_in=2.0,
        rod_weight_per_ft=2.20,
    )

    # Profile 2: Overpumped / Low Inflow Well (Fluid Pound Hazard)
    well_inflow_deficit = WellOperatingState(
        well_id="WELL-005",
        spm=7.5,
        temperature_f=138.0,
        viscosity_cp=420.0,
        fluid_level_ft=1400.0,
        stroke_length_in=120.0,
        pump_depth_ft=4600.0,
        plunger_diameter_in=2.0,
        rod_weight_per_ft=2.16,
    )

    # Profile 3: Gassy Heavy Oil Well (Gas Interference Hazard)
    well_gassy = WellOperatingState(
        well_id="WELL-007",
        spm=6.0,
        temperature_f=135.0,
        viscosity_cp=400.0,
        fluid_level_ft=2000.0,
        gas_fraction=0.30,
        stroke_length_in=140.0,
        pump_depth_ft=4800.0,
        plunger_diameter_in=2.0,
        rod_weight_per_ft=2.16,
    )

    # Profile 4: Balanced Nominal Well (Baseline for Speed Scaling)
    well_nominal = WellOperatingState(
        well_id="WELL-003",
        spm=5.5,
        temperature_f=145.0,
        viscosity_cp=380.0,
        fluid_level_ft=3600.0,
        stroke_length_in=120.0,
        pump_depth_ft=4500.0,
        plunger_diameter_in=2.0,
        rod_weight_per_ft=2.16,
    )

    # -------------------------------------------------------------
    # Define What-If Scenarios
    # -------------------------------------------------------------
    scenarios: list[ScenarioComparison] = []

    # Scenario 1: Thermal CSS Heating (+35 °F) on Heavy Cold Well
    s1 = predictor.simulate_scenario(
        baseline_state=well_heavy_cold,
        modifications={"temperature_f": 160.0},
        scenario_name="S1_Thermal_CSS_Stimulation",
        scenario_description="Cyclic Steam Stimulation (CSS) increases wellbore temp from 125°F to 160°F (+35°F), exponentially thinning heavy crude.",
        thermal_coupling=True,
    )
    scenarios.append(s1)

    # Scenario 2: VFD Speed Reduction (8.5 -> 5.0 SPM) on Heavy Cold Well
    s2 = predictor.simulate_scenario(
        baseline_state=well_heavy_cold,
        modifications={"spm": 5.0},
        scenario_name="S2_VFD_Speed_Throttling",
        scenario_description="Reduce VFD speed from 8.5 SPM to 5.0 SPM to eliminate downstroke velocity mismatch and mitigate rod floating.",
        thermal_coupling=False,
    )
    scenarios.append(s2)

    # Scenario 3: Solvent / Diluent Injection on Heavy Cold Well
    s3 = predictor.simulate_scenario(
        baseline_state=well_heavy_cold,
        modifications={"viscosity_cp": 260.0},
        scenario_name="S3_Diluent_Viscosity_Treatment",
        scenario_description="Downhole light hydrocarbon / solvent dosing reduces effective crude viscosity from 850 cP to 260 cP.",
        thermal_coupling=False,
    )
    scenarios.append(s3)

    # Scenario 4: Pump Speed Optimization (7.5 -> 4.8 SPM) on Inflow-Deficit Well
    s4 = predictor.simulate_scenario(
        baseline_state=well_inflow_deficit,
        modifications={"spm": 4.8, "fluid_level_ft": 2600.0},
        scenario_name="S4_POC_Throttling_Fluid_Recovery",
        scenario_description="Throttle SPM from 7.5 to 4.8 with Pump-Off Control duty cycling, allowing fluid level to recover from 1400 ft to 2600 ft.",
        thermal_coupling=False,
    )
    scenarios.append(s4)

    # Scenario 5: Gas Separator Installation on Gassy Well
    s5 = predictor.simulate_scenario(
        baseline_state=well_gassy,
        modifications={"gas_fraction": 0.04},
        scenario_name="S5_Downhole_Gas_Separation",
        scenario_description="Downhole gas separator reduces barrel free gas fraction from 30% to 4%, restoring volumetric pump fullness.",
        thermal_coupling=False,
    )
    scenarios.append(s5)

    # Scenario 6: Aggressive Overpumping on Nominal Well (5.5 -> 10.5 SPM)
    s6 = predictor.simulate_scenario(
        baseline_state=well_nominal,
        modifications={"spm": 10.5, "fluid_level_ft": 1600.0},
        scenario_name="S6_Aggressive_Overpumping_Drawdown",
        scenario_description="Attempting to force production by increasing SPM to 10.5 draws down fluid level, triggering fluid pound and extreme power demand.",
        thermal_coupling=False,
    )
    scenarios.append(s6)

    # Scenario 7: Combined Thermal + VFD Optimal Co-tuning on Heavy Cold Well
    s7 = predictor.simulate_scenario(
        baseline_state=well_heavy_cold,
        modifications={"temperature_f": 155.0, "spm": 6.0},
        scenario_name="S7_Thermal_Plus_VFD_CoOptimization",
        scenario_description="Combined strategy: CSS heating (+30°F) coupled with VFD tuning (6.0 SPM) for maximum energy efficiency and zero fault risk.",
        thermal_coupling=True,
    )
    scenarios.append(s7)

    # -------------------------------------------------------------
    # Tabulate and Save Results
    # -------------------------------------------------------------
    table_rows = []
    for sc in scenarios:
        b = sc.baseline
        s = sc.scenario
        d = sc.delta
        table_rows.append({
            "scenario_name": sc.scenario_name,
            "description": sc.scenario_description,
            # Operating Deltas
            "baseline_spm": b.spm,
            "scenario_spm": s.spm,
            "delta_spm": d.delta_spm,
            "baseline_temp_f": b.temperature_f,
            "scenario_temp_f": s.temperature_f,
            "delta_temp_f": d.delta_temperature_f,
            "baseline_visc_cp": b.viscosity_cp,
            "scenario_visc_cp": s.viscosity_cp,
            "delta_visc_cp": d.delta_viscosity_cp,
            "pct_visc_change": d.pct_viscosity_change,
            # Mechanical & Production
            "baseline_pprl_lbf": b.pprl_lbf,
            "scenario_pprl_lbf": s.pprl_lbf,
            "delta_pprl_lbf": d.delta_pprl_lbf,
            "baseline_prod_bpd": b.net_production_bpd,
            "scenario_prod_bpd": s.net_production_bpd,
            "delta_prod_bpd": d.delta_production_bpd,
            "pct_prod_change": d.pct_production_change,
            # Energy
            "baseline_power_kw": b.polished_rod_power_kw,
            "scenario_power_kw": s.polished_rod_power_kw,
            "delta_power_kw": d.delta_power_kw,
            "baseline_sec_kwh_bbl": b.specific_energy_kwh_per_bbl,
            "scenario_sec_kwh_bbl": s.specific_energy_kwh_per_bbl,
            "delta_sec_kwh_bbl": d.delta_specific_energy_kwh_bbl,
            # Diagnostic Risks
            "baseline_rf_risk": b.rod_floating_risk,
            "scenario_rf_risk": s.rod_floating_risk,
            "delta_rf_risk": d.delta_rod_floating_risk,
            "baseline_fp_risk": b.fluid_pound_risk,
            "scenario_fp_risk": s.fluid_pound_risk,
            "delta_fp_risk": d.delta_fluid_pound_risk,
            "baseline_gi_risk": b.gas_interference_risk,
            "scenario_gi_risk": s.gas_interference_risk,
            "delta_gi_risk": d.delta_gas_interference_risk,
            "baseline_composite_risk": b.composite_risk_score,
            "scenario_composite_risk": s.composite_risk_score,
            "delta_composite_risk": d.delta_composite_risk,
            "baseline_severity": b.severity,
            "scenario_severity": s.severity,
        })

    res_df = pd.DataFrame(table_rows)
    out_csv = RESULTS_DIR / "whatif_scenarios_results.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Wrote {len(res_df)} scenario evaluations to {out_csv}")

    # Human-readable report
    report_path = RESULTS_DIR / "whatif_scenarios_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TEJAS Phase 7 — Physics-Informed What-If Scenario Analysis Report\n")
        f.write("=" * 80 + "\n\n")

        for sc in scenarios:
            b, s, d = sc.baseline, sc.scenario, sc.delta
            f.write(f"[{sc.scenario_name}]\n")
            f.write(f"Description: {sc.scenario_description}\n")
            f.write(f"  Operational Changes:\n")
            f.write(f"    SPM:           {b.spm:.1f} -> {s.spm:.1f} ({d.delta_spm:+.1f})\n")
            f.write(f"    Temperature:   {b.temperature_f:.1f} °F -> {s.temperature_f:.1f} °F ({d.delta_temperature_f:+.1f} °F)\n")
            f.write(f"    Viscosity:     {b.viscosity_cp:.0f} cP -> {s.viscosity_cp:.0f} cP ({d.delta_viscosity_cp:+.0f} cP / {d.pct_viscosity_change:+.1f}%)\n")
            f.write(f"    Fluid Mobility: {b.mobility_index:.2f} -> {s.mobility_index:.2f} (idx)\n")
            f.write(f"  Mechanical & SRP Loading Response:\n")
            f.write(f"    PPRL:          {b.pprl_lbf:.0f} lbf -> {s.pprl_lbf:.0f} lbf ({d.delta_pprl_lbf:+.0f} lbf)\n")
            f.write(f"    MPRL:          {b.mprl_lbf:.0f} lbf -> {s.mprl_lbf:.0f} lbf ({d.delta_mprl_lbf:+.0f} lbf)\n")
            f.write(f"    Load Range:    {b.load_range_lbf:.0f} lbf -> {s.load_range_lbf:.0f} lbf ({d.delta_load_range_lbf:+.0f} lbf)\n")
            f.write(f"  Production & Energy Implications:\n")
            f.write(f"    Net Production: {b.net_production_bpd:.1f} bpd -> {s.net_production_bpd:.1f} bpd ({d.delta_production_bpd:+.1f} bpd / {d.pct_production_change:+.1f}%)\n")
            f.write(f"    Power Demand:  {b.polished_rod_power_kw:.2f} kW -> {s.polished_rod_power_kw:.2f} kW ({d.delta_power_kw:+.2f} kW)\n")
            f.write(f"    Specific Energy:{b.specific_energy_kwh_per_bbl:.2f} -> {s.specific_energy_kwh_per_bbl:.2f} kWh/bbl ({d.delta_specific_energy_kwh_bbl:+.2f} kWh/bbl)\n")
            f.write(f"  Diagnostic Risk Shift:\n")
            f.write(f"    Rod Floating:  {b.rod_floating_risk:.1f} -> {s.rod_floating_risk:.1f} ({d.delta_rod_floating_risk:+.1f})\n")
            f.write(f"    Fluid Pound:   {b.fluid_pound_risk:.1f} -> {s.fluid_pound_risk:.1f} ({d.delta_fluid_pound_risk:+.1f})\n")
            f.write(f"    Gas Interf:    {b.gas_interference_risk:.1f} -> {s.gas_interference_risk:.1f} ({d.delta_gas_interference_risk:+.1f})\n")
            f.write(f"    Overall Risk:  {b.composite_risk_score:.1f} ({b.severity}) -> {s.composite_risk_score:.1f} ({s.severity})\n")
            f.write(f"  Key Insights:\n")
            for ins in sc.engineering_insights:
                f.write(f"    * {ins}\n")
            f.write("-" * 80 + "\n\n")

    print(f"Wrote human-readable report to {report_path}")

    # Output Summary Table to Console
    print("\n" + "=" * 80)
    print("PHASE 7 WHAT-IF SCENARIOS SUMMARY")
    print("=" * 80)
    summary_cols = ["scenario_name", "delta_visc_cp", "delta_pprl_lbf", "delta_prod_bpd", "delta_power_kw", "delta_composite_risk", "scenario_severity"]
    print(res_df[summary_cols].to_string(index=False))


if __name__ == "__main__":
    main()
