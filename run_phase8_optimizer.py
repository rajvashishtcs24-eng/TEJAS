"""
run_phase8_optimizer.py
-----------------------
Executes multi-objective optimization across representative well profiles
and outputs actionable Digital Twin decision recommendations.
"""

from pathlib import Path
import json
import pandas as pd
from data_pipeline.well_prediction import WellOperatingState
from data_pipeline.optimizer import TEJASWellOptimizer, OptimizationRecommendation

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results" / "phase8"


def format_state_summary(res):
    return (
        f"SPM: {res.spm:.1f} | Temp: {res.temperature_f:.1f}°F | Visc: {res.viscosity_cp:.0f} cP | "
        f"Prod: {res.net_production_bpd:.1f} bpd | Power: {res.polished_rod_power_kw:.2f} kW | "
        f"SEC: {res.specific_energy_kwh_per_bbl:.2f} kWh/bbl | Risk: {res.composite_risk_score:.1f}/100 ({res.severity})"
    )


def format_delta_summary(d):
    return (
        f"Delta Prod: {d.delta_production_bpd:+.1f} bpd ({d.pct_production_change:+.1f}%) | "
        f"Delta Power: {d.delta_power_kw:+.2f} kW | "
        f"Delta SEC: {d.delta_specific_energy_kwh_bbl:+.2f} kWh/bbl | "
        f"Delta Risk: {d.delta_composite_risk:+.1f} pts"
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    optimizer = TEJASWellOptimizer(
        max_acceptable_risk=48.0,
        w_production=0.45,
        w_risk=0.35,
        w_energy=0.20,
    )

    # -----------------------------------------------------------------
    # Define 4 Representative Well Operating Profiles
    # -----------------------------------------------------------------
    wells = [
        # Profile 1: Cold Heavy Crude Well (Rod Floating Hazard)
        WellOperatingState(
            well_id="WELL-019",
            spm=8.5,
            temperature_f=125.0,
            viscosity_cp=850.0,
            fluid_level_ft=3400.0,
            stroke_length_in=120.0,
            pump_depth_ft=5000.0,
            plunger_diameter_in=2.0,
            rod_weight_per_ft=2.20,
        ),
        # Profile 2: Inflow-Deficit Pounding Well (Fluid Pound Hazard)
        WellOperatingState(
            well_id="WELL-005",
            spm=7.5,
            temperature_f=138.0,
            viscosity_cp=420.0,
            fluid_level_ft=1400.0,
            stroke_length_in=120.0,
            pump_depth_ft=4600.0,
            plunger_diameter_in=2.0,
            rod_weight_per_ft=2.16,
        ),
        # Profile 3: Gassy Well (Gas Interference Hazard)
        WellOperatingState(
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
        ),
        # Profile 4: Balanced Nominal Well
        WellOperatingState(
            well_id="WELL-003",
            spm=5.5,
            temperature_f=145.0,
            viscosity_cp=380.0,
            fluid_level_ft=3600.0,
            stroke_length_in=120.0,
            pump_depth_ft=4500.0,
            plunger_diameter_in=2.0,
            rod_weight_per_ft=2.16,
        ),
    ]

    recommendations: List[OptimizationRecommendation] = []
    for w in wells:
        rec = optimizer.optimize_well(w)
        recommendations.append(rec)

    # -----------------------------------------------------------------
    # Print Console Decision Logs
    # -----------------------------------------------------------------
    print("=" * 80)
    print("TEJAS Phase 8 — Multi-Objective Optimization Decision Engine")
    print("=" * 80)

    for rec in recommendations:
        cur = rec.current_state
        best = rec.recommended_scenario
        max_p = rec.max_production_scenario
        max_e = rec.max_efficiency_scenario

        print(f"\n[{rec.well_id}] DIGITAL TWIN OPTIMIZATION RECOMMENDATION")
        print("-" * 80)
        print(f"CURRENT STATE:")
        print(f"  {format_state_summary(cur)}")
        if rec.operational_warnings:
            for w in rec.operational_warnings:
                print(f"  [WARNING] {w}")

        print(f"\nBEST RECOMMENDED SCENARIO ({best.scenario_id}):")
        print(f"  Parameters:    {best.modifications}")
        print(f"  Result:        {format_state_summary(best.state)}")
        print(f"  Expected Delta:{format_delta_summary(best.delta)}")
        print(f"  Utility Score: {best.utility_score:.4f} (Pareto: {best.is_pareto_optimal})")

        print(f"\nALTERNATIVE SAFE SCENARIOS:")
        print(f"  [Max Safe Production] ({max_p.scenario_id}):")
        print(f"    Parameters:  {max_p.modifications}")
        print(f"    Result:      Prod={max_p.state.net_production_bpd:.1f} bpd | Power={max_p.state.polished_rod_power_kw:.2f} kW | Risk={max_p.state.composite_risk_score:.1f} ({max_p.state.severity})")
        print(f"  [Max Energy Efficiency] ({max_e.scenario_id}):")
        print(f"    Parameters:  {max_e.modifications}")
        print(f"    Result:      SEC={max_e.state.specific_energy_kwh_per_bbl:.2f} kWh/bbl | Power={max_e.state.polished_rod_power_kw:.2f} kW | Prod={max_e.state.net_production_bpd:.1f} bpd")

        print(f"\nWHY THIS SCENARIO WAS SELECTED:")
        print(f"  {rec.selection_rationale}")
        print("-" * 80)

    # -----------------------------------------------------------------
    # Save Outputs
    # -----------------------------------------------------------------
    # 1. Summary CSV
    csv_rows = []
    for rec in recommendations:
        cur = rec.current_state
        best = rec.recommended_scenario
        d = best.delta
        csv_rows.append({
            "well_id": rec.well_id,
            "baseline_spm": cur.spm,
            "baseline_temp_f": cur.temperature_f,
            "baseline_visc_cp": cur.viscosity_cp,
            "baseline_prod_bpd": cur.net_production_bpd,
            "baseline_power_kw": cur.polished_rod_power_kw,
            "baseline_risk": cur.composite_risk_score,
            "baseline_severity": cur.severity,
            "recommended_scenario": best.scenario_id,
            "rec_modifications": json.dumps(best.modifications),
            "rec_spm": best.state.spm,
            "rec_temp_f": best.state.temperature_f,
            "rec_visc_cp": best.state.viscosity_cp,
            "rec_prod_bpd": best.state.net_production_bpd,
            "rec_power_kw": best.state.polished_rod_power_kw,
            "rec_risk": best.state.composite_risk_score,
            "rec_severity": best.state.severity,
            "delta_prod_bpd": d.delta_production_bpd,
            "delta_power_kw": d.delta_power_kw,
            "delta_risk": d.delta_composite_risk,
            "delta_sec_kwh_bbl": d.delta_specific_energy_kwh_bbl,
            "utility_score": best.utility_score,
            "pareto_candidates_count": len(rec.all_pareto_scenarios),
            "selection_rationale": rec.selection_rationale,
        })

    csv_df = pd.DataFrame(csv_rows)
    out_csv = RESULTS_DIR / "optimization_results.csv"
    csv_df.to_csv(out_csv, index=False)
    print(f"\nWrote optimization summary to {out_csv}")

    # 2. Detailed Report
    report_path = RESULTS_DIR / "optimization_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("TEJAS Phase 8 — Multi-Objective Optimization Report\n")
        f.write("=" * 80 + "\n\n")

        for rec in recommendations:
            cur = rec.current_state
            best = rec.recommended_scenario
            max_p = rec.max_production_scenario
            max_e = rec.max_efficiency_scenario

            f.write(f"WELL ID: {rec.well_id}\n")
            f.write("=" * 80 + "\n")
            f.write(f"1. CURRENT OPERATING STATE:\n")
            f.write(f"   {format_state_summary(cur)}\n")
            if rec.operational_warnings:
                for w in rec.operational_warnings:
                    f.write(f"   [WARNING] {w}\n")

            f.write(f"\n2. BEST RECOMMENDED OPERATING POINT ({best.scenario_id}):\n")
            f.write(f"   Target Parameters: {best.modifications}\n")
            f.write(f"   Predicted State:   {format_state_summary(best.state)}\n")
            f.write(f"   Expected Deltas:   {format_delta_summary(best.delta)}\n")
            f.write(f"   Utility Score:     {best.utility_score:.4f} (Pareto Optimal: {best.is_pareto_optimal})\n")

            f.write(f"\n3. ALTERNATIVE SAFE OPERATING OPTIONS:\n")
            f.write(f"   A) Maximum Safe Production ({max_p.scenario_id}):\n")
            f.write(f"      Parameters: {max_p.modifications}\n")
            f.write(f"      State:      Prod={max_p.state.net_production_bpd:.1f} bpd | Power={max_p.state.polished_rod_power_kw:.2f} kW | Risk={max_p.state.composite_risk_score:.1f} ({max_p.state.severity})\n")
            f.write(f"   B) Maximum Energy Efficiency ({max_e.scenario_id}):\n")
            f.write(f"      Parameters: {max_e.modifications}\n")
            f.write(f"      State:      SEC={max_e.state.specific_energy_kwh_per_bbl:.2f} kWh/bbl | Power={max_e.state.polished_rod_power_kw:.2f} kW | Prod={max_e.state.net_production_bpd:.1f} bpd\n")

            f.write(f"\n4. SELECTION RATIONALE & ENGINEERING REASONING:\n")
            f.write(f"   {rec.selection_rationale}\n\n")
            f.write(f"5. PARETO FRONTIER CANDIDATES ({len(rec.all_pareto_scenarios)} Total):\n")
            for pc in rec.all_pareto_scenarios[:6]:
                f.write(f"   * [{pc.scenario_id}] {pc.modifications} -> Prod={pc.state.net_production_bpd:.1f} bpd, Power={pc.state.polished_rod_power_kw:.2f} kW, Risk={pc.state.composite_risk_score:.1f} ({pc.state.severity})\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Wrote detailed report to {report_path}")


if __name__ == "__main__":
    main()
