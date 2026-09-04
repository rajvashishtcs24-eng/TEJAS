# TEJAS — Phase 8: Multi-Objective Optimization & Digital Twin Decision Engine

## 1. Overview & Operational Role

The **Phase 8 Digital Twin Optimizer** acts as an explainable, physics-constrained **Decision Support Engine** for oilfield operators and production engineers managing Sucker-Rod Pumping (SRP) systems in heavy-oil assets.

```
+----------------------------------------------------------------------------------------------------+
|                         TEJAS Phase 8 Digital Twin Decision Engine                                 |
+----------------------------------------------------------------------------------------------------+
  [CURRENT OPERATING STATE]       [SEARCH & SIMULATION]            [DECISION RECOMMENDATIONS]
  - Well Depth, Stroke, Rods      - Feasible Parameter Space       - Best Balanced Operating Point
  - Current SPM, Temp, Viscosity  - Phase 7.5 Physics Engine --->  - Max Safe Production Scenario
  - Annular Fluid Level, Gas %    - Multi-Objective Pareto Filter  - Max Energy Efficiency Scenario
  - Diagnostic Risk & Severity    - Utility Scoring & Constraints  - Plain-Language Physics Rationale
+----------------------------------------------------------------------------------------------------+
```

> **Important Operational Guardrail:** This system provides **Decision Support & Advisory Intelligence**, NOT automated/autonomous control. All recommendations must be reviewed by qualified reservoir and production personnel before setpoint implementation.

---

## 2. Multi-Objective Optimization Formulation

The engine balances three competing operational goals:

1. **Maximize Net Production:** $\max Q_{net} \quad [\text{bbl/day}]$
2. **Minimize Mechanical & Diagnostic Risk:** $\min R_{composite} \quad [0 - 100]$
3. **Minimize Energy Consumption & Power Demand:** $\min \text{SEC} \quad [\text{kWh / bbl}], \quad \min \text{Power } (kW)$

### Mathematical Utility Function:
For all feasible non-dominated operating points on the Pareto frontier:
$$U = w_{prod} \cdot \bar{Q}_{net} + w_{risk} \cdot (1 - \bar{R}_{composite}) + w_{energy} \cdot (1 - \overline{\text{SEC}})$$
where $\bar{Q}, \bar{R}, \overline{\text{SEC}} \in [0, 1]$ are min-max normalized metrics across feasible candidates, with default balanced weights:
* $w_{prod} = 0.45$ (Production priority)
* $w_{risk} = 0.35$ (Equipment protection & rod string integrity)
* $w_{energy} = 0.20$ (Electrical power & carbon footprint minimization)

---

## 3. Explicit Engineering Constraints

Candidates violating any of the following physical or operational limits are flagged and excluded from recommendation:

| Constraint | Limit / Threshold | Physical Rationale |
| :--- | :---: | :--- |
| **Max Acceptable Risk** | $R_{composite} < 48.0$ | Prohibits recommendations in High ($\ge 55$) or Critical ($\ge 80$) severity tiers |
| **PPRL Limit** | $\text{PPRL} \le 24,000\text{ lbf}$ | Prevents tensile rod-string failure or gearbox gear-tooth overload |
| **MPRL Limit** | $\text{MPRL} \ge 500\text{ lbf}$ | Prevents rod compression and severe downstroke buckling / floating |
| **VFD Speed Bounds** | $3.0 \le SPM \le 10.0$ | Operational limits of surface pumping units and variable speed drives |

---

## 4. Candidate Search Space

The optimizer generates candidate what-if states tailored to the well's active diagnostic profile:
1. **VFD Speed Sweeps:** $SPM \in [3.5, 9.5]$ in $0.5\text{ SPM}$ increments.
2. **Thermal Stimulation (CSS):** Wellbore temperature increases $\Delta T \in [+15, +30, +45]^\circ\text{F}$.
3. **Pump-Off Control (POC) Duty Cycling:** Fluid level recovery to $[2400, 3000, 3500]\text{ ft}$ at throttled SPM $[4.0, 5.0, 6.0]$.
4. **Downhole Gas Separation Retrofits:** Free gas fraction reduction from $\ge 20\% \rightarrow 4\%$.
5. **Thermal + VFD Co-Optimization:** Simultaneous thermal heating and speed tuning.

---

## 5. Decision Output Schema

For any given well, the decision engine outputs:

```
[WELL_ID] DIGITAL TWIN OPTIMIZATION RECOMMENDATION
--------------------------------------------------------------------------------
1. CURRENT STATE:
   SPM | Temperature | Viscosity | Net Production | Power (kW) | SEC (kWh/bbl) | Risk / Severity
   [Optional Warning if in High/Critical tier]

2. BEST RECOMMENDED SCENARIO:
   Target Modifications | Predicted State | Expected Deltas (Prod, Power, Risk) | Utility Score

3. ALTERNATIVE SAFE SCENARIOS:
   - [Max Safe Production]: Highest production within safe risk bounds
   - [Max Energy Efficiency]: Lowest kWh/bbl specific energy within safe risk bounds

4. WHY THIS SCENARIO WAS SELECTED:
   Plain-language engineering reasoning explaining physics mechanism and expected outcomes.
```

---

## 6. Demonstration Case Studies

### Case 1: Cold Heavy Crude (`WELL-019`) — Rod Floating Hazard
* **Current State:** $SPM = 8.5$, $T = 125^\circ\text{F}$, $\mu = 850\text{ cP}$, $\text{Prod} = 346.2\text{ bpd}$, $\text{Power} = 11.55\text{ kW}$, $\text{Risk} = 76.0/100$ (**High Severity**)
* **Recommended Scenario (`CoOpt_T+35F_SPM_5.5`):**
  - **Setpoints:** $T = 160^\circ\text{F}$ (CSS Steam Soaking), $SPM = 5.5$
  - **Expected Deltas:** $\Delta \text{Visc} = -531\text{ cP}$, $\Delta \text{Power} = -5.88\text{ kW} \text{ (-50.9\%)}$, $\Delta \text{SEC} = -0.32\text{ kWh/bbl}$, $\Delta \text{Risk} = -65.0\text{ pts}$
  - **New State:** $\text{Prod} = 283.9\text{ bpd}$, $\text{Power} = 5.67\text{ kW}$, $\text{Risk} = 11.0/100$ (**Low Severity**)
* **Selection Reason:** Thermal CSS treatment ($160^\circ\text{F}$) cuts viscosity by $-62.5\%$, unlocking fluid mobility; Adjusting pumping speed from 8.5 to 5.5 SPM rebalances rod kinematics, avoids downstroke impact, cuts power demand in half, and eliminates rod-floating hazard.

### Case 2: Inflow-Deficit Pounding Well (`WELL-005`) — Fluid Pound Hazard
* **Current State:** $SPM = 7.5$, $FL = 1400\text{ ft}$ (Annulus Depleted), $\text{Prod} = 186.8\text{ bpd}$, $\text{Power} = 5.73\text{ kW}$, $\text{Risk} = 69.9/100$ (**High Severity**)
* **Recommended Scenario (`POC_SPM_5.0_FL_3000`):**
  - **Setpoints:** $SPM = 5.0$, $FL = 3000\text{ ft}$ (via POC Duty-Cycle / Off-Timer)
  - **Expected Deltas:** $\Delta \text{Prod} = +66.8\text{ bpd} \text{ (+35.8\%)}$, $\Delta \text{Power} = -1.21\text{ kW} \text{ (-21.1\%)}$, $\Delta \text{Risk} = -57.4\text{ pts}$
  - **New State:** $\text{Prod} = 253.6\text{ bpd}$, $\text{Power} = 4.52\text{ kW}$, $\text{Risk} = 12.5/100$ (**Low Severity**)
* **Selection Reason:** Throttling pump speed from 7.5 to 5.0 SPM allows annular fluid to recover to 3000 ft, raising pump barrel fillage from 50% to 100%, producing **+35.8% more net liquid with lower power and zero fluid pound**.

### Case 3: Gassy Heavy Oil Well (`WELL-007`) — Gas Interference Hazard
* **Current State:** $SPM = 6.0$, $\text{Gas} = 30\%$, $FL = 2000\text{ ft}$, $\text{Prod} = 151.3\text{ bpd}$, $\text{Risk} = 100.0/100$ (**Critical Severity**)
* **Recommended Scenario (`GasSep_SPM_7.0`):**
  - **Setpoints:** Gas Fraction $= 4\%$ (via Downhole Gas Separator), $SPM = 7.0$
  - **Expected Deltas:** $\Delta \text{Prod} = +149.4\text{ bpd} \text{ (+98.7\%)}$, $\Delta \text{SEC} = -0.31\text{ kWh/bbl}$, $\Delta \text{Risk} = -57.3\text{ pts}$
  - **New State:** $\text{Prod} = 300.7\text{ bpd}$, $\text{Power} = 7.41\text{ kW}$, $\text{Risk} = 42.7/100$ (**Medium Severity**)
* **Selection Reason:** Downhole gas separation eliminates compressible vapor cushioning in the pump barrel, doubling volumetric lifting capacity.

---

## 7. Deliverables & Output Files

1. **Optimizer Implementation:** [`data_pipeline/optimizer.py`](file:///c:/Users/rajva/Downloads/TEJAS/data_pipeline/optimizer.py)
2. **Demonstration & Runner Script:** [`run_phase8_optimizer.py`](file:///c:/Users/rajva/Downloads/TEJAS/run_phase8_optimizer.py)
3. **Tabular Results:** [`results/phase8/optimization_results.csv`](file:///c:/Users/rajva/Downloads/TEJAS/results/phase8/optimization_results.csv)
4. **Full Decision Report:** [`results/phase8/optimization_report.txt`](file:///c:/Users/rajva/Downloads/TEJAS/results/phase8/optimization_report.txt)
