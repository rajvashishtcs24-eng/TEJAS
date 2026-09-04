# TEJAS — Phase 7 & 7.5: Physics-Informed Well Prediction & Calibration Layer

## 1. Overview & Physics Architecture

The **Phase 7.5 Physics-Informed Well Prediction Layer** provides a calibrated, explainable simulation and sensitivity module for heavy-oil wells operating on Sucker-Rod Pumping (SRP) systems.

It models the governing causal physics chains:
$$\text{Temperature } (T) \longrightarrow \text{Viscosity } (\mu) \longrightarrow \text{Fluid Mobility } (\lambda) \longrightarrow \text{SRP Loading} \longrightarrow \text{Condition Risks} \longrightarrow \text{Production \& Power}$$

```
+-----------------------------------------------------------------------------------------------+
|                             What-If Scenario Simulation Engine                                |
+-----------------------------------------------------------------------------------------------+
  [SYNTHETIC_INPUT]            [GOVERNING PHYSICS & ASSUMPTIONS]         [PREDICTED_OUTPUT]
  - SPM                        - Andrade/Arrhenius Thermal Rheology       - Net Production (bpd)
  - Wellbore Temp (°F)   --->  - SHM Kinematics & Rod Inertia       --->  - PPRL / MPRL / Range
  - Viscosity (cP)             - Annulus Couette Viscous Shear           - Power (kW) & SEC (kWh/bbl)
  - Annular Fluid Level (ft)   - API RP 11L Volumetric Inflow-Fillage     - Diagnostic Risk Deltas
+-----------------------------------------------------------------------------------------------+
```

---

## 2. Explicit Taxonomy of Quantities

| Category | Description | Variables in System |
| :--- | :--- | :--- |
| **`[SYNTHETIC_INPUT]`** | Direct synthetic or operational well inputs | Well architecture ($L_{pump}, S, D_{plunger}, \rho_{rod}$), $SPM, T, \mu, FL, \text{Gas Fraction}$ |
| **`[PHYSICS_DERIVED]`** | Evaluated via fundamental first principles | Static rod weight ($W_{air}, W_{buoy}$), fluid column load ($F_o$), SHM rod inertia ($F_{inertia}$), Couette viscous shear ($F_{drag}$), theoretical displacement ($Q_{theor}$), cycle work ($W_{stroke}$), polished rod power ($kW$), specific energy ($SEC$) |
| **`[HEURISTIC_ASSUMP]`** | Empirical / domain-derived coupling relationships | Thermal viscosity decay constant ($\beta = 0.028 \text{ / }^\circ\text{F}$), critical submergence threshold ($FL_{crit} = 2800\text{ ft}$), card work fullness coefficient ($\eta_{card} = 0.72$), valve intake choking factor $\eta_{valve}(\mu)$, rod float stroke loss $\eta_{rf\_stroke}(\text{Risk}_{RF})$ |
| **`[PREDICTED_OUTPUT]`** | Simulated scenario state & differential responses | $\Delta \mu, \Delta \text{PPRL}, \Delta \text{Production}, \Delta \text{Power}, \Delta \text{Risk}_{RF}, \Delta \text{Risk}_{FP}, \Delta \text{Risk}_{GI}, \Delta \text{SEC}$ |

---

## 3. Calibrated Governing Equations (Phase 7.5 Audit)

### 1. Thermal Viscosity Coupling ($T \rightarrow \mu \rightarrow \lambda$)
* **Viscosity Rheology:** `[HEURISTIC_ASSUMP]`
  $$\mu(T) = \mu_0 \cdot \exp\left(-\beta \cdot (T - T_0)\right) \quad [\text{cP}]$$
  where $\beta = 0.028 \text{ / }^\circ\text{F}$ (Andrade exponential approximation).
* **Relative Mobility Index:** `[PHYSICS_DERIVED]`
  $$\lambda_{rel} = \frac{400.0}{\mu}$$

### 2. Sucker-Rod String Kinematics & Polished Rod Loads
* **Static Buoyant Rod Weight:** `[PHYSICS_DERIVED]`
  $$W_{buoyant} = (\rho_{rod} \cdot L_{pump}) \cdot \left(1 - \frac{\text{SG}_{fluid}}{\text{SG}_{steel}}\right) \quad [\text{lbf}]$$
* **Plunger Static Hydrostatic Fluid Load:** `[PHYSICS_DERIVED]`
  $$F_{fluid} = (0.433 \cdot \text{SG}_{fluid}) \cdot FL \cdot \left(\frac{\pi}{4} D_{plunger}^2\right) \quad [\text{lbf}]$$
* **Simple Harmonic Motion Rod Inertial Amplitude:** `[PHYSICS_DERIVED]`
  $$F_{inertia} = \left(\frac{W_{air}}{g}\right) \cdot \left(\frac{S}{24}\right) \cdot \left(\frac{2\pi \cdot SPM}{60}\right)^2 \quad [\text{lbf}]$$
* **Downhole Hydrodynamic Viscous Drag (Couette Shear):** `[PHYSICS_DERIVED]`
  $$F_{drag} = 0.25 \cdot \left(\frac{L_{pump}}{4500}\right) \cdot \left(\frac{S}{120}\right) \cdot \left(\frac{\mu}{400}\right) \cdot \left(\frac{SPM}{6}\right) \cdot 350.0 \quad [\text{lbf}]$$
* **Polished Rod Loads:** `[PHYSICS_DERIVED]`
  $$\text{PPRL} = W_{buoyant} + F_{fluid} + F_{inertia} + F_{friction, mech} + F_{drag}$$
  $$\text{MPRL} = W_{buoyant} - F_{inertia} - F_{friction, mech} - F_{drag}$$

### 3. Volumetric Delivery & Production Coupling (Calibrated in Phase 7.5)
* **Theoretical Pump Displacement:** `[PHYSICS_DERIVED]`
  $$Q_{theor} = 0.11662 \cdot D_{plunger}^2 \cdot S \cdot SPM \quad [\text{bpd}]$$
* **Pump Barrel Submergence Fillage:** `[PHYSICS_DERIVED + HEURISTIC_ASSUMP]`
  $$\eta_{fill} = \text{clip}\left(\frac{FL}{2800}, 0.20, 1.0\right) \times \text{clip}(1.0 - 1.5 \cdot \text{Gas Fraction}, 0.20, 1.0)$$
* **Viscous Valve Intake Throttling Efficiency:** `[HEURISTIC_ASSUMP]`
  $$\eta_{valve}(\mu) = \text{clip}\left(1.0 - 0.12 \cdot \frac{\mu - 300}{700}, 0.76, 1.0\right)$$
* **Rod Floating Kinematic Stroke Loss:** `[HEURISTIC_ASSUMP]`
  $$\eta_{rf\_stroke} = \text{clip}\left(1.0 - 0.20 \cdot \frac{\text{Risk}_{RF} - 30}{70}, 0.78, 1.0\right)$$
* **Net Surface Liquid Production:** `[PREDICTED_OUTPUT]`
  $$Q_{net} = Q_{theor} \cdot \left(\eta_{fill} \cdot \eta_{valve} \cdot \eta_{rf\_stroke}\right) \quad [\text{bpd}]$$

### 4. Energy & Power Consumption
* **Stroke Cycle Work Envelope:** `[PHYSICS_DERIVED]`
  $$W_{stroke} = (\text{PPRL} - \text{MPRL}) \cdot S \cdot 0.72 \quad [\text{lbf}\cdot\text{in}]$$
* **Polished Rod Power Demand:** `[PHYSICS_DERIVED]`
  $$\text{Power } (kW) = \left(\frac{W_{stroke} \cdot SPM}{12 \cdot 33000}\right) \times 0.7457 \quad [\text{kW}]$$
* **Specific Energy Consumption:** `[PHYSICS_DERIVED]`
  $$\text{SEC} = \frac{\text{Power } (kW) \times 24}{Q_{net}} \quad [\text{kWh / bbl}]$$

---

## 4. Audit & Calibration Findings (Phase 7.5)

| Audited Relationship | Pre-Audit Observation | Physics Diagnostic & Resolution | Status |
| :--- | :--- | :--- | :---: |
| **Thermal $+35^\circ\text{F} \rightarrow \Delta Q$** | $\Delta Q = 0$ at $-531\text{ cP}$ | In cold heavy oil ($850\text{ cP}$), viscous standing valve choking and rod float kinematics reduce delivery efficiency to $\sim 72\%$. Heating restores $\eta_{vol}$ to $100\%$, yielding an authentic $+80.9\text{ bpd}$ recovery. | ✅ **Calibrated** |
| **SPM Throttling ($8.5 \rightarrow 5.0$)** | $-61.8\%$ production drop | Throttling eliminates rod float stroke loss ($\eta_{rf\_stroke} \rightarrow 1.0$), cushioning production drop while cutting power by $-6.40\text{ kW}$. | ✅ **Calibrated** |
| **POC Inflow Recovery ($FL+1200\text{ft}$)** | $+39.2\text{ bpd}$ gain | Authentically models Pump-Off Control: matching SPM to inflow raises barrel fillage from $50\% \rightarrow 93\%$, delivering higher net liquid with lower power. | ✅ **Verified Sound** |
| **Overpumping ($SPM \rightarrow 10.5$)** | $+2.1\text{ bpd}$ at $+178\%$ power | Captures steep diminishing volumetric returns when drawdown induces fluid pound ($\eta_{fill} = 57\%$). | ✅ **Verified Sound** |

---

## 5. Calibrated What-If Scenario Matrix

| Scenario | Operating Shift | $\Delta \text{Viscosity}$ | $\Delta \text{PPRL}$ | $\Delta \text{Production}$ | $\Delta \text{Power}$ | $\Delta \text{Risk}$ | Final Severity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **S1: Thermal CSS Heating** | $T: 125 \rightarrow 160^\circ\text{F}$ ($+35^\circ\text{F}$) | **-531 cP** ($-62.5\%$) | -183 lbf | **+80.9 bpd** | -0.51 kW | **-36.7 pts** | **Medium** ($48.3$) |
| **S2: VFD Speed Throttling** | $SPM: 8.5 \rightarrow 5.0$ | 0 cP | **-1006 lbf** | -119.3 bpd | **-6.40 kW** | **-34.6 pts** | **Medium** ($50.4$) |
| **S3: Chemical Diluent Dosing**| $\mu: 850 \rightarrow 260\text{ cP}$ | **-590 cP** ($-69.4\%$) | -203 lbf | **+84.1 bpd** | -0.57 kW | **-38.2 pts** | **Medium** ($46.8$) |
| **S4: POC Inflow Recovery** | $SPM: 7.5 \rightarrow 4.8, FL+1200\text{ft}$ | 0 cP | +897 lbf | **+39.2 bpd** | -1.84 kW | **-51.6 pts** | **Low** ($23.4$) |
| **S5: Gas Separator Retrofit** | $\text{Gas}: 30\% \rightarrow 4\%$ | 0 cP | 0 lbf | **+107.3 bpd** | 0.00 kW | **-57.3 pts** | **Medium** ($27.7$) |
| **S6: Overpumping Drawdown** | $SPM: 5.5 \rightarrow 10.5, FL-2000\text{ft}$| 0 cP | -1108 lbf | +2.1 bpd | **+5.74 kW** | **+47.3 pts** | **High** ($73.4$) |
| **S7: Thermal + VFD Co-Tuning**| $T+30^\circ\text{F}, SPM=6.0$ | **-483 cP** | **-883 lbf** | -39.1 bpd | **-5.12 kW** | **-58.1 pts** | **Low** ($26.9$) |

---

## 6. Limitations & Scope

1. **Synthetic & Physics-Approximated Data Only:** Calibrations are based on established petroleum engineering relationships (API RP 11L, Couette shear, Andrade viscosity), not empirical field telemetry.
2. **Quasi-Static Lumped Rod Model:** Inertia is approximated via SHM kinematics. Elastic stress wave reflection along the rod string is not solved dynamically (wave equation omitted by design).
3. **No Automatic Inflow Transients:** Reservoir inflow response to thermal steam soaking is modeled as a steady-state mobility enhancement, without multi-phase transient reservoir simulation.
