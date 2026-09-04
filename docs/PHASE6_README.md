# TEJAS — Phase 6: Risk & Engineering Assessment Layer

## 1. Overview & System Architecture

The **Phase 6 Risk & Engineering Assessment Engine** provides physics-grounded, explainable diagnostic interpretation on top of the Phase 5B 1D-CNN classifier.

```
+------------------------------------+
| 200-Point Dynamometer Card Profile |
+------------------------------------+
                  |
                  v
       +--------------------+
       |  Phase 5B 1D-CNN   |
       +--------------------+
                  |
                  v  Softmax Probabilities: P(Normal), P(RF), P(FP), P(GI)
+-------------------------------------------------------------+
|          TEJAS Phase 6 Risk & Engineering Engine            |
|  - Operating Context (SPM, Viscosity, Temp, Fluid Level)    |
|  - Load & Impact Features (Downstroke Slope, MPRL, Work)    |
+-------------------------------------------------------------+
                  |
                  +---> Condition-Specific Risk Scores (0-100)
                  +---> Overall Severity Tier (Low, Medium, High, Critical)
                  +---> Physics-Grounded Engineering Explanation
                  +---> Actionable Operational Recommendations
```

> **Important Distinction:** The ML classifier computes statistical pattern probabilities from card geometry. The Risk Engine applies explicit petroleum engineering physics (stress dynamics, fluid drag, pump-off kinetics) to assess mechanical hazard and recommend operations.

---

## 2. Input Specifications

| Input Category | Variable | Units | Engineering Relevance |
| :--- | :--- | :--- | :--- |
| **Model Output** | `P(Condition)` | $[0, 1]$ | Softmax class probabilities from 1D-CNN |
| **Operating Context** | `viscosity` | $\text{cP}$ | Fluid drag resisting rod string downstroke motion |
| | `SPM` | $\text{spm}$ | Stroke velocity; bounds rod string free-fall time |
| | `temperature` | $^\circ\text{F}$ | Thermal coupling governing heavy oil viscosity |
| | `fluid_level` | $\text{ft}$ | Annular fluid column above pump (inflow indicator) |
| | `pump_depth` | $\text{ft}$ | Pump setting depth (determines submergence ratio) |
| | `production_rate` | $\text{bpd}$ | Well output volume; drops under gas lock/interference |
| **Card Features** | `max_abs_slope_down_z` | $\text{z-score/pos}$ | Magnitude of downstroke impact shock |
| | `MPRL_raw_lbf` | $\text{lbf}$ | Minimum polished rod load (identifies downstroke float) |
| | `card_area_shape_norm` | $[0, 1]$ | Volumetric card fullness (pump efficiency indicator) |

---

## 3. Physics-Informed Risk Formulations

### A. Rod Floating ($R_{RF}$)
* **Mechanism:** Heavy crude drag ($\mu$), high pumping speed ($SPM$), and cold well temperatures ($T$) prevent the rod string from dropping as fast as the polished rod. The rod string goes into compression and subsequently impacts the plunger near the bottom of the stroke.
* **Operating Index ($I_{RF}$):**
  $$I_{RF} = 0.35 \cdot f_{\mu} + 0.30 \cdot f_{SPM} + 0.15 \cdot f_T + 0.20 \cdot f_{spike}$$
  where:
  - $f_{\mu} = \text{clip}\left(\frac{\mu - 350}{650}, 0, 1\right)$
  - $f_{SPM} = \text{clip}\left(\frac{SPM - 5.5}{5.0}, 0, 1\right)$
  - $f_T = \text{clip}\left(\frac{145 - T}{45}, 0, 1\right)$
  - $f_{spike} = \text{clip}\left(\frac{\text{slope\_down\_z} - 15}{90}, 0, 1\right)$
* **Risk Score:**
  $$R_{RF} = P(RF) \times (35.0 + 65.0 \cdot I_{RF}) \in [0, 100]$$

### B. Fluid Pound ($R_{FP}$)
* **Mechanism:** Well inflow underperforms pump displacement. The pump barrel partially fills with liquid and vapor. On the downstroke, the plunger free-falls through vapor and violently slams into the fluid surface mid-stroke, causing shock fatigue on rod couplings and surface unit gearboxes.
* **Operating Index ($I_{FP}$):**
  $$I_{FP} = 0.35 \cdot f_{FL} + 0.20 \cdot f_{sub} + 0.30 \cdot f_{spike} + 0.15 \cdot f_{MPRL}$$
  where:
  - $f_{FL} = \text{clip}\left(\frac{3500 - \text{fluid\_level}}{2500}, 0, 1\right)$
  - $f_{sub} = \text{clip}\left(\frac{0.65 - (\text{fluid\_level}/\text{pump\_depth})}{0.50}, 0, 1\right)$
  - $f_{spike} = \text{clip}\left(\frac{\text{slope\_down\_z} - 15}{90}, 0, 1\right)$
  - $f_{MPRL} = \text{clip}\left(\frac{7500 - MPRL}{3500}, 0, 1\right)$
* **Risk Score:**
  $$R_{FP} = P(FP) \times (40.0 + 60.0 \cdot I_{FP}) \in [0, 100]$$

### C. Gas Interference ($R_{GI}$)
* **Mechanism:** Free gas in the pump barrel compresses and expands during the cycle without opening the traveling valve on schedule, reducing effective pumped displacement and overall thermal lifting efficiency.
* **Operating Index ($I_{GI}$):**
  $$I_{GI} = 0.45 \cdot f_{area} + 0.30 \cdot f_{prod} + 0.25 \cdot f_{range}$$
* **Risk Score:**
  $$R_{GI} = P(GI) \times (30.0 + 70.0 \cdot I_{GI}) \in [0, 100]$$

### D. Normal Operation ($R_{Normal}$)
$$R_{Normal} = P(\text{Normal}) \times 8.0 \times \text{clip}\left(\frac{SPM}{10.0}, 0.5, 1.5\right) \in [0, 15]$$

---

## 4. Severity Tier Mapping

The composite risk score $R = \max(R_{RF}, R_{FP}, R_{GI}, R_{Normal})$ is categorized into four operational tiers:

| Tier | Risk Range | Description & Urgency | Operational Protocol |
| :--- | :---: | :--- | :--- |
| **Low** | $0 \le R < 25$ | Nominal baseline operation | Continue standard monitoring and routine maintenance |
| **Medium** | $25 \le R < 55$ | Moderate efficiency drag / minor fault | Review operating parameters; schedule VFD/speed trim |
| **High** | $55 \le R < 80$ | Significant mechanical stress / pound | Actively throttle pump speed or adjust pump-off cycles |
| **Critical** | $80 \le R \le 100$ | Severe impact shock / rod buckling hazard | Immediate speed reduction, CSS cycle initiation, or shutdown |

---

## 5. Assessment Summary Across 400 Synthetic Cards

### Severity Distribution by Actual Condition
| Actual Condition | Low | Medium | High | Critical | Total | Mean Risk Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 158 | 0 | 0 | 0 | **158** | **4.75 / 100** |
| **Gas Interference** | 25 | 40 | 0 | 0 | **65** | **27.90 / 100** |
| **Rod Floating** | 0 | 1 | 49 | 39 | **89** | **77.00 / 100** |
| **Fluid Pound** | 0 | 0 | 37 | 51 | **88** | **80.66 / 100** |
| **Total** | **183** | **41** | **86** | **90** | **400** | — |

---

## 6. Representative Example Assessments

### Example 1: Rod Floating (High Severity)
* **Card ID:** `491f0416` (`WELL-019`)
* **Predicted:** `Rod Floating` (Confidence: 99.8%)
* **Risk Score:** **80.5 / 100** (Tier: **Critical**)
* **Contributing Factors:** Elevated viscosity (1055.6 cP) increases rod downstroke fluid drag; Low wellbore temperature (130.7 °F); Severe mechanical impact spike on downstroke.
* **Engineering Explanation:** Severe downstroke rod deceleration diagnosed. Downstroke velocity lag caused by heavy crude drag (1056 cP) and pumping speed (6.6 SPM) creates compressive rod stress and sharp impact on plunger re-engagement.
* **Recommended Action:** IMMEDIATE ACTION: Reduce VFD stroke speed from 6.6 SPM to 5.0 SPM to restore gravity downstroke tracking. Schedule CSS (Cyclic Steam Stimulation) / diluent injection to reduce crude viscosity. Inspect rod string for compressive buckling.

### Example 2: Fluid Pound (Critical Severity)
* **Card ID:** `f6e80b85` (`WELL-005`)
* **Predicted:** `Fluid Pound` (Confidence: 100.0%)
* **Risk Score:** **87.4 / 100** (Tier: **Critical**)
* **Contributing Factors:** Depleted fluid column (1240 ft) indicates pump-off condition; Low pump submergence ratio (28.4%); Sharp fluid-surface impact shock detected (slope_z=118.4); Depressed minimum load (4210 lbf).
* **Engineering Explanation:** Incomplete pump barrel fillage diagnosed due to reservoir inflow deficit (1240 ft fluid level). Plunger free-falls through vapor pocket before abruptly impacting fluid surface on downstroke, transmitting shock waves to polished rod and gearbox.
* **Recommended Action:** IMMEDIATE ACTION: Throttle pump speed from 7.4 SPM to 5.2 SPM or activate automatic Pump-Off Controller (POC) with intermittent timer (duty cycling) to permit annular fluid recovery. Inspect valves and rod guides.

### Example 3: Gas Interference (Medium Severity)
* **Card ID:** `40eff1f0` (`WELL-007`)
* **Predicted:** `Gas Interference` (Confidence: 98.4%)
* **Risk Score:** **38.2 / 100** (Tier: **Medium**)
* **Contributing Factors:** Volumetric card area collapsed to 0.54 due to gas compression; Suppressed production rate (27.5 bpd).
* **Engineering Explanation:** Compressible free gas inside the pump barrel is delaying traveling valve opening on downstroke. Volumetric lifting capacity is degraded due to cyclic gas compression.
* **Recommended Action:** ADVISORY: Monitor casinghead gas backpressure. Optimize stroke length vs SPM to maximize compression ratio in barrel.

### Example 4: Normal Operation (Low Severity)
* **Card ID:** `33b108c1` (`WELL-003`)
* **Predicted:** `Normal` (Confidence: 99.7%)
* **Risk Score:** **3.8 / 100** (Tier: **Low**)
* **Contributing Factors:** Operating within nominal mechanical baseline.
* **Engineering Explanation:** Sucker-rod card exhibits uniform load profile and normal full-stroke work envelope. Operating parameters within nominal thresholds.
* **Recommended Action:** Maintain baseline pumping parameters. Continue regular SCADA monitoring and scheduled lubrication.

---

## 7. Deliverables & Output Files

- **Core Engine Module:** [`data_pipeline/risk_engine.py`](file:///c:/Users/rajva/Downloads/TEJAS/data_pipeline/risk_engine.py)
- **Evaluation Script:** [`evaluate_risk_engine.py`](file:///c:/Users/rajva/Downloads/TEJAS/evaluate_risk_engine.py)
- **Full Assessment Results (400 cards):** [`results/phase5/risk_assessment_results.csv`](file:///c:/Users/rajva/Downloads/TEJAS/results/phase5/risk_assessment_results.csv)
- **Assessment Report:** [`results/phase5/risk_assessment_report.txt`](file:///c:/Users/rajva/Downloads/TEJAS/results/phase5/risk_assessment_report.txt)
