# TEJAS — Phase 4: Dynamometer-Card Feature Extraction

## Status
34 interpretable features extracted for all 400 cards from
`processed_cards_shape.npy` / `processed_cards_magnitude.npy` /
`processed_metadata.csv`. **No ML model trained.** Separability was checked
with one-way ANOVA (a descriptive statistical test — it does not fit or
predict anything, so it doesn't cross into modeling).

`condition_label` is the only classification target used anywhere.
`risk_level` / `recommended_action` are carried through as reference
columns only, per instruction. `data_source=synthetic_physics_v1` preserved
on every row.

## Modifications made to the literal spec (flagged, not silent)
1. **Card area/work needs real physical units, which neither processed
   array alone provides.** Both are dimensionless (normalized position ×
   normalized/z-scored load). Computed **two** area features instead of
   one: `card_work_raw_lbf_in` (true work per stroke, shoelace integral
   over the **raw preserved** position/load in real inches/lbf) and
   `card_area_shape_norm` (dimensionless loop area from the shape-normalized
   array, a pure ML shape feature). Neither alone is what "card area/work"
   conventionally means — this labels both correctly instead of picking one.
2. **PPRL/MPRL/range/mean/std computed twice** — once from the magnitude
   (well z-score) representation as you specified, and once from the
   **raw preserved load** in true lbf (`*_raw_lbf` suffix), since a real
   engineering question like "does this exceed rated rod load" needs actual
   lbf, not a per-well z-score.
3. **Slope statistics report both mean and peak |slope|** per branch, not
   just mean. A brief impact spike (the whole diagnostic signal for rod
   floating/fluid pound) gets averaged away in a mean-slope-only metric.
   `max_abs_slope_down_z` turned out to rank 5th of 34 by ANOVA F-statistic
   — this addition mattered.
4. **Two positional shape descriptors added**: `position_of_peak_load`,
   `position_of_min_load` — *where* along the normalized stroke (0→1) the
   extreme load occurs. This is diagnostic on its own (rod floating spikes
   near the bottom of stroke, fluid pound spikes mid-stroke, gas
   interference has no sharp spike at all) and costs nothing extra.

## Full feature documentation

### Context-derived (carried from metadata, not computed from the card)
| Feature | Units | Why useful for SRP diagnosis |
|---|---|---|
| `SPM` | strokes/min | Operating speed; directly drives inertial loading and rod-floating risk |
| `temperature` | °F | Reservoir/wellbore temperature; drives viscosity via the thermal coupling |
| `viscosity` | cP | Heavy-crude viscosity; primary driver of rod-floating and pump drag |
| `fluid_level` | ft | Fluid column height above pump; relates to fillage and fluid-pound risk |
| `pump_depth` | ft | Rod string / pump setting depth; scales rod weight and inertia |
| `production_rate` | bbl/day | Well output; context for overall well health |
| `stroke_length` | in | Polished-rod stroke length; scales position and inertia |

### Card-derived: headline load statistics (magnitude z-score)
| Feature | Units | Definition | Why useful |
|---|---|---|---|
| `PPRL_z` | z-score (well-normalized) | max load over full cycle | Peak polished-rod load; classic SRP diagnostic quantity, cross-well comparable |
| `MPRL_z` | z-score | min load over full cycle | Minimum load; low values flag free-fall/fluid-pound behavior |
| `load_range_z` | z-score | PPRL_z − MPRL_z | Overall loading swing; elevated for impact-spike conditions |
| `mean_load_z` | z-score | mean of full load array | Central tendency; ranked #1 discriminator by ANOVA (F=953) |
| `std_load_z` | z-score | std of full load array | Overall load variability |

### Card-derived: headline load statistics (raw, true lbf)
| Feature | Units | Definition | Why useful |
|---|---|---|---|
| `PPRL_raw_lbf` | lbf | max of raw preserved load | Real peak load — needed for e.g. rod-string load-capacity reasoning |
| `MPRL_raw_lbf` | lbf | min of raw preserved load | Real minimum load |
| `load_range_raw_lbf` | lbf | PPRL_raw − MPRL_raw | Real-unit loading swing |
| `mean_load_raw_lbf` | lbf | mean of raw load | Real-unit central tendency |
| `std_load_raw_lbf` | lbf | std of raw load | Real-unit variability |

### Card-derived: area / work
| Feature | Units | Definition | Why useful |
|---|---|---|---|
| `card_work_raw_lbf_in` | lbf·in | Shoelace (polygon) area of the raw closed loop | Proportional to actual polished-rod work per stroke — ties directly to the SIH steam/energy-optimization goal |
| `card_area_shape_norm` | dimensionless | Shoelace area of the shape-normalized [0,1]×[0,1] loop | Pure shape "fullness" descriptor, scale-independent; low values flag distorted/inefficient cards |

### Card-derived: upstroke / downstroke statistics (magnitude z-score)
| Feature | Units | Definition | Why useful |
|---|---|---|---|
| `mean_up_z`, `std_up_z`, `min_up_z`, `max_up_z` | z-score | Stats over indices 0–99 (upstroke) | Isolates upstroke-specific abnormalities (e.g. gas-interference pickup smearing) |
| `mean_down_z`, `std_down_z`, `min_down_z`, `max_down_z` | z-score | Stats over indices 100–199 (downstroke) | Isolates downstroke-specific abnormalities (rod float, fluid pound both manifest here) |

### Card-derived: slope statistics (magnitude z-score, per unit normalized position)
| Feature | Units | Definition | Why useful |
|---|---|---|---|
| `mean_slope_up_z` | z-score/unit pos | Mean dLoad/dPosition, upstroke | Overall upstroke loading trend |
| `max_abs_slope_up_z` | z-score/unit pos | Peak absolute slope, upstroke | Flags any sharp upstroke transition |
| `mean_slope_down_z` | z-score/unit pos | Mean dLoad/dPosition, downstroke | Overall downstroke trend |
| `max_abs_slope_down_z` | z-score/unit pos | Peak absolute slope, downstroke | **Strongest structural feature for spike detection** — captures the sharp rod-float/fluid-pound impact that a mean would wash out. Ranked #5 by ANOVA. |

### Card-derived: geometric / shape descriptors
| Feature | Units | Definition | Why useful |
|---|---|---|---|
| `position_of_peak_load` | normalized position [0,1] | Where PPRL occurs along the cycle | Spike *location* is diagnostic — e.g. rod floating spikes near the bottom of stroke |
| `position_of_min_load` | normalized position [0,1] | Where MPRL occurs | Ranked #16 by ANOVA — meaningfully separates conditions |
| `up_down_load_ratio_raw` | dimensionless | mean(raw upstroke load) / mean(raw downstroke load) | Captures overall upstroke-vs-downstroke load asymmetry in real (not z-scored) terms |

## QA results

**NaN / infinite:** none found in any of the 34 features.

**Constant / near-constant features:** none.

**Duplicate columns:** `MPRL_z` and `min_down_z` are exactly identical
(r=1.0) — expected, since the minimum load over the full cycle always
falls on the downstroke branch for every condition in this dataset. Kept
both (not removed), per instruction, but flagged here since a modeler
should know they're redundant.

**Range sanity checks:** all passed — PPRL > MPRL always, load ranges and
card work always positive, positional features bounded in [0,1], stroke
length and SPM within physically sane bounds. `up_down_load_ratio_raw`
ranged [1.16, 2.29] with 0 extreme outliers (<0.5 or >5).

**Strong correlations (|r| > 0.9, reported only, nothing removed):**
9 pairs found, e.g. `MPRL_z`↔`min_down_z` (r=1.0, exact duplicate above),
`std_down_z`↔`mean_slope_down_z` (r=-0.985), `min_up_z`↔`mean_slope_up_z`
(r=0.937). Full list and matrix in `feature_correlation_matrix.csv`. These
are largely expected — several features are different views of the same
underlying downstroke-distortion signal — and are left as-is for Phase 5
to handle (e.g. via feature selection or regularization) rather than
removed here.

**Separability (one-way ANOVA F-statistic across the 4 conditions,
descriptive statistic — not a trained classifier):**
**30 of 34 features show statistically significant differences (p<0.01)**
across conditions. Top discriminators:

| Rank | Feature | F-statistic | p-value |
|---|---|---|---|
| 1 | `mean_load_z` | 953.2 | 1.0e-180 |
| 2 | `MPRL_z` | 509.9 | 1.4e-135 |
| 3 | `min_down_z` | 509.9 | 1.4e-135 |
| 4 | `mean_down_z` | 390.7 | 6.3e-118 |
| 5 | `max_abs_slope_down_z` | 366.0 | 9.1e-114 |
| 6 | `max_up_z` | 356.1 | 4.9e-112 |
| 7 | `viscosity` | 289.3 | 2.1e-99 |
| 8 | `std_down_z` | 273.3 | 4.5e-96 |

The 4 non-significant features were `production_rate`, `pump_depth`,
`stroke_length`, and `card_work_raw_lbf_in` (p=0.025, borderline) — all
either pure well-mechanical/production context or a feature whose signal
is currently diluted by absolute well-to-well scale differences (raw work
in lbf·in isn't normalized by well size, unlike the z-score features).

**Conclusion: yes, the feature set appears clearly capable of separating
the four synthetic conditions.** The boxplots
(`phase4_top_features_boxplots.png`) show visually clean separation on the
top features — e.g. `max_abs_slope_down_z` is near-zero for Normal/Gas
Interference and elevated (with wide spread reflecting severity) for Rod
Floating/Fluid Pound, exactly matching the intended physics.

## Output files
- `data/features/dynacard_features.csv` — 400 rows × 34 features +
  identifiers (`card_id`, `well_id`, `condition_label`, `risk_level`,
  `recommended_action`, `severity`, `split`, `data_source`)
- `data/features/feature_correlation_matrix.csv`
- `data/features/feature_separability_anova.csv`
- `data/features/PHASE4_QA_REPORT.txt` — full console QA output
- `qa_plots/phase4_top_features_boxplots.png`
- `qa_plots/phase4_correlation_heatmap.png`
- `qa_plots/phase4_scatter_separability.png`

## Limitations (carried forward, still true)
- This is synthetic physics-based data (`data_source=synthetic_physics_v1`)
  — strong separability here reflects that the generator was designed with
  distinct condition signatures, not a guarantee of real-field performance.
- `card_work_raw_lbf_in` not being a strong discriminator on its own
  (p=0.025) is worth remembering in Phase 5 — it may need to be normalized
  by well size/stroke length to become useful, or may simply add little
  beyond what `PPRL_raw_lbf`/`load_range_raw_lbf` already capture.
- The 9 flagged strong-correlation pairs mean effective dimensionality is
  lower than 34 — worth considering in Phase 5 (e.g. dropping
  `min_down_z` as a literal duplicate of `MPRL_z`) but deliberately left
  untouched here per instruction.

## What Phase 5 should consume
- `data/features/dynacard_features.csv`, features = the 34 columns listed
  above (or `FEATURE_COLS` in `feature_extraction.py`).
- Target: `condition_label` only.
- Split: use the existing `split` column (train/val, grouped by well —
  now genuinely meaningful per the Phase 1.1 patch).
- Compare feature-based ML (Random Forest / XGBoost) against a CNN on the
  raw card arrays, per your original ML-strategy instruction — this
  feature set is what the feature-based path should train on.
