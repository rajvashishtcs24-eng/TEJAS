# TEJAS — Phase 1: Dynamometer Card Data

## Status
No real dynamometer dataset was available (SIH organizers provided none; every
published real-field dataset — Bahrain SPE-194949, Shenyang/China, Mossoró/Brazil
— is proprietary SCADA data, not publicly released). Phase 1 therefore uses a
**physics-based synthetic generator**, which is a documented, defensible approach
used by several of the papers above when field data isn't available.

**This is simulated data, not field data — it is labeled `data_source:
synthetic_physics_v1` in every row and must stay labeled that way through every
downstream phase.** If real field data becomes available later (e.g. from a
mentor, PSU partner, or public release), it should replace this, and the
generator becomes a data-augmentation / edge-case tool rather than the primary
source.

## What was built
- `data_gen/srp_physics.py` — quasi-static single-DOF force model (buoyant rod
  weight + fluid load + SHM-approximated inertia + friction). Explicitly *not*
  a full Gibbs wave-equation model (per your call: simplified quasi-static was
  the right tradeoff for this stage).
- `data_gen/conditions.py` — condition-specific shape distortions layered on
  the base card, following the qualitative signatures documented in SRP
  diagnostic literature:
  - **Normal** — smooth parallelogram card.
  - **Rod Floating** — flattened/dipped load region mid-downstroke (rod not
    transmitting load) + a sharp impact spike where it re-engages. Severity
    scales with SPM and viscosity, matching the project's stated mechanism
    ("heavy/high-viscosity crude can contribute to rod/downstroke mismatch...
    resulting in impact loading").
  - **Fluid Pound** — near free-fall load on the downstroke until the plunger
    hits the fluid surface inside the barrel (incomplete fillage), then a
    sharp impact spike at the fillage transition point.
  - **Gas Interference** — smoothed/rounded corners at stroke reversal (delayed
    valve opening from compressible gas) and reduced net card area.
- `data_gen/generate_dataset.py` — batch generator producing 400 cards against
  the agreed schema (`card_id, well_id, timestamp, position[], load[], SPM,
  stroke_length, temperature, viscosity, fluid_level, pump_depth,
  production_rate, condition_label, risk_level, recommended_action` + severity
  + data_source).
- `data_gen/visualize.py` — QA plot confirming shape distinctness.

## QA results
- Visual: all 4 conditions produce visually distinct card shapes (see
  `qa_plots/condition_shapes_qa.png`).
- Numeric sanity check (PPRL/MPRL/load range by condition):

  | Condition | PPRL | MPRL | Load Range |
  |---|---|---|---|
  | Normal | 13,876 | 7,489 | 6,387 |
  | Rod Floating | 15,945 | 7,065 | 8,880 |
  | Fluid Pound | 12,515 | 4,642 | 7,873 |
  | Gas Interference | 12,883 | 7,637 | 5,246 |

  Rod Floating shows the highest peak load (impact spike), Fluid Pound the
  lowest minimum load (free-fall dip), Gas Interference the narrowest range
  (efficiency loss) — all consistent with the underlying physics, and
  separable even before real feature engineering (Phase 4-5).

## Explicit limitations (do not overclaim these)
1. Quasi-static, not transient — no rod-string wave propagation, elasticity,
   or distributed mass. Real cards will show additional wave-reflection
   ripple that this model does not capture.
2. Condition severity thresholds (e.g. "SPM > 7 -> rod floating more likely")
   are heuristic knobs tuned for visual/numeric separability, not calibrated
   against real well data.
3. `risk_level` / `recommended_action` in this dataset are a **placeholder
   heuristic**, not the Phase 8/9 risk/decision engine. They exist only so
   the schema is populated end-to-end; revisit once the real risk engine
   exists.
4. Class balance (Normal 42%, Rod Floating 22%, Fluid Pound 18%, Gas
   Interference 18%) is arbitrary, not sampled from a real field distribution.

## Files delivered
- `data/synthetic/dynacards_synthetic_v1.csv` — 400 synthetic cards
- `qa_plots/condition_shapes_qa.png` — visual QA
- `data_gen/*.py` — generator source (re-runnable, seeded for reproducibility)

## Suggested next step (Phase 2)
Build the preprocessing pipeline: load this CSV, parse `position`/`load` JSON
columns into numpy arrays, resample all cards to a common number of points,
and normalize load/position — then move to Phase 3 (visualize real vs.
synthetic side by side, once/if real data arrives) and Phase 4 (feature
extraction: PPRL, MPRL, load range, card area, slopes, shape descriptors).
