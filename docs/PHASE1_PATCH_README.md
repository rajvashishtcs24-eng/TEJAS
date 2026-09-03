# TEJAS — Phase 1 Patch: Persistent Well Identity

## Change made (scope-limited, as requested)
`data_gen/generate_dataset.py` only. No changes to `srp_physics.py`,
`conditions.py` (condition generation/shapes untouched), schema, or
`data_source`.

- Added `sample_well_baseline(rng)`: called **once per well_id** (20 total),
  producing:
  - **Mechanical fields (persistent, no drift, ever):** `stroke_length_in`,
    `pump_depth_ft`, `rod_weight_per_ft`, `fluid_specific_gravity`,
    `plunger_diameter_in`, `friction_lbf`.
  - **Dynamic baselines (a well's "typical operating point"):** SPM,
    temperature, viscosity, fluid level, production — these are the center
    that individual cards drift around.
- Replaced `sample_well_config(condition, rng)` with
  `sample_well_config(baseline, condition, rng)`: mechanical fields are
  copied exactly from the well's baseline every time; dynamic fields =
  baseline + small Gaussian drift (normal short-term fluctuation), plus the
  same condition-informed directional excursion as before (e.g. Rod
  Floating still skews SPM/viscosity up) — now expressed as a deviation
  from *that well's own* baseline rather than an independent full-range
  redraw. This keeps the condition generators receiving physically
  appropriate parameters while giving each well a persistent identity.
- `generate_dataset()`: builds all 20 well baselines up front, then for each
  of the 400 cards picks a `well_id` and looks up its baseline before
  building that card's `WellConfig`.

Condition weights, `GENERATORS` (card-shape logic), schema fields, and
`data_source="synthetic_physics_v1"` are all unchanged.

## Regeneration
Ran `generate_dataset.py` (seed=42) → 400 cards, `dynacards_synthetic_v1.csv`
overwritten in place (same path, same schema).

New condition counts: Normal 158, Rod Floating 89, Fluid Pound 88, Gas
Interference 65 (differs slightly from the pre-patch run since the
underlying parameter sampling changed, as expected).

## Verification: persistence actually achieved
- **Mechanical fields identical within every well:** confirmed
  `stroke_length` / `pump_depth` each have exactly 1 unique value per
  `well_id` across all 20 wells (checked directly, not assumed).
- **Dynamic fields drift around a per-well mean, not fully random:** e.g.
  WELL-002 averages ~7.1 SPM / high viscosity (~640 cP) across its cards,
  while WELL-016 averages ~5.3 SPM / ~497 cP — each well now has a
  consistent, distinguishable operating fingerprint instead of every card
  being an independent draw.
- Card shapes per condition remain visually and physically correct after
  the patch (re-checked against `phase2_raw_vs_processed.png`) — the fix
  only changed *which wells* generate *what parameter values*, not the
  condition physics itself.

## Phase 2 rerun (on patched raw data)
- 400/400 cards valid, 0 invalid.
- Condition counts preserved exactly through preprocessing.
- 20 unique wells, 13–24 cards/well (mean 20).
- Train/val well split: 16 / 4 wells → 321 / 79 cards.

### Confirmed: split is now genuinely grouped by persistent well characteristics
- **Zero wells appear in both train and val** (checked directly — no
  leakage).
- Mechanical persistence (`stroke_length`, `pump_depth`) survived intact
  through the full preprocessing pipeline into `processed_metadata.csv`.
- Per-well dynamic averages are now stable and distinguishable (see table
  below) — this was the missing property before the patch, and is what
  makes the well-level split, and the magnitude-preserving (well z-score)
  normalization from Phase 2, actually meaningful now.

| well_id | split | n_cards | mean SPM | mean temp | mean viscosity |
|---|---|---|---|---|---|
| WELL-002 | train | 21 | 7.1 | 118.7 | 639.8 |
| WELL-016 | val | 24 | 5.3 | 139.3 | 496.9 |
| WELL-004 | train | 23 | 6.0 | 130.7 | 490.0 |
| WELL-020 | val | 21 | 5.4 | 124.1 | 527.1 |

(Full 20-well table in the regenerated `processed_metadata.csv`.)

## What this unblocks
- The Phase 2 magnitude-preserving (well z-score) load normalization now
  removes a *real* well-specific baseline rather than an arbitrary one.
- The well-level train/val split can now be used as evidence of
  cross-well generalization in Phase 5, since held-out wells genuinely have
  different (but internally consistent) operating characteristics from
  training wells.

## Still out of scope (unchanged from before)
- No ML/modeling — this was a data-generation and preprocessing patch only.
- No architecture changes.
- `risk_level` / `recommended_action` remain Phase 1 placeholders, not real
  targets.
- Only 20 wells / ~20 cards each — reasonable for a first prototype, but a
  4-well validation set is still coarse; worth keeping in mind before
  drawing strong generalization conclusions in Phase 5.
