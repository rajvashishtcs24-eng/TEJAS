# TEJAS — Phase 2: Preprocessing Pipeline

## Status
Preprocessing pipeline built and QA'd against the Phase 1 synthetic dataset
(`dynacards_synthetic_v1.csv`, `data_source=synthetic_physics_v1`, preserved
unchanged throughout). **No model was trained in this phase.**

## Corrections made before implementation (flagged and resolved)
1. **Position is not a valid single-value resampling axis.** A dynamometer
   card is a closed loop — each position value maps to two load values (one
   on the upstroke, one on the downstroke). Resolved by splitting every card
   at its position peak into an upstroke branch and a downstroke branch,
   normalizing position by stroke length within each branch (0→1), and
   resampling each branch independently onto 100 evenly spaced points
   (100 + 100 = 200 total). No crank angle is used anywhere, satisfying the
   "no 0→2π" instruction while keeping the resampling well-defined. This is
   also the conventional way dynamometer cards are analyzed (separate
   upstroke/downstroke curves), not just a workaround.
2. **`well_id` is not currently bound to consistent physical parameters.**
   Confirmed in Phase 1's `generate_dataset.py`: each card gets an
   independently randomized `WellConfig`, and `well_id` is assigned
   afterward with no relationship to those parameters. A well-level
   train/val split was still produced (20 unique wells is a reasonable
   count, and it does prevent literal duplicate-card leakage), but it is
   **not yet a meaningful domain-generalization split** — see Limitations.

## What was built
- `data_pipeline/preprocess.py`
  - Loads the raw CSV, parses `position[]`/`load[]` JSON per card.
  - Validates every card (length match, min point count, finite values,
    valid stroke length, non-degenerate position range, correct
    `data_source` tag). Invalid cards are dropped and logged with a reason,
    never silently discarded.
  - Branch-splits and resamples every card onto a normalized 0→1 position
    grid (200 points: 100 upstroke + 100 downstroke), per the correction
    above.
  - Produces **two parallel load representations**, kept as separate output
    files (neither is treated as "the" chosen one — that comparison is
    deferred to Phase 4/5):
    - **Shape-normalized**: per-card min-max → `[0, 1]`. Pure shape, no
      magnitude information. Best for CNN/shape-based ML.
    - **Magnitude-preserving**: well-level z-score (`(load - well_mean) /
      well_std`, stats computed per `well_id` across all its cards).
      Retains relative load magnitude (e.g. PPRL/MPRL separation between
      conditions), intended for feature extraction / risk modeling.
  - Preserves the original raw `position`/`load` JSON columns untouched in
    the metadata (never overwritten).
  - Reports a well-level 80/20 train/val split (by `well_id`, not by card).

## Output files
- `data/processed/processed_cards_shape.npy` — shape `[400, 200, 2]`,
  channels `[normalized_position, shape_normalized_load]`
- `data/processed/processed_cards_magnitude.npy` — shape `[400, 200, 2]`,
  channels `[normalized_position, magnitude_normalized_load]`
- `data/processed/processed_metadata.csv` — one row per valid card: all
  original context fields (`SPM`, `stroke_length`, `temperature`,
  `viscosity`, `fluid_level`, `pump_depth`, `production_rate`,
  `condition_label`, `severity`, `data_source`), raw `position`/`load` JSON
  preserved, plus `n_raw_points`, `branch_split_index`, and `split`
  (train/val).

Array index `i` in both `.npy` files corresponds to row `i` of
`processed_metadata.csv` — they must be loaded together, in order.

Within each card's 200 points: **indices 0–99 = upstroke, indices 100–199 =
downstroke.** Position runs 0→1 for the upstroke and 1→0 for the downstroke
(preserving the natural closed-loop trace direction).

## QA results
- **400/400 cards valid** (0 invalid — expected, since this is clean
  synthetic data; the validation logic is now in place and tested for when
  real/noisier data arrives).
- Raw point counts: min=200, max=200 (uniform, as generated).
- Condition counts preserved exactly through preprocessing: Normal 167,
  Rod Floating 90, Fluid Pound 73, Gas Interference 70.
- Position range after normalization: exactly `[0.0, 1.0]` — correct.
- Load ranges: shape-normalized confirmed pinned to `[0, 1]` per card;
  magnitude-preserving (z-score) confirmed to vary meaningfully across
  conditions (e.g. Rod Floating / Fluid Pound show higher peak z-scores
  than Normal / Gas Interference, consistent with their impact-spike
  physics) — spot-checked numerically, not just visually.
- 20 unique wells, 12–28 cards per well (mean 20).
- Train/val well split: 16 / 4 wells → 321 / 79 cards. Condition
  distribution roughly preserved in both splits (see
  `processed_metadata.csv`, `split` column), though with only 4 validation
  wells the split is coarse.

### QA plots
- `qa_plots/phase2_raw_vs_processed.png` — raw card vs. branch-split
  resampled card, one example per condition. Verified the impact-spike
  location and pre/post-impact plateaus land in the physically correct
  place after resampling (checked numerically, not just by eye).
- `qa_plots/phase2_shape_vs_magnitude.png` — same cards in both load
  representations side by side.
- `qa_plots/phase2_random_samples.png` — 8 randomly selected processed
  cards across wells/conditions.

## Explicit limitations (do not overclaim these)
1. **Well-level split is provisional, not yet meaningful for
   generalization claims.** Because `well_id` isn't bound to consistent
   physical parameters in Phase 1's generator, splitting by well doesn't
   yet isolate "well-specific" behavior the way a real domain split should.
   **Recommended next step:** patch `generate_dataset.py` so each `well_id`
   gets one fixed baseline `WellConfig`, with multiple cards per well
   generated over time via small drift/noise around that baseline — then
   this split (and the well-level z-score normalization) will actually mean
   something. Not done in Phase 2, since it's a Phase 1 change; flagging
   for your decision on when to circle back.
2. **Magnitude-preserving normalization inherits the same caveat** — the
   per-well z-score stats are computed correctly, but until well_id is
   physically meaningful, they don't yet achieve genuine well-specific
   baseline removal. The mechanism is correctly implemented and ready to
   become meaningful once Phase 1 is patched.
3. Branch-splitting via `argmax` on position assumes a clean single peak.
   Synthetic position arrays are noise-free by construction, so this is
   exact here. Real sensor data may have local noise near the peak that
   could require smoothing before peak-finding — not an issue yet, but
   worth remembering for Phase 3+ once real data is involved.
4. `risk_level` / `recommended_action` are carried through in metadata
   for reference only — **not used as ML labels anywhere in this
   pipeline**, and should not be used as targets downstream either (per
   your instruction — they're still Phase 1's placeholder heuristic, not
   the real Phase 8/9 risk engine).

## What Phase 3/4 should consume
- Load `processed_cards_shape.npy` / `processed_cards_magnitude.npy`
  alongside `processed_metadata.csv` (matched by row index).
- Use `condition_label` from metadata as the classification target for
  Phase 5 (not `risk_level`/`recommended_action`).
- Use the `split` column for train/val partitioning — but see Limitation 1
  before treating validation results as evidence of cross-well
  generalization.
- Phase 4 (feature extraction: PPRL, MPRL, load range, card area, slopes,
  shape descriptors) should be computed from the **magnitude-preserving**
  array where magnitude matters (PPRL/MPRL/load range), and can use either
  array for pure shape descriptors — Phase 4/5 should empirically compare
  which representation classifies better, per your instruction not to
  choose between them prematurely.
