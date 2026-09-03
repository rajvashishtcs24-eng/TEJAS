"""
conditions.py
-------------
Applies condition-specific distortions to the ideal base card produced by
srp_physics.base_card(). Each function encodes the QUALITATIVE signature
documented in petroleum-engineering literature for that condition - these
are shape rules, not a rigorous derivation from first principles. Treat
severity parameters as tunable knobs to produce a spread of realistic,
distinguishable synthetic examples, not as calibrated physical constants.

Conditions implemented:
  - normal
  - rod_floating     : viscous drag / high SPM prevents the rod string from
                        tracking the polished rod on the downstroke; rod
                        "floats", then re-engages with a sharp impact spike
                        near the bottom of stroke.
  - fluid_pound       : incomplete pump fillage; plunger free-falls through
                        unfilled barrel space on the downstroke (near-zero
                        net load) until it impacts the fluid surface,
                        producing a sharp mid-stroke impact spike.
  - gas_interference  : compressible gas in the barrel delays valve opening;
                        load pickup/release at top of stroke is smeared into
                        a rounded transition instead of a sharp corner, and
                        net card area (efficiency) shrinks.
"""

import numpy as np
from srp_physics import WellConfig, base_card

RNG = np.random.default_rng()


def _add_noise(load: np.ndarray, rel_std: float, rng) -> np.ndarray:
    noise = rng.normal(0, rel_std * np.std(load), size=load.shape)
    return load + noise


def generate_normal(well: WellConfig, n_points=200, rng=RNG, noise=0.01):
    theta, pos, load = base_card(well, n_points)
    load = _add_noise(load, noise, rng)
    meta = {"condition_label": "Normal", "severity": 0.0}
    return theta, pos, load, meta


def generate_rod_floating(well: WellConfig, n_points=200, rng=RNG, noise=0.015,
                           severity: float = None):
    """Downstroke: a flattened/dipped 'float' region where load barely
    changes (rod not transmitting load), followed by a sharp impact spike
    just before bottom-of-stroke when the rod catches up to the polished rod.
    Severity scales with SPM and viscosity (both increase the chance the
    rod string cannot free-fall fast enough)."""
    theta, pos, load = base_card(well, n_points)

    if severity is None:
        # crude severity heuristic: higher SPM + higher viscosity -> worse float
        severity = np.clip((well.spm - 4) / 6.0 + (well.viscosity_cp - 200) / 1200.0, 0.15, 1.0)

    down_mask = theta > np.pi
    down_theta = theta[down_mask]

    # float region: middle 55% of the downstroke, load pinned near a low plateau
    float_start, float_end = np.pi + 0.25 * np.pi, np.pi + 0.80 * np.pi
    float_zone = (down_theta >= float_start) & (down_theta <= float_end)

    plateau_level = well.rod_weight_buoyant_lbf - well.friction_lbf * (0.3 + 0.5 * severity)
    down_load = load[down_mask].copy()
    down_load[float_zone] = plateau_level + rng.normal(0, 5, size=float_zone.sum())

    # impact spike right after the float zone ends (rod catches up)
    spike_center = float_end + 0.05
    spike_width = 0.06
    spike_height = well.inertia_amplitude_lbf * (1.2 + 2.5 * severity)
    spike = spike_height * np.exp(-((down_theta - spike_center) ** 2) / (2 * spike_width ** 2))
    down_load += spike

    load[down_mask] = down_load
    load = _add_noise(load, noise, rng)
    meta = {"condition_label": "Rod Floating", "severity": float(severity)}
    return theta, pos, load, meta


def generate_fluid_pound(well: WellConfig, n_points=200, rng=RNG, noise=0.015,
                          fillage_fraction: float = None):
    """Incomplete pump fillage. On the downstroke, load stays near buoyant
    rod weight only (near free-fall, minimal fluid/inertia contribution)
    until the plunger reaches the fluid level inside the barrel at
    `fillage_fraction` of the stroke, then a sharp impact spike occurs."""
    theta, pos, load = base_card(well, n_points)

    if fillage_fraction is None:
        fillage_fraction = rng.uniform(0.35, 0.7)  # lower = worse pound

    down_mask = theta > np.pi
    down_theta = theta[down_mask]
    down_pos = pos[down_mask]

    # position at which impact occurs (measured from top of stroke downward)
    impact_pos = well.stroke_length_in * (1 - fillage_fraction)
    pre_impact = down_pos > impact_pos  # still falling before hitting fluid

    down_load = load[down_mask].copy()
    down_load[pre_impact] = well.rod_weight_buoyant_lbf * 0.55 + rng.normal(
        0, 6, size=pre_impact.sum()
    )

    impact_idx = np.argmin(np.abs(down_pos - impact_pos))
    spike_height = well.inertia_amplitude_lbf * 2.0 + well.friction_lbf
    spike = spike_height * np.exp(-((np.arange(len(down_theta)) - impact_idx) ** 2) / (2 * 2.5 ** 2))
    down_load += spike

    load[down_mask] = down_load
    load = _add_noise(load, noise, rng)
    meta = {"condition_label": "Fluid Pound", "severity": float(1 - fillage_fraction)}
    return theta, pos, load, meta


def generate_gas_interference(well: WellConfig, n_points=200, rng=RNG, noise=0.012,
                               gas_fraction: float = None):
    """Compressible gas delays valve opening/closing: load pickup at the
    bottom of the upstroke and load release at the top of the downstroke
    become smeared/rounded transitions instead of sharp corners, and the
    effective card area (net stroke / pump efficiency) shrinks."""
    theta, pos, load = base_card(well, n_points)

    if gas_fraction is None:
        gas_fraction = rng.uniform(0.15, 0.45)

    # smooth (sigmoid) blending window widens with gas_fraction, applied at
    # both stroke-reversal corners (theta=0/2pi and theta=pi)
    width = 0.3 + 1.8 * gas_fraction

    def smooth_corner(center):
        return 1.0 / (1.0 + np.exp(-(theta - center) / width * 4))

    # reduce the sharpness of the load rise at bottom (theta ~ 0) and
    # the load drop at top (theta ~ pi) by blending toward mid-load
    mid_load = well.rod_weight_buoyant_lbf + 0.5 * well.fluid_load_lbf
    blend_bottom = np.exp(-((theta - 0) ** 2) / (2 * width ** 2)) + np.exp(
        -((theta - 2 * np.pi) ** 2) / (2 * width ** 2)
    )
    blend_top = np.exp(-((theta - np.pi) ** 2) / (2 * width ** 2))

    load = load * (1 - gas_fraction * 0.5 * (blend_bottom + blend_top)) + (
        mid_load * gas_fraction * 0.5 * (blend_bottom + blend_top)
    )

    # net card area shrinkage: compress load range slightly toward mean
    mean_load = np.mean(load)
    load = mean_load + (load - mean_load) * (1 - 0.25 * gas_fraction)

    load = _add_noise(load, noise, rng)
    meta = {"condition_label": "Gas Interference", "severity": float(gas_fraction)}
    return theta, pos, load, meta


GENERATORS = {
    "Normal": generate_normal,
    "Rod Floating": generate_rod_floating,
    "Fluid Pound": generate_fluid_pound,
    "Gas Interference": generate_gas_interference,
}
