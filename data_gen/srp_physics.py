"""
srp_physics.py
--------------
Simplified QUASI-STATIC physics model for generating synthetic surface
dynamometer cards (polished-rod Load vs Position over one pumping cycle).

IMPORTANT / HONESTY NOTE:
This is NOT a full transient sucker-rod dynamics model. A production-grade
model would solve the damped wave equation along the rod string (Gibbs
model), including rod elasticity, distributed inertia, and travelling
stress waves. This module uses a lumped, single-degree-of-freedom
quasi-static approximation:

    Load(theta) = buoyant_rod_weight + fluid_load(theta) + inertia(theta)
                  +/- friction

...with condition-specific modifications layered on top to reproduce the
QUALITATIVE, well-documented card signatures for each fault type (see
conditions.py). It is good enough to produce visually and numerically
distinct card shapes per condition for a first prototype, but every card
generated here is SYNTHETIC and must be labeled as such downstream -
never presented or stored as if it were field-measured data.

Units: inches (position), lbf (load), seconds (time), deg F, cP, ft.
"""

from dataclasses import dataclass
import numpy as np

G_FTPS2 = 32.174  # standard gravity, ft/s^2
STEEL_SG = 7.85    # specific gravity of steel rods


@dataclass
class WellConfig:
    """Physical + operating parameters for one synthetic well/stroke."""
    stroke_length_in: float = 120.0        # polished rod stroke, in
    spm: float = 6.0                       # strokes per minute
    pump_depth_ft: float = 4500.0          # rod string / pump setting depth
    rod_weight_per_ft: float = 2.16        # lb/ft, e.g. API 76 grade steel rod
    fluid_specific_gravity: float = 0.92   # heavy crude/water mix
    plunger_diameter_in: float = 2.0
    friction_lbf: float = 400.0            # lumped rod/tubing + pump friction
    fluid_level_ft: float = 3800.0         # fluid column height above pump
    temperature_F: float = 140.0
    viscosity_cp: float = 400.0            # oil viscosity at well temp
    production_bpd: float = 55.0

    @property
    def buoyancy_factor(self) -> float:
        return 1.0 - (self.fluid_specific_gravity / STEEL_SG)

    @property
    def rod_weight_air_lbf(self) -> float:
        return self.rod_weight_per_ft * self.pump_depth_ft

    @property
    def rod_weight_buoyant_lbf(self) -> float:
        return self.rod_weight_air_lbf * self.buoyancy_factor

    @property
    def plunger_area_in2(self) -> float:
        return np.pi / 4.0 * self.plunger_diameter_in ** 2

    @property
    def fluid_load_lbf(self) -> float:
        # F_o: static fluid column load carried by the plunger on the upstroke
        gradient_psi_per_ft = 0.433 * self.fluid_specific_gravity
        return gradient_psi_per_ft * self.fluid_level_ft * self.plunger_area_in2

    @property
    def omega_rad_s(self) -> float:
        return self.spm * 2.0 * np.pi / 60.0

    @property
    def inertia_amplitude_lbf(self) -> float:
        # (mass) * (S/2) * omega^2, mass = W_air / g  (lb-force <-> lb-mass at 1g)
        stroke_ft = self.stroke_length_in / 12.0
        mass_lbm = self.rod_weight_air_lbf / G_FTPS2
        return mass_lbm * (stroke_ft / 2.0) * self.omega_rad_s ** 2


def stroke_position(theta: np.ndarray, well: WellConfig) -> np.ndarray:
    """Polished-rod position (in), theta=0 at bottom dead center, theta=pi at top.
    SHM approximation of the crank/walking-beam kinematics."""
    return (well.stroke_length_in / 2.0) * (1.0 - np.cos(theta))


def inertia_term(theta: np.ndarray, well: WellConfig) -> np.ndarray:
    """Quasi-static inertial load contribution, +ve near bottom, -ve near top."""
    return well.inertia_amplitude_lbf * np.cos(theta)


def base_card(well: WellConfig, n_points: int = 200):
    """Ideal 'textbook normal' card before any condition-specific distortion.
    Returns (theta, position, load_up, load_down) arrays.
    theta in [0, 2*pi); 0..pi = upstroke, pi..2*pi = downstroke.
    """
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pos = stroke_position(theta, well)
    inertia = inertia_term(theta, well)

    load = np.empty_like(theta)
    up_mask = theta <= np.pi

    load[up_mask] = (
        well.rod_weight_buoyant_lbf
        + well.fluid_load_lbf
        + inertia[up_mask]
        + well.friction_lbf
    )
    load[~up_mask] = (
        well.rod_weight_buoyant_lbf
        + inertia[~up_mask]
        - well.friction_lbf
    )
    return theta, pos, load
