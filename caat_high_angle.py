"""High-impact-angle GRB helper based on CAAT model."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi, sqrt
from typing import Dict


@dataclass
class Env:
    """Environment defaults."""

    rho: float = 1.225  # air density [kg/m^3] (ISA sea level)
    g: float = 9.81  # gravity [m/s^2]


@dataclass
class Human:
    """Human body representation."""

    r_person: float = 0.30  # person radius [m]


# Annex F reference points: (Characteristic dimension w [m], Frontal area A [m^2])
ANNEXF_POINTS = [
    (1.0, 0.1),
    (3.0, 0.5),
    (8.0, 2.5),
    (20.0, 12.5),
    (40.0, 25.0),
]


def frontal_area_from_char_dim(w_m: float) -> float:
    """Linear interpolate/extrapolate Annex-F frontal area A for characteristic dimension w."""

    pts = ANNEXF_POINTS
    if w_m <= pts[0][0]:
        (x0, y0), (x1, y1) = pts[0], pts[1]
        return y0 + (y1 - y0) * (w_m - x0) / (x1 - x0)
    if w_m >= pts[-1][0]:
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y0 + (y1 - y0) * (w_m - x0) / (x1 - x0)
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x0 <= w_m <= x1:
            return y0 + (y1 - y0) * (w_m - x0) / (x1 - x0)
    raise ValueError("Interpolation failed for characteristic dimension w.")


def terminal_velocity(m: float, rho: float, CD: float, A: float, g: float) -> float:
    """Compute terminal velocity from drag balance."""

    cd_a = CD * A
    if cd_a <= 0:
        raise ValueError("C_D * A must be > 0.")
    return sqrt((2 * m * g) / (rho * cd_a))


def safety_factor_from_energy(E_tot_J: float) -> float:
    """Piecewise CAAT safety factor."""

    E_kJ = E_tot_J / 1000.0
    if E_kJ < 12.0:
        return 2.3
    if E_kJ <= 3125.0:
        return 1.4 * (E_kJ ** 0.2)
    return 7.0


def effective_rD(w: float, r_person: float) -> float:
    """Effective 'person + UA' radius."""

    return r_person + 0.5 * w


@dataclass
class Inputs:
    """Inputs required for the high-impact-angle computation."""

    m: float  # mass [kg]
    w: float  # characteristic dimension [m] (tip-to-tip)
    CD: float  # drag coefficient [-]
    A_override: float | None = None  # optional frontal area override [m^2]
    rho: float | None = None  # optional air density override [kg/m^3]
    g: float | None = None  # optional gravity override [m/s^2]
    r_person: float | None = None  # optional person radius [m]
    Fs_override: float | None = None  # optional safety factor override


def compute_critical_area_high_angle(
    inp: Inputs, env: Env = Env(), human: Human = Human()
) -> Dict[str, float | str]:
    """Compute High-Impact-Angle critical area AC and related terms."""

    rho = env.rho if inp.rho is None else inp.rho
    g = env.g if inp.g is None else inp.g
    r_person = human.r_person if inp.r_person is None else inp.r_person

    A = inp.A_override if inp.A_override is not None else frontal_area_from_char_dim(inp.w)
    Vt = terminal_velocity(inp.m, rho, inp.CD, A, g)
    E_tot = 0.5 * inp.m * Vt * Vt
    Fs = inp.Fs_override if inp.Fs_override is not None else safety_factor_from_energy(E_tot)

    rD = effective_rD(inp.w, r_person)
    AC = Fs * pi * (rD ** 2)
    r_equiv = sqrt(AC / pi)

    return {
        "model": "HIGH_ANGLE",
        "A_frontal_m2": A,
        "CD_used": inp.CD,
        "rho_kg_m3": rho,
        "V_terminal_mps": Vt,
        "E_terminal_J": E_tot,
        "Fs_used": Fs,
        "rD_m": rD,
        "AC_m2": AC,
        "AC_equiv_radius_m": r_equiv,
    }


__all__ = [
    "Env",
    "Human",
    "Inputs",
    "compute_critical_area_high_angle",
]