# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import cmath
from typing import NamedTuple, Sequence


class LorentzPole(NamedTuple):
    """Meep-convention Lorentzian: chi(f) = sigma * fn^2 / (fn^2 - f^2 - i*f*gamma)."""

    fn: float
    gamma: float
    sigma: float


def analytic_eps(eps_inf: float, poles: Sequence[LorentzPole], f: float) -> complex:
    # pole = (fn, gamma, sigma) or (fn, gamma, sigma, fc): coupling frequency
    # fc defaults to fn (Lorentz); fn=0, fc=fp is Drude -- the same 4-tuple
    # convention as coeffs._pole_recurrence.
    eps = complex(eps_inf)
    for p in poles:
        fc = p[3] if len(p) > 3 else p[0]
        eps += p[2] * fc**2 / (p[0]**2 - f**2 - 1j * f * p[1])
    return eps


class Material(NamedTuple):
    eps_inf: float
    poles: tuple
    valid_fmin: float
    valid_fmax: float

    def n(self, f: float) -> complex:
        return cmath.sqrt(analytic_eps(self.eps_inf, self.poles, f))


# Ported from meep python/materials.py (um_scale = 1, meep natural units).
# cSi: A. Deinega et al., JOSA A 28, 770 (2011). Valid 0.4-1.0 um.
cSi = Material(
    eps_inf=1.0,
    poles=(
        LorentzPole(3.64, 0.0, 8.0),
        LorentzPole(2.76, 2 * 0.063, 2.85),
        LorentzPole(1.73, 2 * 2.5, -0.107),
    ),
    valid_fmin=1.0,
    valid_fmax=1 / 0.4,
)

# SiO2: Horiba Technical Note 08. Valid 0.25-1.77 um.
SiO2 = Material(
    eps_inf=1.0,
    poles=(LorentzPole(1 / 0.103320160833333, 1 / 12.3984193, 1.12),),
    valid_fmin=1 / 1.77,
    valid_fmax=1 / 0.25,
)

# Si3N4: K. Luke et al., Opt. Lett. 40, 4823 (2015). Valid 0.31-5.504 um.
Si3N4 = Material(
    eps_inf=1.0,
    poles=(
        LorentzPole(1 / 0.1353406, 0.0, 3.0249),
        LorentzPole(1 / 1239.842, 0.0, 40314.0),
    ),
    valid_fmin=1 / 5.504,
    valid_fmax=1 / 0.31,
)
