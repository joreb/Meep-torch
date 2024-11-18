# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""mp.Medium / mp.PML -> meep-torch Material / LorentzPole / Kerr.

meep Medium(epsilon=|index=|eps_inf=, E_susceptibilities=, chi3=) maps to the
engine's nondispersive eps (float) or dispersive Material (LorentzPole list),
with a separate Kerr chi3 painted by the scene. Only the eps/Lorentz/Kerr
subset the 34 ported examples use is supported (master spec W1).
"""
from dataclasses import dataclass, field
from typing import Sequence

from ..core.materials import LorentzPole, Material


@dataclass(frozen=True)
class LorentzianSusceptibility:
    frequency: float
    gamma: float = 0.0
    sigma: float = 1.0

    def to_pole(self) -> LorentzPole:
        return LorentzPole(fn=float(self.frequency), gamma=float(self.gamma),
                           sigma=float(self.sigma))


@dataclass
class Medium:
    epsilon: float | None = None
    index: float | None = None
    eps_inf: float | None = None
    material: Material | None = None     # meep-torch ext: wrap a dispersive engine Material
    E_susceptibilities: Sequence[LorentzianSusceptibility] = field(default_factory=tuple)
    chi3: float = 0.0
    # W5 materials
    D_conductivity: float = 0.0
    B_conductivity: float = 0.0
    mu: float = 1.0

    def resolve(self) -> tuple[float | Material, float]:
        """Return (eps_value_or_Material, chi3)."""
        if self.material is not None:
            return self.material, float(self.chi3)
        if self.epsilon is not None and not isinstance(self.epsilon, Material):
            base = float(self.epsilon)
        elif isinstance(self.epsilon, Material):
            return self.epsilon, float(self.chi3)
        elif self.index is not None:
            base = float(self.index) ** 2
        elif self.eps_inf is not None:
            base = float(self.eps_inf)
        else:
            base = 1.0
        poles = tuple(s.to_pole() for s in self.E_susceptibilities)
        if poles:
            return Material(eps_inf=base, poles=poles,
                            valid_fmin=0.0, valid_fmax=float("inf")), float(self.chi3)
        return base, float(self.chi3)


@dataclass(frozen=True)
class PML:
    thickness: float = 1.0
    # meep PML(side=...) etc. not needed by the 34 examples; thickness only.


@dataclass(frozen=True)
class Absorber:
    """meep mp.Absorber: matched sigma_d = sigma_b quadratic ramp instead of
    PML coordinate stretching. Applied on all cell edges (direction args from
    meep are accepted and ignored -- 1D/2D cells here absorb on every edge)."""

    thickness: float = 1.0
    direction: int | None = None

    def __init__(self, thickness: float = 1.0, direction: int | None = None, **_kw):
        object.__setattr__(self, "thickness", float(thickness))
        object.__setattr__(self, "direction", direction)