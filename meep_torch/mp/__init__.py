# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-shaped Python API (W1). `import meep_torch.mp as mp` mirrors the meep
surface the 34 ported examples use; adapter-only, wraps the engine.

W1 scope: Simulation, Medium, PML, Block/Cylinder/Prism, Vector3,
GaussianSource/ContinuousSource/Source/EigenModeSource, FluxRegion,
add_flux/get_fluxes/get_flux_freqs, add_mode_monitor/get_eigenmode_coefficients,
get_array, run(until=). No Harminv/symmetries/oblique/near2far (later waves).
"""
import math

from .coords import GridSpec, cell_index, grid_spec, pml_npml, rect_corners, to_corner
from .geometry import Block, Cylinder, Prism, build_scene_items
from .material import (ALL, Absorber, DrudeSusceptibility, FreqRange,
                       GyrotropicDrudeSusceptibility,
                       GyrotropicLorentzianSusceptibility,
                       GyrotropicSaturatedSusceptibility, High,
                       LorentzianSusceptibility, Low, Medium, MultilevelAtom,
                       NoisyDrudeSusceptibility, NoisyLorentzianSusceptibility,
                       PML, Susceptibility, Transition, get_rotation_matrix)
from .run_control import (DftFields, DftFlux, DftLdos, DftNear2Far, DftObj, Harminv,
                          Volume, after_sources, after_time, at_beginning, at_end,
                          at_every, at_time, before_time, during_sources,
                          stop_after_walltime, stop_when_fields_decayed, with_prefix)
from .simulation import (FluxRegion, Identity, Mirror, Near2FarRegion, P, R,
                         Rotate2, Rotate4, Simulation, Symmetry, X, Y, Z)
from .sources import (ContinuousSource, CustomSource, Dielectric, EigenModeSource,
                      Ex, Ey, Ez, GaussianSource, Hz, Hy, Source, pol_for)
from .vec import Vector3, interpolate

inf = math.inf


# meep module-level retrieval functions: operate on the Dft objects the add_*
# monitor methods return (each carries its Simulation and freqs).
def get_fluxes(f):
    return f.sim.get_fluxes(f)


def get_flux_freqs(f):
    return f.sim.get_flux_freqs(f)


def get_ldos_freqs(f):
    return list(f.freqs)


def get_near2far_freqs(f):
    return list(f.freqs)