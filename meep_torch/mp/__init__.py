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
from .material import LorentzianSusceptibility, Medium, PML
from .simulation import FluxRegion, Simulation
from .sources import (ContinuousSource, Dielectric, EigenModeSource, Ex, Ey, Ez,
                      GaussianSource, Hz, Hy, Source, pol_for)
from .vec import Vector3, interpolate

inf = math.inf


class Mirror:
    """meep mp.Mirror symmetry -- NOT emulated in W1 (full cell run).

    Present so meep syntax parses; passing it has no effect (the examples were
    ported with full cells). Halving lands with a later workstream.
    """

    def __init__(self, *args, **kwargs):
        self.args = args