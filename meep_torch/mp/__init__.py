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
from .run_control import (DftEnergy, DftFields, DftFlux, DftForce, DftLdos,
                          DftNear2Far, DftObj, FluxData, Harminv, PadeDFT,
                          Volume, after_sources, after_time, at_beginning,
                          at_end, at_every, at_time, before_time,
                          during_sources, stop_after_walltime,
                          stop_when_fields_decayed)
from .simulation import (EnergyRegion, FluxRegion, ForceRegion, Identity, Mirror,
                         ModeRegion, Near2FarRegion, P, R, Rotate2, Rotate4,
                         Simulation, Symmetry, X, Y, Z)
from .sources import (ALL_COMPONENTS, AUTOMATIC, Bx, By, Bz, ContinuousSource,
                      CustomSource, Dielectric, DiffractedPlanewave, Dx, Dy, Dz,
                      EVEN_Y, EVEN_Z, EigenModeSource, Ex, Ey, Ez,
                      GaussianBeam2DSource, GaussianBeam3DSource,
                      GaussianBeamSource, GaussianSource, Hx, Hy, Hz,
                      IndexedSource, NO_PARITY, ODD_Y, ODD_Z, Source, pol_for)
from .vec import Vector3, interpolate
from .verbosity_mgr import Verbosity, verbosity

inf = math.inf

# predefined media (meep geom.py definitions, verbatim)
vacuum = Medium(epsilon=1)
air = Medium(epsilon=1)
metal = Medium(epsilon=-inf)
perfect_electric_conductor = Medium(epsilon=-inf)
perfect_magnetic_conductor = Medium(mu=-inf)


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
# C++ binding layer: sources/dft (SWIG-surface parity)
from .cpp_bindings_src import (continuous_src_time, custom_py_src_time,
                               custom_src_time, dft_data, dft_energy, dft_fields,
                               dft_flux, dft_force, dft_near2far,
                               diffractedplanewave, eigenmode_data,
                               gaussian_src_time, gaussianbeam,
                               gyrotropic_susceptibility,
                               lorentzian_susceptibility,
                               multilevel_susceptibility, near_src_data,
                               noisy_lorentzian_susceptibility, pol,
                               polarization_state, sourcedata, src_time,
                               susceptibility)

# C++ binding layer: fields/structure (SWIG-surface parity)
from .cpp_bindings_core import (absorber, boundary_region, cond_profile, fields,
                                flux_vol, h5file, initialize, structure)


# C++ binding layer (SWIG-surface parity)
from .cpp_bindings_geom import (D1, D2, D3, Dcyl, clean_vec, geom_box,
                                geom_epsilon, grid_volume, identity, ivec,
                                iveccyl, material_function, mirror,
                                one_ivec, one_vec, r_to_minus_r_symmetry,
                                rotate2, rotate4, signed_direction,
                                simple_material_function, symm_matrix,
                                symmetry, unit_ivec, unit_vec, vec, vector3, veccyl,
                                vol1d, vol2d, vol3d, volcyl, volone, voltwo,
                                volume, volume_list, zero_ivec, zero_vec)

# python-layer parity exports (implemented in submodules; wired here)
from .geometry import (Cone, Ellipsoid, GeometricObject, Lattice, Matrix, Mesh,
                       Sphere, Wedge, cartesian_to_lattice, cartesian_to_reciprocal,
                       get_rotation_matrix, lattice_to_cartesian,
                       lattice_to_reciprocal, reciprocal_to_cartesian,
                       reciprocal_to_lattice)
from .sources import SourceTime
from .visualization import Animate2D, JS_Animation, plot2D
from .material import MaterialGrid

# module-level meep function surface (functions.py: portable helpers, single-
# process parallel semantics, Dft accessors; output.py: HDF5 output family).
# The star imports honor each module's __all__; functions.interpolate
# deliberately replaces vec.interpolate (full meep list semantics).
from .functions import *   # noqa: F401,F403
from .output import *      # noqa: F401,F403
