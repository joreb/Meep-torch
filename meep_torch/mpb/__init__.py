# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb-shaped API: ``from meep_torch import mpb``.

Mirrors the surface of meep's mpb python module (solver.py + mpb_data.py)
on the meep-torch Bloch-FDTD + harminv backend; see solver.ModeSolver for
what is backed vs raised.
"""
from .band_functions import (apply_band_func, apply_band_func_thunk,
                             display_group_velocities, display_yparities,
                             display_zparities,
                             combine_band_functions, fix_bfield_phase,
                             fix_dfield_phase, fix_efield_phase,
                             fix_hfield_phase, output_at_kpoint,
                             output_bfield, output_bfield_x, output_bfield_y,
                             output_bfield_z, output_bpwr, output_dfield,
                             output_dfield_x, output_dfield_y, output_dfield_z,
                             output_dpwr, output_dpwr_in_objects,
                             output_efield, output_efield_x, output_efield_y,
                             output_efield_z, output_epsilon, output_hfield,
                             output_hfield_x, output_hfield_y, output_hfield_z,
                             output_tot_pwr)
from .array import MPBArray
from .mpb_data import MPBData, map_data
from .solver import (EVEN_Y, EVEN_Z, NO_PARITY, ODD_Y, ODD_Z,
                     ModeSolver)


def with_hermitian_epsilon():
    return False
