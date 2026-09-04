# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep.mpb-shaped API: ``from meep_torch import mpb``.

Mirrors the surface of meep's mpb python module (solver.py + mpb_data.py)
on the meep-torch Bloch-FDTD + harminv backend; see solver.ModeSolver for
what is backed vs raised.
"""
from .mpb_data import MPBData, map_data
from .solver import (EVEN_Y, EVEN_Z, NO_PARITY, ODD_Y, ODD_Z, ModeSolver,
                     MPBArray)


def with_hermitian_epsilon():
    return False
