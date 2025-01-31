# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/point_dipole_cyl.py

W7.75 UPDATE (2026-08-08): UNBLOCKED -- stepper_bor + core.ldos cover
this (the W7.5 m=0 LDOS gate is exactly this computation in vacuum).
Port scheduled with the example-port pass.

BLOCKED: cylindrical coordinates (+ LDOS).

meep-torch does not implement cylindrical coordinates (+ LDOS). To port, meep-torch would need
dimensions=mp.CYLINDRICAL (r,z) FDTD with azimuthal m-number expansion (also uses dft_ldos). See the meep original for the physics.

meep original: meep/python/examples/point_dipole_cyl.py
"""
