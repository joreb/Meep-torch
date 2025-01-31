# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/dipole_in_vacuum_cyl_off_axis.py

W7.75 UPDATE (2026-08-08): partially unblocked -- stepper_bor runs any
m, but an off-axis dipole needs the m-summed far fields. W7.9 UPDATE: the
|m| >= 1 radiation integrals landed (farfield_bor_m, m=1 oracle gate
<=5% rms); the port needs the per-m source decomposition sum.

BLOCKED: cylindrical coordinates + near2far.

meep-torch does not implement cylindrical coordinates + near2far. To port, meep-torch would need
dimensions=mp.CYLINDRICAL plus add_near2far/get_farfield for the radiation pattern. See the meep original for the physics.

meep original: meep/python/examples/dipole_in_vacuum_cyl_off_axis.py
"""
