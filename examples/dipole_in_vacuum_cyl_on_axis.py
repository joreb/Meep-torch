# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/dipole_in_vacuum_cyl_on_axis.py

W7.75 UPDATE (2026-08-08): UNBLOCKED -- stepper_bor (m=0) +
near2far_cyl cover this (see plot_radiation_pattern_dipole.py for the
working pattern). Port scheduled with the example-port pass.

BLOCKED: cylindrical coordinates + near2far.

meep-torch does not implement cylindrical coordinates + near2far. To port, meep-torch would need
dimensions=mp.CYLINDRICAL plus add_near2far/get_farfield for the radiation pattern. See the meep original for the physics.

meep original: meep/python/examples/dipole_in_vacuum_cyl_on_axis.py
"""
