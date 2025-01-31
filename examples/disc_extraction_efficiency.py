# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/disc_extraction_efficiency.py

W7.75 UPDATE (2026-08-08): partially unblocked -- stepper_bor + m=0
near2far landed; extraction efficiency needs |m| >= 1 far fields (landed, W7.9) and
cyl flux normalization (still open).

BLOCKED: cylindrical coordinates + near2far.

meep-torch does not implement cylindrical coordinates + near2far. To port, meep-torch would need
dimensions=mp.CYLINDRICAL plus add_near2far/get_farfield for the disc extraction efficiency. See the meep original for the physics.

meep original: meep/python/examples/disc_extraction_efficiency.py
"""
