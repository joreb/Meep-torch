# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/binary_grating_n2f.py

W4.5 UPDATE (2026-08-08): periodic tiling landed
(near2far.tile_faces, grating-order lobe gate). Port pending the example's
DiffractedPlanewave normalization pass.

BLOCKED: near-to-far-field transform.

meep-torch does not implement near-to-far-field transform. To port, meep-torch would need
add_near2far / get_farfields for the diffracted far-field spectrum. See the meep original for the physics.

meep original: meep/python/examples/binary_grating_n2f.py
"""
