# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port stub: meep/python/examples/mie_scattering.py

BLOCKED: near-to-far field transform (add_near2far / get_farfield) for the
scattering cross section, plus cylindrical-friendly far-field angular pattern.

meep-torch has no near2far flux region or far-field transform. To port,
meep-torch would need a near2far DFT surface + a Stratton-Chu/k-space
propagation integral to reconstruct the far field. See the meep original
for the physics (it also cross-checks against PyMieScatt).

meep original: meep/python/examples/mie_scattering.py
"""