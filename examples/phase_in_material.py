# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""STUB -- meep/python/examples/phase_in_material.py is NOT ported.

Blocking feature: meep's `sim1.fields.phase_in_material(sim2.structure, dt)`
applies a per-cell phase shift exp(-i omega dt) to the fields inside the
material region of a second simulation, treating the local refractive index as
an optical path length. It is a specialized internal used for things like
virtual-displacement force/gradient calculations and phase-manipulation
diagnostics, not a standalone FDTD propagation.

What would be needed in meep-torch:
  - a per-cell complex phase mask built from a second geometry's epsilon field
    (a second rasterized scene), and
  - complex-valued field storage (meep-torch currently runs real-valued TM/TE
    FDTD only), so the exp(-i omega dt) multiplier has no carrier to act on.
  - an entry point analogous to meep's fields.phase_in_material.

The closest available primitives (rasterize_scene for the second geometry, and
the DFT monitors' complex accumulators) do not add up to this operation. A
faithful port would require complex-field FDTD support first; forcing it onto
real TM stepping would not reproduce the phase-shifted-epsilon diagnostic.

Run: this file is a stub; `uv run python examples/phase_in_material.py` prints
this notice and exits 0.
"""
import sys

print(__doc__)
print("phase_in_material.py: STUB -- requires complex-field FDTD + phase_in_material op (unimplemented)")