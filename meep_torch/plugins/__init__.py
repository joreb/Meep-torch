# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch plugins -- gplugins-style integration layer.

- ``meep_torch.plugins.gdsfactory``: rasterize gdsfactory Components into
  meep-torch simulations (the GPL-free counterpart of gplugins.gmeep).
- ``meep_torch.plugins.rings``: half-ring coupler extraction + analytic
  ring-spectrum assembly.
- ``meep_torch.plugins.validate3d``: true-3D validation of 2D
  effective-index designs (extrusion + 3D mode sources).
"""
