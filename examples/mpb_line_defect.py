# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/mpb_line_defect.py.

W1 line-defect waveguide in the triangular rod lattice: 7-row
rectangular supercell (one rod removed), projected TM bands along kx via
the doubly-periodic Bloch stepper + harminv. The bulk TM gap of the tri
lattice (~0.27-0.44, from tri_rods_mpb.json M/K edges) is shaded; modes
inside it are defect-guided.

SUBSTITUTIONS: mpb.ModeSolver supercell -> step_bloch2_tm; a point
source does not excite every folded continuum band (the gate in
test_mpb_track.py checks no-spurious containment instead).

Run: uv run python examples/mpb_line_defect.py
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tests" / "examples"))
from test_mpb_track import _bloch2_bands

OUT = pathlib.Path(__file__).parent
DATA = OUT.parent / "tests" / "examples" / "data"
ld = json.load(open(DATA / "line_defect_mpb.json"))
tri = json.load(open(DATA / "tri_rods_mpb.json"))

res = 26
nx, ny = int(round(math.sqrt(3) * res)), 7 * res
Lx, Ly = nx / res, 7.0
ss = 2
xi = ((torch.arange(nx * ss, dtype=torch.float64) + 0.5) / ss) / res
yj = ((torch.arange(ny * ss, dtype=torch.float64) + 0.5) / ss) / res
II, JJ = torch.meshgrid(xi, yj, indexing="ij")
fine = torch.ones(nx * ss, ny * ss, dtype=torch.float64)
for i2 in (0, 1):
    for j2 in range(7):
        cx, cy = i2 * Lx / 2 % Lx, (j2 + i2 / 2) % Ly
        if abs(cx) < 1e-9 and abs(cy - 3.0) < 1e-9:
            continue                              # removed rod
        dxp = (II - cx) - Lx * torch.round((II - cx) / Lx)
        dyp = (JJ - cy) - Ly * torch.round((JJ - cy) / Ly)
        fine[dxp ** 2 + dyp ** 2 < ld["r"] ** 2] = ld["eps"]
eps = fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
dx = 1.0 / res

gap_lo = tri["M"][0]                              # bulk band-1 top (M)
gap_hi = min(tri["M"][1], tri["K"][1])            # bulk band-2 bottom
fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.axhspan(gap_lo, gap_hi, color="orange", alpha=0.25, label="bulk TM gap")
for kxf in (0.1, 0.2, 0.3, 0.4, 0.5):
    kx = kxf * 2 * math.pi / math.sqrt(3)
    ky = kxf * 2 * math.pi
    for f in _bloch2_bands(eps, kx, ky, 0.4 * dx, dx, 0.5, t_end=350.0):
        ax.plot(kxf, f, "b.", ms=4)
ax.set_xlabel("$k_x$ (reciprocal units)")
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("W1 line defect, projected bands (7-row supercell)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "mpb_line_defect.png", dpi=150)
plt.close(fig)
print("mpb_line_defect.png written (defect-guided modes appear in the gap)")
