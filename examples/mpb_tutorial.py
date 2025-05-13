# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/mpb_tutorial.py.

The mpb tutorial's first computation -- TM bands of a square rod lattice
-- via the Bloch + harminv route. Thin wrapper over the machinery shared
with mpb_sq_rods.py (eps=12 tutorial rods).

Run: uv run python examples/mpb_tutorial.py
"""
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
res, ss, n = 16, 4, 16
dx = 1.0 / res
xi = (torch.arange(n * ss, dtype=torch.float64) + 0.5) / ss - 0.5
II, JJ = torch.meshgrid(xi, xi, indexing="ij")
fine = torch.ones(n * ss, n * ss, dtype=torch.float64)
fine[((II - n / 2) * dx) ** 2 + ((JJ - n / 2) * dx) ** 2 < 0.2 ** 2] = 12.0
eps = fine.reshape(n, ss, n, ss).mean(dim=(1, 3))

fig, ax = plt.subplots(figsize=(5.4, 4.4))
for kxf in np.arange(0.1, 0.51, 0.1):
    for f in _bloch2_bands(eps, 2 * math.pi * float(kxf), 0.0, 0.4 * dx, dx, 0.7):
        ax.plot(kxf, f, "b.", ms=4)
ax.set_xlabel("$k_x$ ($2\\pi/a$)")
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("mpb tutorial: square eps=12 rods, TM bands")
fig.tight_layout()
fig.savefig(OUT / "mpb_tutorial.png", dpi=150)
plt.close(fig)
print("mpb_tutorial.png written")
