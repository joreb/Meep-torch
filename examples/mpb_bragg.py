# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/mpb_bragg.py.

1D quarter-wave-ish Bragg stack (n=1/3, w=0.25) TM bands via the
doubly-periodic Bloch stepper + harminv, mpb crosses overlaid
(bragg_mpb.json, kx=0.3/0.5).

Run: uv run python examples/mpb_bragg.py
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
ref = json.load(open(OUT.parent / "tests" / "examples" / "data" / "bragg_mpb.json"))

res = 32
dx = 1.0 / res
x = (torch.arange(res, dtype=torch.float64) + 0.5) * dx
eps1 = torch.where((x > 0.5 - ref["w"] / 2) & (x < 0.5 + ref["w"] / 2),
                   torch.tensor(ref["n_hi"] ** 2), torch.tensor(ref["n_lo"] ** 2))
eps = eps1[:, None].expand(res, 4).clone()

fig, ax = plt.subplots(figsize=(5.4, 4.4))
for kxf in np.arange(0.1, 0.51, 0.1):
    for f in _bloch2_bands(eps, 2 * math.pi * float(kxf), 0.0, 0.4 * dx, dx,
                           1.0, t_end=400.0):
        ax.plot(kxf, f, "b.", ms=4)
for kxf, bands in zip(ref["kx"], ref["bands"]):
    ax.plot([kxf] * len(bands), bands, "rx", ms=8, mew=1.4)
ax.set_xlabel("$k_x$ ($2\\pi/a$)")
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("Bragg stack bands, n=1/3 (dots: ours, x: mpb)")
fig.tight_layout()
fig.savefig(OUT / "mpb_bragg.png", dpi=150)
plt.close(fig)
print("mpb_bragg.png written (large gap at the zone edge)")
