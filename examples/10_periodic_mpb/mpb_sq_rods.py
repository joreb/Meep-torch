# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Square GaAs rod lattice: TM bands (vs mpb)
#
# TM bands of the square rod lattice (r = 0.2, eps = 11.56 GaAs) along
# Gamma-X via the Bloch stepper + harminv, with mpb reference crosses at
# kx = 0.2 and 0.4. A TM gap opens between bands 1 and 2.

# %%
"""meep-torch port of meep/python/examples/mpb_sq_rods.py.

TM bands of the square rod lattice (r=0.2, eps=11.56 GaAs) along Gamma-X
via the doubly-periodic Bloch stepper + harminv, mpb crosses overlaid
(sq_rods_mpb.json: kx=0.2/0.4 within the same 2% family as the gated
eps=8.9 lattice).

Run: uv run python examples/10_periodic_mpb/mpb_sq_rods.py
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

sys.path.insert(0, str(pathlib.Path(__file__).parents[2] / "tests" / "examples"))
from test_mpb_track import _bloch2_bands

OUT = pathlib.Path(__file__).parent / "output"
ref = json.load(open(pathlib.Path(__file__).parents[2] / "tests" / "examples" / "data" / "sq_rods_mpb.json"))

# %% [markdown]
# ## Supersampled rod and Gamma-X sweep

# %%
res, ss, n = 16, 4, 16
dx = 1.0 / res
xi = (torch.arange(n * ss, dtype=torch.float64) + 0.5) / ss - 0.5
II, JJ = torch.meshgrid(xi, xi, indexing="ij")
fine = torch.ones(n * ss, n * ss, dtype=torch.float64)
fine[((II - n / 2) * dx) ** 2 + ((JJ - n / 2) * dx) ** 2 < ref["r"] ** 2] = ref["eps"]
eps = fine.reshape(n, ss, n, ss).mean(dim=(1, 3))

fig, ax = plt.subplots(figsize=(5.4, 4.4))
for kxf in np.arange(0.1, 0.51, 0.1):
    for f in _bloch2_bands(eps, 2 * math.pi * float(kxf), 0.0, 0.4 * dx, dx, 0.7):
        ax.plot(kxf, f, "b.", ms=4)
for kxf, bands in zip(ref["kx"], ref["bands"]):
    ax.plot([kxf] * len(bands), bands, "rx", ms=8, mew=1.4)
ax.set_xlabel("$k_x$ ($2\\pi/a$)")
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("square GaAs rods TM bands (dots: ours, x: mpb)")
fig.tight_layout()
fig.savefig(OUT / "mpb_sq_rods.png", dpi=150)
plt.close(fig)
print("mpb_sq_rods.png written (TM gap between bands 1 and 2)")
