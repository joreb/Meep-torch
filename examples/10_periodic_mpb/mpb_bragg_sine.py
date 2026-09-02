# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Sinusoidal-index Bragg stack bands (vs mpb)
#
# Band structure of a 1D Bragg stack whose refractive index varies smoothly
# (cos profile, index 1..3) rather than in binary steps. Computed with the
# doubly-periodic Bloch stepper + harminv and overlaid with meep.mpb
# reference bands; the first gap opens at k = 0.5.

# %%
"""meep-torch port of meep/python/examples/mpb_bragg_sine.py.

Band structure of a 1D sinusoidal-index Bragg stack (index 1..3, cos
profile) via the doubly-periodic Bloch stepper + harminv, overlaid with
the meep.mpb reference bands (tests/examples/data/bragg_sine_mpb.json).

SUBSTITUTIONS: mpb.ModeSolver -> step_bloch2_tm k-sweep + W3 harminv
(first ~3 bands; mpb's 8-band sweep reduced).

Run: uv run python examples/10_periodic_mpb/mpb_bragg_sine.py
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
ref = json.load(open(pathlib.Path(__file__).parents[2] / "tests" / "examples" / "data" / "bragg_sine_mpb.json"))

# %% [markdown]
# ## eps profile from the oracle parameters and k-sweep

# %%
res = 32
dx = 1.0 / res
x = (torch.arange(res, dtype=torch.float64) + 0.5) * dx
idx = ref["index_min"] + 0.5 * (ref["index_max"] - ref["index_min"]) \
    * (1 + torch.cos(2 * math.pi * x))
eps = (idx ** 2)[:, None].expand(res, 4).clone()

fig, ax = plt.subplots(figsize=(5.6, 4.4))
ks = np.arange(0.05, 0.51, 0.05)
for k in ks:
    for f in _bloch2_bands(eps, 2 * math.pi * float(k), 0.0, 0.4 * dx, dx,
                           0.8, t_end=400.0):
        if f < 0.8:
            ax.plot(k, f, "b.", ms=4)
for kx, bands in zip(ref["kx"], ref["bands"]):
    ax.plot([kx] * len(bands), bands, "rx", ms=7, mew=1.4)
ax.set_xlabel("$k_x$ ($2\\pi/a$)")
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("sine-index Bragg bands (dots: meep-torch, x: meep.mpb)")
fig.tight_layout()
fig.savefig(OUT / "mpb_bragg_sine.png", dpi=150)
plt.close(fig)
print("mpb_bragg_sine.png written (first gap opens at k = 0.5)")
