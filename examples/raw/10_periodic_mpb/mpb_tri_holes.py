# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Triangular air-hole lattice: TM bands at M (vs mpb)
#
# Large air holes (r = 0.45) in an eps = 12 background, folded onto the
# rectangular sqrt(3) x 1 supercell. Only the TM bands are computed — the
# famous complete gap of this structure also needs TE, which the bloch2
# stepper does not cover.

# %%
"""meep-torch port of meep/python/examples/mpb_tri_holes.py.

Triangular lattice of air holes (r=0.45, eps=12): TM bands at M via the
rect-supercell fold, mpb crosses (tri_holes_mpb.json).

SUBSTITUTIONS: TM only -- the famous COMPLETE gap of this structure
needs the TE bands too; a TE doubly-periodic Bloch stepper is the noted
remaining gap (bloch2 is TM).

Run: uv run python examples/10_periodic_mpb/mpb_tri_holes.py
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "tests" / "examples"))
from test_mpb_track import _bloch2_bands

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
ref = json.load(open(pathlib.Path(__file__).parents[3] / "tests" / "examples" / "data" / "tri_holes_mpb.json"))
import torch

# %% [markdown]
# ## Supercell rasterization and the M-point ringdown
#
# Two holes per supercell (corner + center) with periodic wrap-around,
# supersampled 4x; a single Bloch ringdown at M yields the band frequencies
# compared against the mpb crosses.

# %%
# air holes (eps=1) in an eps=12 background, rect sqrt(3) x 1 supercell
res = 26
nx, ny = int(round(math.sqrt(3) * res)), res
Lx, Ly = nx / res, 1.0
ss = 4
xi = ((torch.arange(nx * ss, dtype=torch.float64) + 0.5) / ss) / res
yj = ((torch.arange(ny * ss, dtype=torch.float64) + 0.5) / ss) / res
II, JJ = torch.meshgrid(xi, yj, indexing="ij")
fine = torch.full((nx * ss, ny * ss), float(ref["eps"]))
for cx, cy in ((0.0, 0.0), (Lx / 2, Ly / 2)):
    dxp = (II - cx) - Lx * torch.round((II - cx) / Lx)
    dyp = (JJ - cy) - Ly * torch.round((JJ - cy) / Ly)
    fine[dxp ** 2 + dyp ** 2 < ref["r"] ** 2] = 1.0
eps = fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
dx = 1.0 / res

got = _bloch2_bands(eps, math.pi / math.sqrt(3), math.pi, 0.4 * dx, dx, 0.6,
                    t_end=350.0)
fig, ax = plt.subplots(figsize=(4.2, 4.2))
for f in got:
    ax.plot(0, f, "b.", ms=6)
for f in ref["M_tm"]:
    ax.plot(0, f, "rx", ms=9, mew=1.5)
ax.set_xticks([0], ["M"])
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("tri air holes, TM at M (dots: ours, x: mpb)")
fig.tight_layout()
fig.savefig(OUT / "mpb_tri_holes.png", dpi=150)
plt.close(fig)
print("mpb_tri_holes.png  TM at M:", [round(f, 4) for f in got[:5]],
      "mpb:", [round(f, 4) for f in ref["M_tm"]])
