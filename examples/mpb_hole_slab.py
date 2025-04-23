# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/mpb_hole_slab.py.

z-even guided bands of the triangular air-hole slab (h=0.5, r=0.3,
eps=12) at the K point via the 3D Bloch stepper (periodic x/y, absorber
z, midplane source selects z-even parity), vs meep.mpb crosses.

Run: uv run python examples/mpb_hole_slab.py
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
import test_mpb_track as T
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bloch3 import (build_bloch3_coeffs,
                                            init_bloch3_state, step_bloch3)
from meep_torch.modes.harminv import harminv

OUT = pathlib.Path(__file__).parent
ref = json.load(open(OUT.parent / "tests" / "examples" / "data" / "hole_slab_mpb.json"))

res = 16
nx, ny, nz = int(round(math.sqrt(3) * res)), res, 4 * res
Lx, Ly, dx = nx / res, 1.0, 1.0 / res
ss = 4
xi = ((torch.arange(nx * ss, dtype=torch.float64) + 0.5) / ss) / res
yj = ((torch.arange(ny * ss, dtype=torch.float64) + 0.5) / ss) / res
II, JJ = torch.meshgrid(xi, yj, indexing="ij")
plane = torch.full((nx * ss, ny * ss), float(ref["eps"]))
for cx, cy in ((0.0, 0.0), (Lx / 2, Ly / 2)):
    dxp = (II - cx) - Lx * torch.round((II - cx) / Lx)
    dyp = (JJ - cy) - Ly * torch.round((JJ - cy) / Ly)
    plane[dxp ** 2 + dyp ** 2 < ref["r"] ** 2] = 1.0
plane = plane.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
eps = torch.ones(nx, ny, nz, dtype=torch.float64)
jz0, jz1 = nz // 2 - int(ref["h"] / 2 * res), nz // 2 + int(ref["h"] / 2 * res)
eps[:, :, jz0:jz1] = plane[:, :, None]

c = build_bloch3_coeffs(eps, dx, 0.0, 2 * math.pi / 3, n_abs_z=res // 2)
st = init_bloch3_state(nx, ny, nz)
pulse = GaussianPulse(0.3, 0.3)
src = ("Ey", nx // 3 + 1, ny // 5 + 2, nz // 2)
n_steps = int(round(250.0 / c.dt))
series = np.empty(n_steps, dtype=complex)
for t in range(n_steps):
    st = step_bloch3(st, c, pulse.amplitude((t + 0.5) * c.dt), src)
    series[t] = st.Ey[nx // 3 + 1, ny // 5 + 2, nz // 2].item()
modes = [m for m in harminv(series[int(50.0 / c.dt):], c.dt, 0.15, 0.45)
         if m.Q > 100 and m.amp > 1e-9]
got = sorted(m.freq for m in modes)

fig, ax = plt.subplots(figsize=(4.2, 4.2))
for f in got:
    ax.plot(0, f, "b.", ms=6)
for f in ref["K_zeven"]:
    ax.plot(0, f, "rx", ms=9, mew=1.5)
ax.set_xticks([0], ["K"])
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("hole slab z-even at K (dots: ours, x: mpb)")
fig.tight_layout()
fig.savefig(OUT / "mpb_hole_slab.png", dpi=150)
plt.close(fig)
print("mpb_hole_slab.png  bands:", [round(f, 4) for f in got[:5]])
