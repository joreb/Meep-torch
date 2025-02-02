# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/disc_radiation_pattern.py.

Radiation pattern of an Er dipole embedded in a high-index disc
(radius 1.2, n = 2.4, thickness 0.29) via the W7.9 general-m BOR
near-to-far transform (m = 1).

SUBSTITUTIONS:
  * mp cylindrical + get_farfield -> stepper_bor + farfield_bor_m.
  * Resolution 25 (meep's 100); single m = 1 term (the meep example's
    dipole-in-disc at r > 0 formally sums m; the m = 1 term dominates
    for r_src well inside the first radial lobe).

Run: uv run python examples/disc_radiation_pattern.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.near2far_cyl import farfield_bor_m, run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state

OUT = pathlib.Path(__file__).parent
FCEN, RES = 1.0, 25
N_DISC, R_DISC, H_DISC = 2.4, 1.2, 0.29

dr = 1.0 / RES
nr, nz = int(3.0 * RES), int(6.0 * RES)
eps = torch.ones(nr, nz, dtype=torch.float64)
jz0 = int((3.0 - H_DISC / 2) * RES)
jz1 = int((3.0 + H_DISC / 2) * RES)
eps[: int(R_DISC * RES), jz0:jz1] = N_DISC ** 2
c = build_bor_coeffs(nr, nz, dr, m=1, eps=eps, n_abs_r=RES, n_abs_z=RES)
src = ("Er", int(0.5 * R_DISC * RES), nz // 2)
pulse = GaussianPulse(FCEN, 0.2)
_, planes = run_bor_dft(init_bor_state(nr, nz), c,
                        int(round(80.0 / c.dt)),
                        src=src, src_profile=pulse, freqs=[FCEN])
thetas = np.radians(np.linspace(5, 175, 120))
pattern = farfield_bor_m(planes, c, 1, ir=int(2.0 * RES),
                         jz0=int(1.5 * RES), jz1=int(4.5 * RES),
                         thetas=thetas, k=2 * math.pi * FCEN)[0]

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.4, 5))
ax.plot(thetas, pattern, "b-", lw=1.2)
ax.plot(-thetas, pattern, "b-", lw=1.2)
ax.set_theta_zero_location("N")
ax.set_title(f"Er dipole in n={N_DISC} disc (m=1)")
fig.tight_layout()
fig.savefig(OUT / "disc_radiation_pattern.png", dpi=150)
plt.close(fig)
pk = float(np.degrees(thetas[pattern.argmax()]))
print(f"disc_radiation_pattern.png  main lobe at theta = {pk:.1f} deg")
