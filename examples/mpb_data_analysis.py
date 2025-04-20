# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/mpb_data_analysis.py.

Mode-profile analysis: extract the Bloch mode FIELD PATTERN at a band
frequency -- run the doubly-periodic cell, DFT Ez at the harminv band
frequency, plot |Ez| over the unit cell (tri-rod lattice, first band
at K).

SUBSTITUTIONS: mpb's mode-coefficient export -> time-domain DFT at the
band frequency (equivalent up to normalization for isolated bands).

Run: uv run python examples/mpb_data_analysis.py
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
from test_mpb_track import _tri_eps
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bloch2 import init_bloch2_state, step_bloch2_tm

OUT = pathlib.Path(__file__).parent
ref = json.load(open(OUT.parent / "tests" / "examples" / "data" / "tri_rods_mpb.json"))
eps, dx = _tri_eps(res=26, r=ref["r"], eps_rod=ref["eps"])
f_band = ref["K"][0]                              # first band at K
kx, ky = 0.0, 2 * math.pi / 3

class _G:
    nx, ny = eps.shape
    zeros = staticmethod(lambda *s: torch.zeros(*s, dtype=torch.float64))

st = init_bloch2_state(_G)
mask = torch.zeros(*eps.shape, dtype=torch.float64)
mask[_G.nx // 3 + 1, _G.ny // 5 + 2] = 1.0
pulse = GaussianPulse(f_band, 0.1)                # narrowband at the mode
dt = 0.4 * dx
n_steps = int(round(300.0 / dt))
acc = np.zeros(eps.shape, dtype=complex)
for t in range(n_steps):
    a = pulse.amplitude((t + 0.5) * dt)
    st = step_bloch2_tm(st, eps, dt, dx, kx, ky, src_c=a, src_s=0.0, src_mask=mask)
    if t > n_steps // 3:
        ph = np.exp(-2j * math.pi * f_band * (t + 1.0) * dt) * dt
        acc += (st.Ez_c + 1j * st.Ez_s).numpy() * ph

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
ax1.imshow(eps.numpy().T, origin="lower", cmap="binary")
ax1.set_title("eps (rect supercell, 2 rods)")
ax2.imshow(np.abs(acc).T, origin="lower", cmap="inferno")
ax2.set_title(f"|Ez| mode profile, band 1 at K (f={f_band:.3f})")
fig.tight_layout()
fig.savefig(OUT / "mpb_data_analysis.png", dpi=150)
plt.close(fig)
peak_on_rod = np.abs(acc)[0:5, 0:5].max() > 0.5 * np.abs(acc).max()
print(f"mpb_data_analysis.png  dielectric-band profile concentrated on rods: "
      f"{'yes' if peak_on_rod else 'no'}")
