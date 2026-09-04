# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # On-axis horizontal dipole in vacuum (BOR, m = 1)
#
# Radiation pattern of an on-axis horizontal (Er) dipole via the
# body-of-revolution near-to-far transform: a single m = +1 run represents the
# circular dipole (x + i y)/sqrt(2), whose exact pattern is (1 + cos^2)/2.

# %%
"""meep-torch port of meep/python/examples/dipole_in_vacuum_cyl_on_axis.py.

Radiation pattern of an on-axis HORIZONTAL (Er) dipole in vacuum via the
BOR (m = 1) near-to-far transform. A single m = +1 run corresponds to the
circular dipole (x + i y)/sqrt(2), whose pattern is (1 + cos^2 theta)/2.

SUBSTITUTIONS:
  * mp cylindrical Simulation + get_farfield -> core stepper_bor +
    near2far_cyl.farfield_bor_m (W7.9). Resolution 25 (meep's 50).

Run: uv run python examples/09_cylindrical/dipole_in_vacuum_cyl_on_axis.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.near2far_cyl import farfield_bor_m, run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## BOR run and near-to-far transform
#
# The Er + i*Ephi source pair at the same point reproduces meep's
# `amplitude=1j` Ep trick for a circular dipole. DFT phasors on a cylindrical
# monitor surface feed `farfield_bor_m` to get |E(theta)|^2.

# %%
FCEN, RES = 1.0, 25

dr = 1.0 / RES
nr, nz = int(3.0 * RES), int(6.0 * RES)
c = build_bor_coeffs(nr, nz, dr, m=1, n_abs_r=RES, n_abs_z=RES)
pulse = GaussianPulse(FCEN, 0.2)
# circular on-axis dipole (x + i y): Er + i Ephi at the same point (the
# meep example's amplitude=1j Ep trick); Er node at r = dr/2
src = [("Er", 1, nz // 2, 1.0), ("Ephi", 1, nz // 2, 1j)]   # ~1.5 dr off-axis (meep #2704)
_, planes = run_bor_dft(init_bor_state(nr, nz), c,
                        int(round(60.0 / c.dt)),
                        src=src, src_profile=pulse, freqs=[FCEN])
thetas = np.radians(np.linspace(5, 175, 100))
pattern = farfield_bor_m(planes, c, 1, ir=int(1.5 * RES),
                         jz0=int(1.5 * RES), jz1=int(4.5 * RES),
                         thetas=thetas, k=2 * math.pi * FCEN)[0]
analytic = (1 + np.cos(thetas) ** 2) / 2

# %% [markdown]
# ## Polar plot vs analytic pattern

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.4, 5))
ax.plot(thetas, pattern, "b-", lw=1.2, label="meep-torch BOR n2f (m=1)")
ax.plot(thetas, analytic, "r--", lw=1.0, label="(1 + cos^2)/2")
ax.plot(-thetas, pattern, "b-", lw=1.2)
ax.plot(-thetas, analytic, "r--", lw=1.0)
ax.set_theta_zero_location("N")
ax.set_title("on-axis horizontal (Er) dipole, m = 1")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "dipole_in_vacuum_cyl_on_axis.png", dpi=150)
plt.close(fig)
rms = float(np.sqrt(np.mean((pattern - analytic / analytic.max()) ** 2)))
print(f"dipole_in_vacuum_cyl_on_axis.png  rms vs (1+cos^2)/2 = {rms:.4f}")
