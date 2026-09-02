# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # On-axis dipole radiation pattern (BOR m=0)
#
# meep-torch port of meep/python/examples/plot_radiation_pattern_dipole.py.
# Radiation pattern of an on-axis Ez point dipole via the W7.75 cylindrical
# (BOR, m = 0) near-to-far transform, overlaid with the analytic sin^2 theta
# dipole pattern. Output: `output/plot_radiation_pattern_dipole.png`.
#
# SUBSTITUTIONS:
# * mp.Simulation(dimensions=CYLINDRICAL) + add_near2far/get_farfield ->
#   core stepper_bor + near2far_cyl (m = 0 Bessel radiation integrals).
#
# Run: `uv run python examples/05_monitors_analysis/plot_radiation_pattern_dipole.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.near2far_cyl import farfield_bor_m0, run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, RES = 1.0, 25

# %% [markdown]
# ## BOR run and far-field transform
# A pulsed on-axis Ez dipole in vacuum; the DFT planes feed the m=0 Bessel
# radiation integrals over polar angle.

# %%
dr = 1.0 / RES
nr, nz = int(3.0 * RES), int(6.0 * RES)
c = build_bor_coeffs(nr, nz, dr, m=0, n_abs_r=RES, n_abs_z=RES)
pulse = GaussianPulse(FCEN, 0.2)
src = ("Ez", 0, nz // 2)
_, planes = run_bor_dft(init_bor_state(nr, nz), c,
                        int(round(60.0 / c.dt)),
                        src=src, src_profile=pulse, freqs=[FCEN])
thetas = np.radians(np.linspace(5, 175, 100))
pattern = farfield_bor_m0(planes, c, ir=int(1.5 * RES),
                          jz0=int(1.5 * RES), jz1=int(4.5 * RES),
                          thetas=thetas, k=2 * math.pi * FCEN)[0]

# %% [markdown]
# ## Plot
# Polar pattern vs sin^2 theta; console prints the rms deviation.

# %%
fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.4, 5))
ax.plot(thetas, pattern, "b-", lw=1.2, label="meep-torch BOR n2f")
ax.plot(thetas, np.sin(thetas) ** 2, "r--", lw=1.0, label="sin^2 theta")
ax.plot(-thetas, pattern, "b-", lw=1.2)
ax.plot(-thetas, np.sin(thetas) ** 2, "r--", lw=1.0)
ax.set_theta_zero_location("N")
ax.set_title("on-axis Ez dipole radiation pattern (m = 0)")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "plot_radiation_pattern_dipole.png", dpi=150)
plt.close(fig)
rms = float(np.sqrt(np.mean((pattern - np.sin(thetas) ** 2) ** 2)))
print(f"plot_radiation_pattern_dipole.png  rms vs sin^2 = {rms:.4f}")
