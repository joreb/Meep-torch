# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Off-axis vertical dipole in vacuum (per-m sum)
#
# A point Ez dipole at r0 > 0 breaks the rotational symmetry, so it decomposes
# into e^{i m phi} azimuthal harmonics with equal weight. The far field is the
# coherent per-m sum at phi = 0, which must recover the plain sin^2(theta)
# dipole pattern (translation only adds a phase).

# %%
"""meep-torch port of meep/python/examples/dipole_in_vacuum_cyl_off_axis.py.

Radiation pattern of an OFF-AXIS vertical (Ez) point dipole in vacuum:
a point source at r0 > 0 decomposes into e^{i m phi} harmonics with
equal weight; the far field is the coherent per-m sum at phi = 0
(m and -m contribute symmetrically for an Ez dipole -> factor 2 for
m > 0). Summing m = 0..3 converges for k r0 ~ 2.

SUBSTITUTIONS: mp cylindrical per-m loop + get_farfield -> stepper_bor
runs per m + farfield_bor_m, coherent sum. Reduced res 25, m <= 3.

Run: uv run python examples/09_cylindrical/dipole_in_vacuum_cyl_off_axis.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from meep_torch.core.near2far_cyl import run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## One BOR run per azimuthal number m
#
# Each m = 0..3 gets its own radial simulation; m > 0 terms carry weight 2
# because +m and -m contribute symmetrically for an Ez dipole. Summing to
# m = 3 converges for k*r0 ~ 2.

# %%
FCEN, RES, R0 = 1.0, 25, 0.3

thetas = np.radians(np.linspace(5, 175, 90))
E_sum = np.zeros(len(thetas), dtype=complex)
dr = 1.0 / RES
nr, nz = int(3.0 * RES), int(6.0 * RES)
from meep_torch.core.near2far_cyl import farfield_bor_m_complex

for m in range(0, 4):
    c = build_bor_coeffs(nr, nz, dr, m=m, n_abs_r=RES, n_abs_z=RES)
    src = ("Ez", int(R0 * RES), nz // 2)
    _, planes = run_bor_dft(init_bor_state(nr, nz), c,
                            int(round(60.0 / c.dt)),
                            src=src, src_profile=GaussianPulse(FCEN, 0.2),
                            freqs=[FCEN])
    Et, _ = farfield_bor_m_complex(planes, c, m, ir=int(1.5 * RES),
                                   jz0=int(1.5 * RES), jz1=int(4.5 * RES),
                                   thetas=thetas, k=2 * math.pi * FCEN)
    w = 1.0 if m == 0 else 2.0                    # +-m symmetric for an Ez dipole
    E_sum += w * Et[0]

# %% [markdown]
# ## Coherent sum vs sin^2(theta)

# %%
pattern = np.abs(E_sum) ** 2
pattern /= pattern.max()
# analytic: off-axis vertical dipole pattern is still sin^2(theta) in the
# far field (translation only adds a phase); the per-m sum must recover it
analytic = np.sin(thetas) ** 2

fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5.2, 5))
ax.plot(thetas, pattern, "b-", lw=1.2, label="per-m sum (m<=3)")
ax.plot(thetas, analytic, "r--", lw=1.0, label="sin^2 theta")
ax.plot(-thetas, pattern, "b-", lw=1.2)
ax.plot(-thetas, analytic, "r--", lw=1.0)
ax.set_theta_zero_location("N")
ax.set_title(f"off-axis Ez dipole (r0={R0}), coherent m-sum")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "dipole_in_vacuum_cyl_off_axis.png", dpi=150)
plt.close(fig)
rms = float(np.sqrt(np.mean((pattern - analytic) ** 2)))
print(f"dipole_in_vacuum_cyl_off_axis.png  rms vs sin^2 = {rms:.4f}")
