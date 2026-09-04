# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Extraction efficiency of a dipole in a high-index slab
#
# A vertical (Ez, m = 0) point dipole sits inside an n = 2.4 slab of thickness
# 0.7 lambda / n. Extraction efficiency is the flux escaping through a plane
# above the slab divided by the total emitted power (LDOS), swept over three
# dipole depths.

# %%
"""meep-torch port of meep/python/examples/point_dipole_cyl.py.

Extraction efficiency of a point dipole in a high-index slab (n = 2.4,
thickness 0.7 lambda / n) via cylindrical coordinates: radiated flux
escaping through the top z-plane vs the total emitted power (LDOS).

SUBSTITUTIONS:
  * mp cylindrical Simulation + flux monitors -> core stepper_bor DFT
    planes; top-plane Poynting flux from the (Er, Hphi) phasors.
  * The meep example sweeps horizontal-dipole positions (m-summed); this
    port ships the on-axis VERTICAL (Ez, m = 0) dipole at three slab
    depths -- the m-sum for horizontal dipoles follows the W7.9 pattern.
  * Resolution 25 (meep's 50).

Run: uv run python examples/09_cylindrical/point_dipole_cyl.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.ldos import ldos_spectrum
from meep_torch.core.near2far_cyl import run_bor_dft
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bor import build_bor_coeffs, init_bor_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Per-depth BOR run: LDOS total vs escaping flux
#
# One run per dipole depth: the stepping loop accumulates DFT phasors of
# (Er, Hphi) for the Poynting flux through a plane above the slab, while the
# source-point Ez series gives the total emitted power via the LDOS overlap.

# %%
FCEN, RES = 1.0, 25
N_SLAB = 2.4
H_SLAB = 0.7 / N_SLAB


def extraction(zfrac):
    dr = 1.0 / RES
    nr, nz = int(3.0 * RES), int(6.0 * RES)
    eps = torch.ones(nr, nz, dtype=torch.float64)
    jz0 = int((3.0 - H_SLAB / 2) * RES)
    jz1 = int((3.0 + H_SLAB / 2) * RES)
    eps[:, jz0:jz1] = N_SLAB ** 2
    c = build_bor_coeffs(nr, nz, dr, m=0, eps=eps, n_abs_r=RES, n_abs_z=RES)
    j_src = jz0 + max(1, int(zfrac * (jz1 - jz0)))
    src = ("Ez", 0, j_src)
    pulse = GaussianPulse(FCEN, 0.2)
    n_steps = int(round(80.0 / c.dt))

    # run once collecting DFT planes AND the source-point Ez series
    from meep_torch.core.stepper_bor import step_bor
    state = init_bor_state(nr, nz)
    e_series = np.empty(n_steps)
    acc = None
    for t in range(n_steps):
        state = step_bor(state, c, pulse.amplitude((t + 0.5) * c.dt), src)
        e_series[t] = state.Ez[0, j_src].real.item()
        ph_e = np.exp(-2j * np.pi * FCEN * (t + 1.0) * c.dt) * c.dt
        ph_h = np.exp(-2j * np.pi * FCEN * (t + 0.5) * c.dt) * c.dt
        er = state.Er.cpu().numpy() * ph_e
        hphi = state.Hphi.cpu().numpy() * ph_h
        acc = (er, hphi) if acc is None else (acc[0] + er, acc[1] + hphi)

    # total emitted ~ LDOS overlap; escaping ~ +z Poynting through a plane
    # above the slab: S_z = -Re(Er conj(Hphi))? sign pinned by positivity
    j_series = np.array([pulse.amplitude((t + 0.5) * c.dt) for t in range(n_steps)])
    total = ldos_spectrum(e_series, j_series, c.dt, [FCEN])[0]
    j_plane = jz1 + int(0.5 * RES)
    er_p = acc[0][:, j_plane]
    hphi_p = 0.5 * (acc[1][:, j_plane - 1] + acc[1][:, j_plane])
    r = (np.arange(nr - 1) + 0.5) * dr
    sz = np.real(er_p * np.conj(hphi_p)) * r
    flux_top = abs(2 * np.pi * dr * sz.sum())
    return flux_top, total


# %% [markdown]
# ## Depth sweep and plot
#
# Relative extraction vs dipole depth, normalized to the best of the three
# positions.

# %%
fracs = [0.25, 0.5, 0.75]
rows = []
for f in fracs:
    fl, tot = extraction(f)
    rows.append((f, fl, tot, fl / tot))
ext_max = max(r[3] for r in rows)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.plot([r[0] for r in rows], [r[3] / ext_max for r in rows], "bo-")
ax.set_xlabel("dipole depth (fraction of slab)")
ax.set_ylabel("relative extraction (norm. to max)")
ax.set_title(f"vertical dipole in n={N_SLAB} slab (m=0)")
fig.tight_layout()
fig.savefig(OUT / "point_dipole_cyl.png", dpi=150)
plt.close(fig)
for f, fl, tot, ext in rows:
    print(f"point_dipole_cyl  zfrac={f}: top-flux/total = {ext:.3e}")
