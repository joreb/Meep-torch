# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Scattering cross-section of a dielectric cylinder
#
# meep-torch port of meep/python/examples/cylinder_cross_section.py.
# Scattering cross-section of a dielectric cylinder (2D TM, E parallel to the
# axis): two-run scattered-field method (DFT phasors subtract linearly),
# scattered power through a closed box -> Q_sca vs the analytic 2D Mie series
# (core/mie2d). Output: `output/cylinder_cross_section.png`.
#
# SUBSTITUTIONS: meep's FINITE-height 3D cylinder -> infinite 2D cylinder
# (the analytic-gated case). True plane wave via a periodic-y Bloch cell
# (ky=0; a finite-aperture line source curves the wavefront -- 15-30%
# Q errors); the periodic cylinder copies at 12 um pitch perturb the
# isolated-cylinder sigma at the few-% level (noted).
#
# Run: `uv run python examples/05_monitors_analysis/cylinder_cross_section.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.mie2d import q_sca
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_bloch import init_bloch_tm_state, step_bloch_tm

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
N_CYL, R_CYL, RES = 2.0, 0.7, 20
# ponytail: f <= 0.8 gates the method (<2%); the x = 2 pi f a > 4 region
# reads 25%+ high -- periodic-copy interference (pitch 12) + resonance
# proximity; isolated-cylinder accuracy there needs a much taller cell
FREQS = [0.6, 0.8, 1.1]

# %% [markdown]
# ## DFT field planes per run
# One Bloch (ky=0) run per frequency, with and without the cylinder; the
# supersampled eps gives sub-pixel averaging on the circle, and the run
# accumulates single-frequency DFT phasors of Ez, Hx, Hy everywhere.

# %%
def dft_planes(with_cyl, fcen):
    dx = 1.0 / RES
    nx, ny = int(10.0 * RES), int(12.0 * RES)     # periodic y, pitch 12
    grid = Grid(nx=nx, ny=ny, dx=dx)
    ss = 4
    xi = (torch.arange(nx * ss, dtype=torch.float64) + 0.5) / (RES * ss)
    yi = (torch.arange(ny * ss, dtype=torch.float64) + 0.5) / (RES * ss)
    II, JJ = torch.meshgrid(xi, yi, indexing="ij")
    fine = torch.ones(nx * ss, ny * ss, dtype=torch.float64)
    if with_cyl:
        fine[(II - 5.0) ** 2 + (JJ - 6.0) ** 2 < R_CYL ** 2] = N_CYL ** 2
    eps = fine.reshape(nx, ss, ny, ss).mean(dim=(1, 3))
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=RES, y_pmc=True)
    st = init_bloch_tm_state(grid)
    mask = grid.zeros(nx, ny)
    mask[int(2.0 * RES), :] = 1.0                  # uniform -> exact plane wave
    pulse = GaussianPulse(fcen, 0.2)
    n_steps = int(round(60.0 / grid.dt))
    acc = {c: 0.0 for c in ("Ez", "Hx", "Hy")}
    for t in range(n_steps):
        a = pulse.amplitude((t + 0.5) * grid.dt)
        st = step_bloch_tm(st, coeffs, 0.0, src_c=a, src_s=0.0, src_mask=mask)
        phE = np.exp(-2j * np.pi * fcen * (t + 1.0) * grid.dt) * grid.dt
        phH = np.exp(-2j * np.pi * fcen * (t + 0.5) * grid.dt) * grid.dt
        acc["Ez"] = acc["Ez"] + (st.Ez_c + 1j * st.Ez_s).numpy() * phE
        acc["Hx"] = acc["Hx"] + (st.Hx_c + 1j * st.Hx_s).numpy() * phH
        acc["Hy"] = acc["Hy"] + (st.Hy_c + 1j * st.Hy_s).numpy() * phH
    return acc, grid

# %% [markdown]
# ## Scattered-field flux box
# Subtracting the empty-cell phasors leaves the scattered field; its net
# outward Poynting flux through a closed box is the scattered power.

# %%
# narrowband run per frequency: band-edge SNR + aperture diffraction ruin
# a single broadband pair (center 0.5%, edges 27-65% off)
dx = 1.0 / RES


def flux_box(planes, ix0, ix1, iy0, iy1):
    """Net outward Poynting of the plane set through the index box."""
    tot = 0.0
    ez, hx, hy = planes["Ez"], planes["Hx"], planes["Hy"]
    for idx, sgn, axis in ((ix1, +1, "x"), (ix0, -1, "x"),
                           (iy1, +1, "y"), (iy0, -1, "y")):
        if axis == "x":
            j = np.arange(iy0, iy1)
            e = ez[idx, j]
            h = 0.5 * (hy[idx - 1, j] + hy[idx, j])
            s = -0.5 * np.real(e * np.conj(h))     # S_x = -Ez Hy
        else:
            i = np.arange(ix0, ix1)
            e = ez[i, idx]
            h = hx[i, idx]                         # Bloch Hx is full-shape
            s = 0.5 * np.real(e * np.conj(h))      # S_y = +Ez Hx
        tot += sgn * float(np.sum(s)) * dx
    return tot

# %% [markdown]
# ## Frequency sweep, Mie comparison and plot
# Q_sca = sigma / (2r) per frequency, against the analytic 2D Mie series; the
# console prints per-point deviations.

# %%
qs, qs_an = [], []
b = int(RES * 1.5)
for f in FREQS:
    p1, grid = dft_planes(True, f)
    p0, _ = dft_planes(False, f)
    scat = {c: p1[c] - p0[c] for c in p1}
    p_sc = flux_box(scat, int(5 * RES) - b, int(5 * RES) + b,
                    int(6 * RES) - b, int(6 * RES) + b)
    # exact plane wave: incident S_x uniform, sample the front face
    ii = int(5 * RES) - b
    j = np.arange(int(6 * RES) - b, int(6 * RES) + b)
    e0 = p0["Ez"][ii, j]
    h0 = 0.5 * (p0["Hy"][ii - 1, j] + p0["Hy"][ii, j])
    s_inc = abs(float(np.mean(-0.5 * np.real(e0 * np.conj(h0)))))
    sigma = p_sc / s_inc                           # scattering width (2D)
    qs.append(sigma / (2 * R_CYL))                 # Q = sigma / geometric width
    qs_an.append(q_sca(N_CYL, 2 * math.pi * f * R_CYL))

fig, ax = plt.subplots(figsize=(5.4, 3.8))
ax.plot(FREQS, qs, "bo-", label="meep-torch (scattered-field box)")
ax.plot(FREQS, qs_an, "rx--", ms=9, label="2D Mie analytic")
ax.set_xlabel("frequency")
ax.set_ylabel("$Q_{sca}$")
ax.set_title(f"cylinder n={N_CYL}, r={R_CYL} (TM)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "cylinder_cross_section.png", dpi=150)
plt.close(fig)
for f, q, qa in zip(FREQS, qs, qs_an):
    print(f"cylinder_cross_section  f={f}: Q={q:.3f} vs Mie {qa:.3f} "
          f"(dev {abs(q - qa) / qa:.1%})")
