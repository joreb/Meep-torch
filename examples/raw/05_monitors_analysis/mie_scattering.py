# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Mie scattering of a periodic cylinder array
#
# meep-torch port of meep/python/examples/mie_scattering.py. Scattering
# spectrum of a PERIODIC cylinder array (pitch 12; 2D TM substitute for
# meep's 3D isolated sphere): the wrapped-copy interference makes the
# near-box "Q" oscillate around the isolated-cylinder Mie curve -- real array
# physics, reproduced by the meep oracle to a few % -- gated against
# `tests/examples/data/cyl_array_mie.json`, with the isolated 2D Mie series
# overlaid as reference. Output: `output/mie_scattering.png`.
#
# Run: `uv run python examples/05_monitors_analysis/mie_scattering.py`

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
# oracle data lives in the repo's tests tree, two levels up from this script
DATA = pathlib.Path(__file__).parents[3] / "tests" / "examples" / "data"

N_CYL, R_CYL, RES = 2.0, 0.7, 20

# %% [markdown]
# ## DFT field planes per frequency
# One Bloch (ky=0, exact plane wave) run per frequency, with and without the
# cylinder; supersampled eps averaging on the circle.

# %%
def planes(with_cyl, fcen):
    dx = 1.0 / RES
    nx, ny = int(10.0 * RES), int(12.0 * RES)     # periodic y, pitch 12
    grid = Grid(nx=nx, ny=ny, dx=dx)
    ss = 4
    xi = (torch.arange(nx * ss, dtype=torch.float64, device=grid.device) + 0.5) / (RES * ss)
    yi = (torch.arange(ny * ss, dtype=torch.float64, device=grid.device) + 0.5) / (RES * ss)
    II, JJ = torch.meshgrid(xi, yi, indexing="ij")
    fine = torch.ones(nx * ss, ny * ss, dtype=torch.float64, device=grid.device)
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
        acc["Ez"] = acc["Ez"] + (st.Ez_c + 1j * st.Ez_s).cpu().numpy() * phE
        acc["Hx"] = acc["Hx"] + (st.Hx_c + 1j * st.Hx_s).cpu().numpy() * phH
        acc["Hy"] = acc["Hy"] + (st.Hy_c + 1j * st.Hy_s).cpu().numpy() * phH
    return acc, grid

# %% [markdown]
# ## Q per frequency
# Scattered-field subtraction, closed-box Poynting flux, normalized by the
# incident flux sampled on the front face.

# %%
def q_at(f):
    p1, grid = planes(True, f)
    p0, _ = planes(False, f)
    sc = {c: p1[c] - p0[c] for c in p1}
    b = int(RES * 1.5)
    cx, cy = int(5 * RES), int(6 * RES)
    dx = 1.0 / RES
    tot = 0.0
    for idx, sgn, axis in ((cx + b, +1, "x"), (cx - b, -1, "x"),
                           (cy + b, +1, "y"), (cy - b, -1, "y")):
        if axis == "x":
            j = np.arange(cy - b, cy + b)
            h = 0.5 * (sc["Hy"][idx - 1, j] + sc["Hy"][idx, j])
            sflux = -0.5 * np.real(sc["Ez"][idx, j] * np.conj(h))
        else:
            i = np.arange(cx - b, cx + b)
            sflux = 0.5 * np.real(sc["Ez"][i, idx] * np.conj(sc["Hx"][i, idx]))
        tot += sgn * float(np.sum(sflux)) * dx
    ii = cx - b
    j = np.arange(cy - b, cy + b)
    e0 = p0["Ez"][ii, j]
    h0 = 0.5 * (p0["Hy"][ii - 1, j] + p0["Hy"][ii, j])
    s_inc = abs(float(np.mean(-0.5 * np.real(e0 * np.conj(h0)))))
    return tot / s_inc / (2 * R_CYL)

# %% [markdown]
# ## Oracle comparison and plot
# Sweep the oracle's frequencies, compare against the meep-oracle array Q and
# the isolated-cylinder Mie reference; console prints per-point deviations.

# %%
import json

ref = json.load(open(DATA / "cyl_array_mie.json"))
fs = ref["freqs"]
qs = [q_at(f) for f in fs]
qa = [q_sca(N_CYL, 2 * math.pi * f * R_CYL) for f in fs]
fig, ax = plt.subplots(figsize=(5.8, 3.9))
ax.plot(fs, qs, "bo-", label="meep-torch (array)")
ax.plot(fs, ref["Q"], "gs--", ms=6, label="meep oracle (array)")
ax.plot(fs, qa, "r:", label="isolated 2D Mie (reference)")
ax.set_xlabel("frequency")
ax.set_ylabel("$Q_{sca}$ (per cylinder)")
ax.set_title(f"cylinder array pitch 12, n={N_CYL}, r={R_CYL} (TM)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "mie_scattering.png", dpi=150)
plt.close(fig)
devs = [abs(q - qr) / qr for q, qr in zip(qs, ref["Q"])]
print(f"mie_scattering.png  max dev vs meep-oracle array: {max(devs):.1%} "
      f"(per-point {[f'{d:.1%}' for d in devs]})")
