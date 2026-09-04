# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Angular reflectance of a dielectric interface
#
# meep-torch port of meep/python/examples/refl-angular.py. Angular reflectance
# of a vacuum->dielectric (n=2) interface at oblique incidence
# (Bloch-periodic y with k_y = 2*pi*f*sin(theta)), s-polarization (2D TM Ez).
# R = 1 - T from the pulse-integrated transmitted flux vs a vacuum reference.
# Gate: R(theta) vs Fresnel s within 2%
# (tests/examples/test_w2_examples.py). Output: `output/refl-angular.png`.
#
# Run: `uv run python examples/04_boundaries/refl-angular.py`

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
from meep_torch.core.stepper_bloch import bloch_xflux, init_bloch_tm_state, step_bloch_tm

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, N2, DX = 0.3, 2.0, 0.04
THETAS = [10.0, 20.0, 30.0, 40.0]

# %% [markdown]
# ## Per-angle reflectance
# For each theta, the cell height is one Bloch period; two runs (vacuum
# reference + interface) give T from the flux ratio and R = 1 - T.

# %%
def _R(theta_deg):
    theta = math.radians(theta_deg)
    k0, ky = 2 * math.pi * FCEN, 2 * math.pi * FCEN * math.sin(theta)
    Ly = (2 * math.pi / ky) if ky else 4.0
    ny = max(20, int(round(Ly / DX)))
    nx, npml = 260, 18
    g = Grid(nx=nx, ny=ny, dx=DX)
    ix_if, ix_src, ix_mon = nx // 2, npml + 5, nx - npml - 15
    w, t0, omega = 1.0 / (0.5 * FCEN), 0.0, k0
    t0 = 6.0 * w

    def run(diel):
        eps = torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device)
        if diel:
            eps[ix_if:, :] = N2 * N2
        c = build_tm_coeffs(g, eps_inf=eps, npml=npml, y_pmc=True)
        st = init_bloch_tm_state(g)
        mask = g.zeros(g.nx, g.ny); mask[ix_src, :] = 1.0
        fs = 0.0
        for t in range(int(round(80.0 / g.dt))):
            tt = (t + 0.5) * g.dt
            genv = math.exp(-((tt - t0) ** 2) / (2 * w * w))
            st = step_bloch_tm(st, c, ky, src_c=genv * math.cos(omega * tt),
                               src_s=-genv * math.sin(omega * tt), src_mask=mask)
            fs += bloch_xflux(st, ix_mon, DX)
        return fs
    ref, slab = run(False), run(True)
    return 1.0 - slab / ref

# %% [markdown]
# ## Fresnel reference, sweep and plot

# %%
def _fresnel_s(theta_deg, n1=1.0, n2=N2):
    th = math.radians(theta_deg)
    s2 = (n1 / n2) * math.sin(th)
    c2 = math.sqrt(max(0.0, 1 - s2 * s2))
    r = (n1 * math.cos(th) - n2 * c2) / (n1 * math.cos(th) + n2 * c2)
    return r * r


R = [_R(th) for th in THETAS]
Ra = [_fresnel_s(th) for th in THETAS]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(THETAS, R, "bo-", label="meep-torch (Bloch)")
ax.plot(THETAS, Ra, "rs-", label="Fresnel s")
ax.set_xlabel("theta (deg)"); ax.set_ylabel("reflectance")
ax.set_title("Angular reflectance: vacuum -> dielectric n=2 (refl-angular)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "refl-angular.png", dpi=150); plt.close(fig)
print("refl-angular.png  R=", ["%.3f" % r for r in R], " Fresnel=", ["%.3f" % r for r in Ra])
