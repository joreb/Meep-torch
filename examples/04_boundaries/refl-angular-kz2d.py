# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Out-of-plane angular reflectance (kz stepper)
#
# meep-torch port of meep/python/examples/refl-angular-kz2d.py. Angular
# reflectance of a planar air/glass interface with the incidence plane OUT of
# the 2D simulation plane: fields F(x) e^{i k_z z} via the kz stepper (W2
# deferral close). p-polarization (E in the (x, z) plane); the sweep runs
# 0..80 deg and should trace the Fresnel p curve through the Brewster zero.
# Output: `output/refl-angular-kz2d.png`.
#
# SUBSTITUTIONS:
# * mp.Simulation(kz_2d="complex") -> core stepper_kz (1D-in-x reduction).
# * Flux-normalized R -> reflected/incident Ez DFT amplitude ratio
#   (|Ez_r / Ez_i| = |r_p| since Ez/|E| = cos theta on both sides).
# * matched-sigma absorber -> kz-aware CFS-CPML on the x-derivatives
#   (npml=50, pml_alpha=0.4). A matched-sigma absorber reflects near
#   grazing because the x wavelength 2 pi/(w cos th) outgrows the graded
#   layer and the onset reads as a hard step (it capped at 65 deg). The
#   CPML is impedance-matched at every angle, so the sweep runs 0..80 deg.
#
# Run: `uv run python examples/04_boundaries/refl-angular-kz2d.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_kz import build_kz_coeffs, init_kz_state, step_kz

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
FCEN, N2 = 1.0, 1.5
NX, DX, NPML = 800, 0.05, 50

# %% [markdown]
# ## Per-angle reflectance
# Two kz runs (with/without the glass half-space); |r|^2 comes from the DFT
# amplitude ratio of the reflected and incident Ez at a fixed probe.

# %%
def refl(theta):
    kz = 2 * math.pi * FCEN * math.sin(theta)

    def run(with_glass):
        eps = torch.ones(NX, dtype=torch.float64)
        if with_glass:
            eps[400:] = N2 ** 2
        c = build_kz_coeffs(NX, DX, kz, eps=eps, npml=NPML,
                            pml_m=3, pml_R0=1e-10, pml_alpha=0.4)
        st = init_kz_state(NX, npml=NPML)
        pulse = GaussianPulse(FCEN, 0.2)
        # x-group velocity ~ cos(theta): scale the run so the pulse makes
        # the round trip even near grazing
        n = int(round(60.0 / max(math.cos(theta), 0.15) / c.dt))
        s = np.empty(n, dtype=complex)
        for t in range(n):
            st = step_kz(st, c, pulse.amplitude((t + 0.5) * c.dt), 100)
            s[t] = st.Ez[300].item()
        return s, c.dt

    s1, dt = run(True)
    s0, _ = run(False)
    t = np.arange(len(s1)) * dt
    kern = np.exp(-2j * math.pi * FCEN * t)
    return abs(np.trapezoid((s1 - s0) * kern, t)
               / np.trapezoid(s0 * kern, t)) ** 2

# %% [markdown]
# ## Sweep, Fresnel overlay and plot
# 0..80 deg in 5-deg steps with the Brewster angle marked; the console prints
# the max deviation from Fresnel.

# %%
# kz-aware CFS-CPML: p-pol Fresnel sweep 0..80 deg (was capped at 65 deg
# with the matched-sigma absorber, which reflects near grazing).
thetas = np.radians(np.arange(0, 85, 5))
R = np.array([refl(t) for t in thetas])
cos_i = np.cos(thetas)
cos_t = np.sqrt(1 - (np.sin(thetas) / N2) ** 2)
R_an = ((N2 * cos_i - cos_t) / (N2 * cos_i + cos_t)) ** 2

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(np.degrees(thetas), R, "bo-", ms=4, lw=0.9, label="meep-torch kz stepper")
ax.plot(np.degrees(thetas), R_an, "r--", lw=1.0, label="Fresnel p-pol")
brewster = math.degrees(math.atan(N2))
ax.axvline(brewster, color="g", ls=":", lw=0.8, label=f"Brewster {brewster:.1f} deg")
ax.set_xlabel("incidence angle (deg, out of plane)")
ax.set_ylabel("reflectance R")
ax.set_title("air/glass, p-pol, out-of-plane incidence (0-80 deg, kz-CPML)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "refl-angular-kz2d.png", dpi=150)
plt.close(fig)
err = float(np.abs(R - R_an).max())
print(f"refl-angular-kz2d.png  max |R - Fresnel| = {err:.4f} over 0..80 deg (kz-CPML)")
