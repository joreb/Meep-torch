# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Hyperbolic (indefinite) medium via per-axis Drude dispersion
#
# A static negative permittivity is unconditionally unstable in explicit
# FDTD, so the metallic axis is modeled with a Drude pole that drives
# Re eps(f) negative in-band while eps_inf stays 1. A TE wave probes one
# tensor axis per configuration: pole on the propagating axis makes the slab
# metallic (T ~ 0), pole on the transverse axis leaves it transparent
# (T ~ 1).

# %%
"""meep-torch rebuild of fdtdx/examples/hyperbolic_dispersive_slab.py.

Hyperbolic (indefinite) medium via per-axis Drude dispersion: a static
negative permittivity is unconditionally unstable in explicit FDTD, so
the metallic axis is modeled with a Drude pole that drives Re eps(f)
negative in-band while eps_inf stays 1:

    eps_m(f) = 1 - sigma fp^2 / (f^2 + i gamma f)   (Drude axis)
    eps_d    = 1                                    (other axes)

A TE wave propagating +x probes one tensor axis per polarization: with
the pole on the y axis the propagating Ey sees the metal (evanescent,
T ~ 0); with the pole on the x axis Ey sees a dielectric (T ~ 1). The
transverse-axis pole is a strong LORENTZ oscillator, not Drude: a Drude
pole (fn=0) on a near-zero E component is a free-electron DC integrator
that noise-diverges (see tests/examples/test_peraxis_dispersion.py) --
the selectivity statement is identical. Prints both transmissions vs
the analytic permittivity; png: eps(f) band + field panels.

Run: uv run python examples/11_3d/fdtdx_ports/hyperbolic_dispersive_slab.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.coeffs import build_te_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTE
from meep_torch.core.run import run_te
from meep_torch.core.sources import GaussianPulse, SoftSource
from meep_torch.core.stepper_te import init_te_state

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
F0 = 0.5                                            # lambda = 2 um
FP, GAMMA = 5.0 * F0, 0.05 * F0                     # Drude, metallic for f < fp
DRUDE = (0.0, GAMMA, 1.0, FP)                       # (fn, gamma, sigma, fc=fp)
NX, NY, DX, NPML = 480, 200, 0.05, 20
grid = Grid(nx=NX, ny=NY, dx=DX)
SLAB0 = 260                                         # half-space -> interior end
SLAB1 = NX - NPML - 4                               # pole mask must avoid PML
SRC_IX, MON_IX = 80, SLAB0 + 40                     # monitor 2 um inside medium
J0, J1 = NPML + 4, NY - NPML - 4
T_END = 80.0
PULSE = GaussianPulse(F0, 0.1)                      # zero-DC (Drude-safe)
LORENTZ = (0.28 / 0.3 * F0, 0.05, 2.2)              # transverse-axis stand-in


def eps_drude(f):
    return 1.0 - DRUDE[2] * FP ** 2 / (f ** 2 + 1j * GAMMA * f)


# %% [markdown]
# ## Three runs: vacuum reference, metallic axis, transparent axis
#
# The pole mask stays inside the interior (a pole under PML damping feeds an
# undamped P into a damped E and diverges); flux 2 um inside the medium is
# normalized by the vacuum run.

# %%
def run_case(axis: str | None):
    kw = {}
    if axis is not None:
        # pole mask confined to the interior: a pole active where its E-axis
        # is PML-damped feeds an undamped P back into damped E -> divergence
        mask = torch.zeros(NX, NY, dtype=grid.dtype, device=grid.device)
        mask[SLAB0:SLAB1, NPML:NY - NPML] = 1.0
        pole = DRUDE if axis == "y" else LORENTZ
        kw = {f"poles_{axis}": [pole], f"pole_masks_{axis}": [mask]}
    coeffs = build_te_coeffs(grid, eps_inf=torch.ones(NX, NY, dtype=grid.dtype, device=grid.device),
                             npml=NPML, **kw)
    smask = grid.zeros(NX, NY)
    smask[SRC_IX, J0:J1] = 1.0
    src = SoftSource(grid, smask[:, :-1], PULSE)     # drives Ey (TE)
    mon = FluxMonitorTE(grid, [F0], ix=MON_IX, lo=NY // 2 - 40,
                        hi=NY // 2 + 40)   # central 4 um of the 8 um beam
    st, (acc,) = run_te(init_te_state(grid, n_poles=(0 if axis is None else 1),
                                      n_poles_y=(1 if axis == "y" else 0)),
                        coeffs, int(round(T_END / grid.dt)), source=src,
                        monitors=[mon])
    return float(mon.flux(acc, DX).item()), st.Hz.cpu().numpy().copy()


P0, _ = run_case(None)                              # vacuum reference
T_metal, hz_m = run_case("y")                       # Ey sees the Drude axis
T_pass, hz_p = run_case("x")                        # Ey sees dielectric axis
T_metal /= P0
T_pass /= P0
eps0 = eps_drude(F0)
print(f"hyperbolic_dispersive_slab  eps_metal(f0) = {eps0.real:.2f}"
      f"{eps0.imag:+.2f}j, eps_other = 1")
print(f"  pole on propagating axis (metal): T = {T_metal:.2e} (expect ~0, "
      f"skin depth {1 / (2 * math.pi * F0 * abs(np.sqrt(eps0).imag)):.3f} um "
      f"<< 2 um)")
print(f"  pole on transverse axis:          T = {T_pass:.3f} (expect ~1)")

# %% [markdown]
# ## eps(f) band diagram and field panels

# %%
fs = np.linspace(0.1 * F0, 1.5 * FP, 400)
ex = eps_drude(fs)
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
axs[0].plot(fs / F0, ex.real, label=r"Re $\epsilon$ (Drude axis)")
axs[0].plot(fs / F0, np.ones_like(fs), label=r"$\epsilon$ (other axes)")
axs[0].axhline(0, color="k", lw=0.8)
axs[0].axvline(1, color="gray", ls="--", lw=0.8, label="source f")
band = fs[(ex.real < 0)] / F0
axs[0].axvspan(band.min(), band.max(), alpha=0.12, color="tab:red",
               label="hyperbolic band")
axs[0].set_xlabel(r"$f/f_0$"); axs[0].set_ylabel(r"Re $\epsilon$")
axs[0].set_ylim(-30, 5); axs[0].legend(fontsize=7)
axs[0].set_title("indefinite tensor: per-axis Drude")
for ax, hz, ttl in ((axs[1], hz_m, f"pole on Ey axis: metallic, T={T_metal:.1e}"),
                    (axs[2], hz_p, f"pole on Ex axis: transparent, T={T_pass:.2f}")):
    v = np.abs(hz).max()
    ax.imshow(hz.T, origin="lower", cmap="RdBu", vmin=-v, vmax=v,
              extent=[0, NX * DX, 0, NY * DX], aspect="auto")
    ax.axvline(SLAB0 * DX, color="k", lw=0.8)
    ax.set_title(ttl, fontsize=9)
    ax.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "hyperbolic_dispersive_slab.png", dpi=150)
plt.close(fig)
print("hyperbolic_dispersive_slab.png written")
