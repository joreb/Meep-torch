# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Fused-quartz reflectance via the meep-syntax shim
#
# meep-syntax twin of refl-quartz.py via `meep_torch.mp`. 1D reflectance of a
# broadband Gaussian pulse off a fused-quartz (SiO2) slab, two-run (empty +
# slab) net-flux subtraction R = -(slab - empty)/empty, vs the analytic
# Fresnel reflectance of the meep-torch SiO2 Lorentzian. Matches
# refl-quartz.py fluxes (the raw twin has no load_minus_flux_data either).
# Output: `output/refl-quartz_mp.png`.
#
# Run: `uv run python examples/03_materials/refl-quartz_mp.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch.core.materials import SiO2, analytic_eps

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
wvl_min, wvl_max = 0.4, 0.8
fmin, fmax = 1.0 / wvl_max, 1.0 / wvl_min
fcen, dfreq, nfreq = 0.5 * (fmax + fmin), fmax - fmin, 50
freqs = np.linspace(fmin, fmax, nfreq)
sz, res = 12.0, 100

# %% [markdown]
# ## Two runs through the shim
# One Simulation builder used twice: empty (incident flux) and with the SiO2
# Block (net forward flux).

# %%
def run_sim(geometry=None):
    sim = mp.Simulation(
        cell_size=mp.Vector3(0, 0, sz),
        resolution=res,
        dimensions=1,
        geometry=geometry or [],
        sources=[mp.Source(mp.GaussianSource(frequency=fcen, fwidth=dfreq),
                           component=mp.Ex, center=mp.Vector3(z=-0.5 * sz + 1.0))],
        boundary_layers=[mp.PML(1.0)],
        default_material=mp.Medium(epsilon=1.0),
    )
    fr = mp.FluxRegion(center=mp.Vector3(z=-0.25 * sz))
    name = sim.add_flux(fcen, dfreq, nfreq, fr)
    sim.run(n_steps=int(round(80.0 / sim.grid.dt)))
    return np.array(sim.get_fluxes(name))


empty = run_sim()
slab = run_sim([mp.Block(size=mp.Vector3(mp.inf, mp.inf, 0.5 * sz),
                          center=mp.Vector3(z=0.25 * sz), material=mp.Medium(material=SiO2))])

# %% [markdown]
# ## Reflectance and plot

# %%
R = -(slab - empty) / empty
wvls = 1.0 / freqs
n_q = np.array([math.sqrt(analytic_eps(SiO2.eps_inf, SiO2.poles, f).real) for f in freqs])
R_analytic = np.abs((1.0 - n_q) / (1.0 + n_q)) ** 2

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wvls, R, "bo-", label="meep-torch (mp shim)")
ax.plot(wvls, R_analytic, "rs-", label="analytic (SiO2 Lorentzian)")
ax.set_xlabel("wavelength (um)"); ax.set_ylabel("reflectance")
ax.set_title("Fused-quartz slab reflectance (mp shim)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "refl-quartz_mp.png", dpi=150); plt.close(fig)
i = nfreq // 2
print("refl-quartz_mp.png  R@%.3fum  shim=%.5f  analytic=%.5f" % (wvls[i], R[i], R_analytic[i]))
