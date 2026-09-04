# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Bend flux via the meep-syntax shim
#
# meep-syntax twin of bend-flux.py via `meep_torch.mp`. Transmission around a
# 90-degree waveguide bend via the meep-syntax S-param path: an EigenModeSource
# at the input port + mode monitors at the input (reflection) and output
# (transmission). The shim routes this to compute_smatrix and returns
# power-normalized S-parameters, matching bend-flux.py (R=|S11|^2, T=|S21|^2)
# within a few 1e-3. See Simulation._run_modes for the measured-incident a_in /
# meep-alpha normalization mapping. Output: `output/bend-flux_mp.png`.
#
# Run: `uv run python examples/01_basics/bend-flux_mp.py`

# %%
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
fcen, fwidth = 0.15, 0.1
freqs = np.linspace(0.105, 0.195, 25)

# %% [markdown]
# ## Simulation setup
# Same two-Block bend geometry as the raw twin, but driven by an
# EigenModeSource with mode monitors as the ports.

# %%
sim = mp.Simulation(
    cell_size=mp.Vector3(16, 32, 0),
    resolution=32,
    geometry=[
        mp.Block(size=mp.Vector3(12, 1, mp.inf), center=mp.Vector3(-2, -11.5, 0),
                 material=mp.Medium(epsilon=12)),
        mp.Block(size=mp.Vector3(1, 28, mp.inf), center=mp.Vector3(3.5, 2, 0),
                 material=mp.Medium(epsilon=12)),
    ],
    sources=[mp.EigenModeSource(mp.GaussianSource(frequency=fcen, fwidth=fwidth),
              component=mp.Ez, center=mp.Vector3(-6, -11.5, 0),
              size=mp.Vector3(0, 4, 0), direction=1)],
    boundary_layers=[mp.PML(1.0)],
    default_material=mp.Medium(epsilon=1.0),
)

# %% [markdown]
# ## Mode monitors and run
# Input monitor at the source plane measures reflection; output monitor on the
# vertical arm measures transmission.
#
# The stop is physical time, not a step count: dt = Courant/resolution, so a
# raw `n_steps` has to be rescaled by hand every time the resolution moves
# (at resolution 32, n_steps=50000 is t=781 -- about 4x longer than needed).
# `until_after_sources` counts from when the sources turn off, so it is
# resolution-independent. 400 is past convergence: R/T are identical to four
# decimals at 400, 600 and 700 (upstream bend-flux.py uses the adaptive
# stop_when_fields_decayed, which the S-param path does not accept -- see
# Simulation.run, step functions are 2D non-eigenmode only).

# %%
m_in = sim.add_mode_monitor(fcen, freqs[1] - freqs[0], len(freqs),
                            mp.FluxRegion(center=mp.Vector3(-6, -11.5, 0), size=mp.Vector3(0, 4, 0)))
m_out = sim.add_mode_monitor(fcen, freqs[1] - freqs[0], len(freqs),
                            mp.FluxRegion(center=mp.Vector3(3.5, 12, 0),
                                          size=mp.Vector3(4, 0, 0), direction=-1))
sim.run(until_after_sources=400)
R = np.abs(sim.get_eigenmode_coefficients(m_in, [1]).alpha[0, :, 0]) ** 2
T = np.abs(sim.get_eigenmode_coefficients(m_out, [1]).alpha[0, :, 0]) ** 2
wl = 1.0 / freqs

# %% [markdown]
# ## Plot

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wl, R, "bo-", label="reflectance |S11|^2", ms=3)
ax.plot(wl, T, "ro-", label="transmittance |S21|^2", ms=3)
ax.plot(wl, 1 - R - T, "go-", label="loss 1-R-T", ms=3)
ax.set_xlabel("wavelength (um)"); ax.set_ylabel("power")
ax.set_title("90-degree bend R/T (mp shim)"); ax.legend(); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(OUT / "bend-flux_mp.png", dpi=150); plt.close(fig)
i = int(np.argmin(np.abs(freqs - fcen)))
print("bend-flux_mp.png  at fcen=%.3f: R=%.3f T=%.3f loss=%.3f" % (fcen, R[i], T[i], 1 - R[i] - T[i]))
