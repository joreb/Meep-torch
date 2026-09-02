# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Ring resonator in cylindrical coordinates
#
# Whispering-gallery modes of a ring resonator computed on a 1D radial grid:
# fields are F(r) e^{i m phi}, so each azimuthal number m is an independent
# radial (BOR) simulation. Resonant frequencies and Q come from harminv on the
# complex probe ringdown.

# %%
"""meep-torch port of meep/python/examples/ring-cyl.py.

Ring-resonator modes in cylindrical coordinates: fields F(r) e^{i m phi}
on a 1D radial grid (W7 radial BOR stepper), harminv resonances per
azimuthal number m.

SUBSTITUTIONS:
  * mp.Simulation(dimensions=CYLINDRICAL, m=m) -> core stepper_cyl.
  * mp.Harminv -> modes.harminv on the complex probe ringdown (W3 FDM).
  * mp.PML(32) -> matched sigma absorber of the same depth (W5).
  * Node eps edge-averaged: reproduces meep's subpixel resonances.

Run: uv run python examples/09_cylindrical/ring-cyl.py
"""
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_cyl import (build_cyl_coeffs, init_cyl_state,
                                         run_cyl)
from meep_torch.modes.harminv import harminv

OUT = pathlib.Path(__file__).parent / "output"

# %% [markdown]
# ## Radial eps profile
#
# The ring (n=3.4, width 1) occupies r in [1, 2] on the radial grid; the two
# boundary cells are edge-averaged, which reproduces meep's subpixel-smoothed
# resonances.

# %%
N_IDX, W, R, PAD, DPML = 3.4, 1.0, 1.0, 4.0, 32.0
RES = 20
FCEN, DF = 0.15, 0.1

dr = 1.0 / RES
sr = R + W + PAD + DPML
nr = int(sr * RES)
eps = torch.ones(nr, dtype=torch.float64)
i0, i1 = int(R * RES), int((R + W) * RES)
eps[i0 + 1: i1] = N_IDX ** 2
eps[i0] = eps[i1] = (1.0 + N_IDX ** 2) / 2.0

# %% [markdown]
# ## Per-m ringdown and resonance extraction
#
# For m = 3, 4, 5 a Gaussian pulse excites the ring; harminv on the ringdown
# tail yields (f, Q) pairs, and the FFT spectra are stacked with one offset
# per m in the figure.

# %%
rows = []
fig, ax = plt.subplots(figsize=(7, 4.2))
for m in (3, 4, 5):
    c = build_cyl_coeffs(nr, dr, m, eps=eps, n_absorber=int(DPML * RES))
    src_ix = int((R + 0.1) * RES)
    _, series = run_cyl(init_cyl_state(nr), c, int(round(260.0 / c.dt)),
                        src_ix=src_ix, src_profile=GaussianPulse(FCEN, DF),
                        probe_ix=src_ix)
    tail = series[int(round(60.0 / c.dt)):]
    modes = [md for md in harminv(tail, c.dt, FCEN - DF, FCEN + DF)
             if md.Q > 10]                       # drop source-tail residuals
    for md in modes[:2]:
        rows.append((m, md.freq, md.Q))
    spec = np.abs(np.fft.rfft(tail.real))
    fr = np.fft.rfftfreq(len(tail), c.dt)
    band = (fr > FCEN - DF) & (fr < FCEN + DF)
    ax.plot(fr[band], spec[band] / spec[band].max() + (m - 3), lw=0.8,
            label=f"m={m}")
ax.set_xlabel("frequency")
ax.set_ylabel("|Ez(f)| (offset per m)")
ax.set_title("ring-cyl whispering-gallery resonances")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "ring-cyl.png", dpi=150)
plt.close(fig)
for m, f, q in rows:
    print(f"ring-cyl  m={m}: f={f:.5f}  Q={q:.1f}")
