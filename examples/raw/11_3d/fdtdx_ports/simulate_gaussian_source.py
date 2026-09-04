# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Exact time reversal of a CW source (fdtdx port)
#
# A CW source at 1550 nm radiates in a fully periodic, lossless cell, then
# the simulation runs *backward* to t = 0. The leapfrog update is symplectic,
# so the reversal is exact to round-off — the mechanism fdtdx exploits for
# O(1)-memory gradients (meep-torch gets the same via checkpointed autograd).
# Outputs forward/backward videos and the reversibility residual.

# %%
"""meep-torch rebuild of fdtdx/examples/simulate_gaussian_source.py.

A CW source (1550 nm) radiates in a fully periodic volume; the
simulation then runs BACKWARD in time to t = 0. The leapfrog update is
symplectic and the periodic cell has no loss, so time reversal is exact
to round-off -- the mechanism fdtdx exploits for O(1)-memory gradients
(meep-torch gets the same via checkpointed reverse-mode autograd).
Outputs: forward + backward videos and the reversibility residual.

Units: um / (um/c) internally; annotations in SI.
Run: uv run python examples/11_3d/fdtdx_ports/simulate_gaussian_source.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.stepper_bloch2 import (Bloch2State, init_bloch2_state,
                                            step_bloch2_tm)

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
WL = 1.55                                          # 1550 nm
RES = 25                                           # 39 px per wavelength
dx = 1.0 / RES
n = int(12.0 * RES)                                # 12 um periodic box


class _G:
    nx = ny = n
    zeros = staticmethod(lambda *s: torch.zeros(*s, dtype=torch.float64))


eps = torch.ones(n, n, dtype=torch.float64)
dt = 0.4 * dx
mask = torch.zeros(n, n, dtype=torch.float64)
mask[n // 2, n // 2 - 3 * RES: n // 2 + 3 * RES] = 1.0   # 6 um line source
omega = 2 * math.pi / WL

# %% [markdown]
# ## Forward run
#
# The ramped CW line source is stepped 30 time units on the periodic cell;
# source amplitudes are recorded so the reversal can undo them exactly.

# %%
st = init_bloch2_state(_G)
N_FWD = int(round(30.0 / dt))
every = max(1, N_FWD // 150)
fwd_frames = []
amps = []                                          # source history for reversal
for t in range(N_FWD):
    a = math.sin(omega * (t + 0.5) * dt) * min(1.0, (t + 0.5) * dt / 5.0)
    amps.append(a)
    st = step_bloch2_tm(st, eps, dt, dx, 0.0, 0.0, src_c=a, src_s=0.0,
                        src_mask=mask)
    if t % every == 0:
        fwd_frames.append(st.Ez_c.cpu().numpy().copy())

# %% [markdown]
# ## Backward run and residual
#
# The leapfrog is inverted step by step (E update undone first, then H, with
# the recorded source subtracted); |Ez(t=0)| after the round trip is the
# reversibility residual.

# %%
# --- exact time reversal: leapfrog inverted (E first, then H), source undone ---
bwd_frames = []
for t in range(N_FWD - 1, -1, -1):
    # invert the E/D update: Dz(n) = Dz(n+1) - dt*(curl H) - dt*src
    from meep_torch.core.stepper_bloch2 import _bwd, _mid_bwd
    cx_c = _bwd(st.Hy_c, 0, dx)
    cy_c = _bwd(st.Hx_c, 1, dx)
    Dz_c = st.Dz_c - dt * (cx_c - cy_c) - dt * amps[t] * mask
    Ez_c = Dz_c / eps
    # invert the H update: H(n-1/2) = H(n+1/2) +/- dt * curl E(n)
    from meep_torch.core.stepper_bloch2 import _fwd
    Hx_c = st.Hx_c + dt * _fwd(Ez_c, 1, dx)
    Hy_c = st.Hy_c - dt * _fwd(Ez_c, 0, dx)
    z = torch.zeros_like(Ez_c)
    st = Bloch2State(Ez_c, z, Hx_c, z, Hy_c, z, Dz_c, z)
    if t % every == 0:
        bwd_frames.append(Ez_c.cpu().numpy().copy())

residual = float(st.Ez_c.abs().max())
print(f"simulate_gaussian_source  reversibility residual |Ez(t=0)| = "
      f"{residual:.3e} (exact reversal to round-off)")

for name, frames in (("forward", fwd_frames), ("backward", bwd_frames)):
    vmax = max(np.abs(f).max() for f in frames) * 0.5 or 1.0
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(frames[0].T, origin="lower", cmap="RdBu",
                   vmin=-vmax, vmax=vmax)
    ax.set_title(f"CW 1550 nm, periodic cell -- {name} time")
    ax.set_xticks([]); ax.set_yticks([])

    def upd(i, im=im, frames=frames):
        im.set_data(frames[i].T)
        return (im,)

    ani = anim.FuncAnimation(fig, upd, frames=len(frames), interval=40)
    ani.save(OUT / f"simulate_gaussian_source_{name}.mp4",
             writer=anim.FFMpegWriter(fps=25))
    plt.close(fig)
print("simulate_gaussian_source_{forward,backward}.mp4 written")
