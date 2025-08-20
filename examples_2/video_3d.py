# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (b): video of a TRUE 3D FDTD ring simulation -- a reduced
LiNbO3 microring (r = 2 um, 300 nm thin-film LN) driven by a point source in the
ring plane; mid-slab |Ey| slice animated -> mp4.

Reduced vs (a)'s 2D racetrack: full 3D at the r=5 gds geometry is out of
video budget; the torus ring shows the genuine 3D field circulating.

Run: uv run python examples_2/video_3d.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state, step_3d

OUT = pathlib.Path(__file__).parent
RES = 12
dx = 1.0 / RES
# 300 nm LN film (thin-film LN standard; a 150 nm slab cannot hold the
# r=2 bend -- pure radiation, verified) and 0.5 um guide width
R_RING, W_RING, H_SLAB = 2.0, 0.5, 0.30
N_LN = 2.2

nx = ny = int(6.0 * RES)
nz = int(1.6 * RES)
cx = cy = 3.0
cz = 0.8
ii = (torch.arange(nx, dtype=torch.float64) + 0.5) * dx
kk = (torch.arange(nz, dtype=torch.float64) + 0.5) * dx
X, Y, Z = torch.meshgrid(ii, ii, kk, indexing="ij")
r = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
eps = torch.ones(nx, ny, nz, dtype=torch.float64)
in_ring = (torch.abs(r - R_RING) < W_RING / 2) & (torch.abs(Z - cz) < H_SLAB / 2)
eps[in_ring] = N_LN ** 2

c3 = build_3d_coeffs(nx, ny, nz, dx, eps=eps, n_abs=RES // 2)
st = init_3d_state(nx, ny, nz)
F0 = 1 / 1.55
prof = ContinuousSource(F0, width=8.0)
src_i = int((cx + R_RING) * RES)
# guided TE mode at the +x point propagates along +-y: transverse in-plane
# E is Ex (Ey is longitudinal -- drove pure radiation, verified)
src_cells = [("Ex", src_i + d, ny // 2, nz // 2) for d in (-2, -1, 0, 1, 2)]

frames = []
n_steps = int(round(150.0 / c3.dt))
every = max(1, n_steps // 220)
for t in range(n_steps):
    a = prof.amplitude((t + 0.5) * c3.dt)
    for sc_ in src_cells:
        st = step_3d(st, c3, a, sc_) if sc_ is src_cells[0] else st
    # inject remaining cells without re-stepping
    for sc_ in src_cells[1:]:
        i_, j_, k_ = sc_[1], sc_[2], sc_[3]
        st.Ex[i_, j_, k_] += c3.dt * a / c3.eps[i_, j_, k_]
    if t % every == 0:
        ex = st.Ex[:, : ny - 1, nz // 2]          # (nx-1, ny-1)
        ey = st.Ey[: nx - 1, :, nz // 2]          # (nx-1, ny-1)
        frames.append(np.sqrt((ex ** 2 + ey ** 2).numpy()))

msk = np.ones_like(frames[0], dtype=bool)
msk[src_i - 5: src_i + 6, ny // 2 - 5: ny // 2 + 6] = False
vmax = max(f[msk].max() for f in frames[len(frames) // 2:]) * 0.9
fig, ax = plt.subplots(figsize=(6, 5.6))
ax.imshow(eps[:, :, nz // 2].numpy().T, origin="lower", cmap="binary", alpha=0.9)
im = ax.imshow(frames[0].T, origin="lower", cmap="inferno",
               vmin=0, vmax=vmax, alpha=0.8)
ax.set_title("3D LiNbO3 microring (300 nm TFLN): mid-slab $|E_\\parallel|$")
ax.set_xticks([]); ax.set_yticks([])


def update(i):
    im.set_data(frames[i].T)
    return (im,)


ani = anim.FuncAnimation(fig, update, frames=len(frames), interval=40)
ani.save(OUT / "b_ring_3d.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print(f"b_ring_3d.mp4 written ({len(frames)} frames, "
      f"{nx}x{ny}x{nz} voxels)")
