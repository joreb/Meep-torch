# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c): inverse-design videos --
  1. design-evolution animation of the Y splitter (re-run of the 16-step
     optimization, one frame per iteration);
  2. 3D FDTD of the FINAL splitter design (150 nm extrusion), mid-slab
     field slice animated.
GDS export lives in export_gds.py.

Run: uv run python examples_2/video_inverse.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

from meep_torch.adjoint import DesignRegion, MaterialGrid, OptimizationProblem
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import ContinuousSource, GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state, step_3d

OUT = pathlib.Path(__file__).parent
NEFF = 1.4684
RES = 20
FCEN = 1 / 1.55
dx = 1.0 / RES
nx, ny = int(12.0 * RES), int(8.0 * RES)
grid = Grid(nx=nx, ny=ny, dx=dx)
cy = ny // 2
jw = int(0.25 * RES)
SEP = int(1.2 * RES)
base = torch.full((nx, ny), 1.44 ** 2, dtype=torch.float64)
i_in1, i_out0 = int(3.0 * RES), int(9.0 * RES)
base[:i_in1, cy - jw: cy + jw] = NEFF ** 2
base[i_out0:, cy + SEP - jw: cy + SEP + jw] = NEFF ** 2
base[i_out0:, cy - SEP - jw: cy - SEP + jw] = NEFF ** 2
mg = MaterialGrid((24, 16), eps_min=1.44 ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=i_in1, ix1=i_out0,
                      iy0=cy - int(2.0 * RES), iy1=cy + int(2.0 * RES))


def run_fn(eps_full):
    coeffs = build_tm_coeffs(grid, eps_inf=eps_full, npml=RES)
    src = LineSource(grid, ix=int(1.5 * RES), j0=cy - 2 * jw, j1=cy + 2 * jw,
                     profile=GaussianPulse(FCEN, 0.1))
    m1 = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=nx - int(1.5 * RES),
                       lo=cy + SEP - 3 * jw, hi=cy + SEP + 3 * jw)
    m2 = FluxMonitorTM(grid, [FCEN], x_normal=True, idx=nx - int(1.5 * RES),
                       lo=cy - SEP - 3 * jw, hi=cy - SEP + 3 * jw)
    _, (a1, a2) = run_tm(init_tm_state(grid), coeffs,
                         int(round(40.0 / grid.dt)), source=src,
                         monitors=[m1, m2], checkpoint_every=300)
    return m1.flux(a1, dx)[0] + m2.flux(a2, dx)[0]


class Sym(OptimizationProblem):
    def _objective(self, rho_half):
        full = torch.cat([rho_half.flip(1), rho_half], dim=1)
        r = conic_filter(full, self.filter_radius, self.dx)
        r = tanh_projection(r, self.beta, self.eta)
        return self.run_fn(self.region.paint(self.base_eps, r))


opt = Sym(run_fn, base, region, filter_radius=0.15, dx=dx, beta=4.0,
          rho0=torch.full((24, 8), 0.5, dtype=torch.float64))
rho = opt.rho.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.04)
design_frames = []
for it in range(16):
    opt.beta = [4, 8, 16, 32][min(it // 4, 3)]
    adam.zero_grad()
    J = opt._objective(rho.clamp(0.0, 1.0))
    (-J).backward()
    adam.step()
    with torch.no_grad():
        full = torch.cat([rho.clamp(0, 1).flip(1), rho.clamp(0, 1)], dim=1)
        r = tanh_projection(conic_filter(full, 0.15, dx), opt.beta)
        design_frames.append(region.paint(base, r).numpy().copy())
    print(f"video_inverse  opt it={it}: J={float(J):.3e}")

fig, ax = plt.subplots(figsize=(7.4, 5))
im = ax.imshow(design_frames[0].T, origin="lower", cmap="viridis",
               extent=[0, nx * dx, 0, ny * dx])
ttl = ax.set_title("Y-splitter inverse design, iteration 0")


def upd(i):
    im.set_data(design_frames[i].T)
    ttl.set_text(f"Y-splitter inverse design, iteration {i}")
    return (im, ttl)


ani = anim.FuncAnimation(fig, upd, frames=len(design_frames), interval=400)
ani.save(OUT / "c_inverse_design_evolution.mp4",
         writer=anim.FFMpegWriter(fps=2.5))
plt.close(fig)
print("c_inverse_design_evolution.mp4 written")

# --- 3D run of the final design (150 nm extrusion), field slice video ---
r_fin = torch.from_numpy(design_frames[-1])
RES3 = 10
sc = RES3 / RES
nx3, ny3 = int(nx * sc), int(ny * sc)
nz3 = int(1.2 * RES3) + 8
eps2d = torch.nn.functional.interpolate(
    r_fin[None, None], size=(nx3, ny3), mode="bilinear")[0, 0]
# replace effective index by the true 3D stack: SiN 2.0 core, SiO2 clad
core2d = (eps2d > (1.44 ** 2 + NEFF ** 2) / 2).to(torch.float64)
eps3 = torch.full((nx3, ny3, nz3), 1.44 ** 2, dtype=torch.float64)
z0 = nz3 // 2 - 1
eps3[:, :, z0:z0 + 2] = (1.44 ** 2 + core2d[:, :, None]
                         * (2.0 ** 2 - 1.44 ** 2))
c3 = build_3d_coeffs(nx3, ny3, nz3, 1.0 / RES3, eps=eps3, n_abs=RES3 // 2)
st = init_3d_state(nx3, ny3, nz3)
prof = ContinuousSource(FCEN, width=6.0)
jw3 = max(2, int(0.25 * RES3))
src_cells = [("Ey", int(1.5 * RES3), j, z0)
             for j in range(ny3 // 2 - jw3, ny3 // 2 + jw3 + 1)]
frames3 = []
n_steps = int(round(70.0 / c3.dt))
every = max(1, n_steps // 175)
for t in range(n_steps):
    a = prof.amplitude((t + 0.5) * c3.dt)
    st = step_3d(st, c3, a, src_cells[0])
    for sc_ in src_cells[1:]:
        i_, j_, k_ = sc_[1], sc_[2], sc_[3]
        st.Ey[i_, j_, k_] += c3.dt * a / c3.eps[i_, j_, k_]
    if t % every == 0:
        frames3.append(st.Ey[:, :, z0].numpy().copy())
msk = np.ones_like(frames3[0], dtype=bool)
msk[int(1.5 * RES3) - 4: int(1.5 * RES3) + 5, :] = False
vmax = max(np.abs(f[msk]).max() for f in frames3[len(frames3) // 2:]) * 0.8
fig, ax = plt.subplots(figsize=(7.4, 5))
ax.imshow(core2d.numpy().T, origin="lower", cmap="binary", alpha=0.9)
im = ax.imshow(frames3[0].T, origin="lower", cmap="RdBu",
               vmin=-vmax, vmax=vmax, alpha=0.75)
ax.set_title("3D FDTD of the final splitter (150 nm SiN): mid-slab $E_y$")
ax.set_xticks([]); ax.set_yticks([])


def upd3(i):
    im.set_data(frames3[i].T)
    return (im,)


ani = anim.FuncAnimation(fig, upd3, frames=len(frames3), interval=40)
ani.save(OUT / "c_splitter_3d.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print(f"c_splitter_3d.mp4 written ({len(frames3)} frames, "
      f"{nx3}x{ny3}x{nz3} voxels)")
