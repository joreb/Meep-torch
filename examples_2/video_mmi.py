# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c3b): video of the inverse-designed mmi_tapered coupler
(674 nm). Rebuilds the optimized design exactly like the 3D validator
(exec of the characterization head + saved rho over the same region),
drives the TE0 mode CW and animates |Ez|. Cores shaded + outlined.

Run after c_y_coupler_mmi_tapered.py:
    uv run python examples_2/video_mmi.py
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent))
OUT = pathlib.Path(__file__).parent

src_txt = open(OUT / "c_y_coupler_mmi_tapered.py").read()
head = src_txt.split("# --- inverse-design optimization")[0]
head = head.split("def s_params_opt(")[0]
g = {"__file__": str(OUT / "c_y_coupler_mmi_tapered.py")}
exec(compile(head, "mmi_head", "exec"), g)

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.sources import ContinuousSource
from meep_torch.core.stepper import init_tm_state, step_tm
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

grid, dx, FCEN, N_CLAD, NEFF = g["grid"], g["dx"], g["FCEN"], g["N_CLAD"], g["NEFF"]
RES, NPML = g["RES"], g["NPML"]
eps_base = g["eps"]
nx, ny = grid.nx, grid.ny
o1x, o1y, _ = g["ports"]["o1"]
o2x, o2y, _ = g["ports"]["o2"]
o3x, o3y, _ = g["ports"]["o3"]
# same design region as the opt stage (mirrors c_y_coupler_mmi_tapered_3d.py)
ix0 = o1x + int((4.0 + 10.0) * RES)
ix1 = min(o2x - int(2.0 * RES), nx - RES - 4)
iy0 = int(max(o3y - 1.6 * RES, RES))
iy1 = int(min(o2y + 1.6 * RES, ny - RES))
rho = torch.from_numpy(np.load(OUT / "c_y_coupler_mmi_tapered_rho.npy"))
mg = MaterialGrid(rho.shape, eps_min=N_CLAD ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
rr = tanh_projection(conic_filter(rho.to(torch.float64), 0.12, dx), 64.0)
eps_fin = region.paint(eps_base.cpu().to(torch.float64), rr).detach()
eps_fin = eps_fin.to(grid.device, dtype=eps_base.dtype)

SPAN = int(1.2 * RES)
p_in = Port(axis="x", index=max(o1x + int(2.0 * RES), NPML + 4),
            lo=o1y - SPAN, hi=o1y + SPAN, direction=+1)
coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML)
src = ModeSourceTM(grid, coeffs, p_in, ContinuousSource(FCEN, width=10.0), FCEN)
st = init_tm_state(grid)
frames = []
n_steps = int(round(160.0 / grid.dt))
every = max(1, n_steps // 180)
sub = max(1, nx // 1000)
for t in range(n_steps):
    h = getattr(src, "h_masks", None)
    if h:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask,
                     src.h_amplitude(t), **h)
    else:
        st = step_tm(st, coeffs, src.amplitude(t), src.mask)
    if t % every == 0:
        frames.append(st.Ez[::sub, ::sub].abs().cpu().numpy())
vmax = max(f.max() for f in frames[len(frames) // 2:]) * 0.6
core = (eps_fin[::sub, ::sub]
        > 0.5 * (float(eps_fin.min()) + float(eps_fin.max()))).cpu().numpy()
fig, ax = plt.subplots(figsize=(14, 3.2))
im = ax.imshow(frames[0].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax,
               extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax.imshow(np.ma.masked_where(~core, core).T, origin="lower", cmap="gray",
          alpha=0.22, extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax.contour(np.linspace(0, nx * dx, core.shape[0]),
           np.linspace(0, ny * dx, core.shape[1]),
           core.T.astype(float), levels=[0.5], colors="k", linewidths=0.6)
ax.set_title("inverse-designed mmi_tapered at 674 nm: $|E_z|$")
ax.set_xlabel("x [um]")


def upd(i):
    im.set_data(frames[i].T)
    return (im,)


anim.FuncAnimation(fig, upd, frames=len(frames), interval=40).save(
    OUT / "c_mmi_propagation.mp4", writer=anim.FFMpegWriter(fps=25))
plt.close(fig)
print("c_mmi_propagation.mp4 written")
