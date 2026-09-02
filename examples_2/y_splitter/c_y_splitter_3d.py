# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Y splitter: true-3D validation
#
# Example (c2b): TRUE-3D validation of the v6 Y splitter (674 nm).
#
# Rebuilds the v6 full-domain design (DEFAULT mmi1x2_with_sbend seed,
# no-symmetry optimization, `output/c_y_splitter_rho.npy`), extrudes it to
# the real stack -- SiN n = 2.0347 (Luke @674 nm), 150 nm slab, SiO2 clad --
# and runs full 3D FDTD on GPU: quasi-TE fundamental launched from the
# true 3D input cross-section, per-arm guided power vs a straight-guide
# 3D reference (fdtdx two-run pattern). No effective index anywhere.
#
# Run after c_y_splitter.py:
# `uv run python examples_2/y_splitter/c_y_splitter_3d.py`

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Reuse the 2D setup head
# The setup portion of c_y_splitter.py (everything before its S-parameter
# function) is exec'd as source text, so grid, ports, and design region
# are byte-identical to the optimization run.

# %%
src_txt = open(HERE / "c_y_splitter.py").read()
head = src_txt.split("def s_params")[0]
g = {"__file__": str(HERE / "c_y_splitter.py")}
exec(compile(head, "splitter_head", "exec"), g)

from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.materials import Si3N4
from meep_torch.core.monitors_3d import FluxMonitor3D
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean
from meep_torch.ports.validate3d import ModePlaneSourceX, extrude

# %% [markdown]
# ## Rebuild the optimized 2D design
# The saved rho goes through the same conic filter + tanh projection at
# BETA=64, painted into the base eps; a straight guide of the same core
# eps serves as the normalization reference.

# %%
grid2, eps_base, region = g["grid"], g["eps_base"], g["region"]
RES2, dx2, FCEN, N_CLAD = g["RES"], g["dx"], g["FCEN"], g["N_CLAD"]
ports = g["ports"]
o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
o3x, o3y, _ = ports["o3"]
rho = torch.from_numpy(np.load(OUT / "c_y_splitter_rho.npy")).to(eps_base.dtype)
r = tanh_projection(conic_filter(rho, 0.12, dx2), 64.0)
eps2d = region.paint(eps_base, r).detach().cpu()
eps2d_ref = torch.full_like(eps2d, N_CLAD ** 2)
eps2d_ref[:, o1y - int(0.25 * RES2): o1y + int(0.25 * RES2)] = float(eps2d.max())

# %% [markdown]
# ## Extrude to the real 3D stack
# Resample to RES=40 (6 cells across the 150 nm core), extrude with the
# REAL SiN index -- no effective medium -- and use float32 to halve VRAM.

# %%
RES3 = 40   # 6 cells across the 150 nm core (RES=20 = unresolved slab)
N_SIN3 = float(Si3N4.n(FCEN).real)                  # REAL index, no eff. medium
H_CORE, PAD_Z = 0.15, 1.5
dx3 = 1.0 / RES3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


DT3 = torch.float32   # shared GPU with the coupler-3D run: halve VRAM


def to3d(e2):
    n3x = int(e2.shape[0] * RES3 / RES2)
    n3y = int(e2.shape[1] * RES3 / RES2)
    ds = F.interpolate(e2[None, None], size=(n3x, n3y),
                       mode="bilinear", align_corners=False)[0, 0]
    return extrude(ds, n_clad=N_CLAD, n_core=N_SIN3, h_core=H_CORE,
                   pad_z=PAD_Z, res=RES3).to(DEV, dtype=DT3)


eps3 = to3d(eps2d)
eps3_ref = to3d(eps2d_ref)
nx, ny, nz = eps3.shape
NPML = RES3
T_RUN = 2 * (5 / 0.2) + nx * dx3 * 1.7 + 40.0
print(f"c_y_splitter_3d  grid {nx}x{ny}x{nz} "
      f"({nx*dx3:.1f} x {ny*dx3:.1f} x {nz*dx3:.1f} um) on {DEV}, "
      f"SiN n={N_SIN3:.4f} h={H_CORE} um, T_RUN={T_RUN:.0f}")

# %% [markdown]
# ## 3D runs
# Two lean-stepper runs (fdtdx two-run pattern): the straight reference
# fixes the input power, then the device run measures per-arm flux.

# %%
s3 = lambda j: int(j * RES3 / RES2)
IX_SRC = max(s3(o1x) + 6, NPML + 4)
IX_OUT = min(s3(o2x) - 6, nx - NPML - 4)
ARM = int(1.2 * RES3)
def run_case(e3, wins):
    c = build_lean(nx, ny, nz, dx3, e3, NPML, dtype=DT3, device=DEV)
    src = ModePlaneSourceX(e3.to(DEV, dtype=DT3), dx3, IX_SRC, FCEN,
                           GaussianPulse(FCEN, 0.2), c.dt)
    mons = [FluxMonitor3D(c.dt, [FCEN], normal="x", idx=IX_OUT,
                          win=(w0, w1, NPML, nz - NPML)) for (w0, w1) in wins]
    _, accs = run_3d_lean(c, int(round(T_RUN / c.dt)), source=src,
                          monitors=mons, compile=(DEV == "cuda"))
    if DEV == "cuda":
        print(f"  run peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    neff = src.neff
    del c, src
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return [float(m.flux(a, dx3)[0]) for m, a in zip(mons, accs)], neff


(p_ref,), neff_in = run_case(eps3_ref, [(s3(o1y) - ARM, s3(o1y) + ARM)])
(p_up, p_dn), _ = run_case(eps3, [(s3(o2y) - ARM, s3(o2y) + ARM),
                                  (s3(o3y) - ARM, s3(o3y) + ARM)])
T_up, T_dn = p_up / p_ref, p_dn / p_ref
tot = T_up + T_dn
imb = abs(T_up - T_dn) / max(tot, 1e-12)
print(f"c_y_splitter_3d  input TE0 neff = {neff_in.real:.4f}")
print(f"c_y_splitter_3d  3D: |S21|^2={T_up*100:.1f}% |S31|^2={T_dn*100:.1f}% "
      f"total {tot*100:.1f}% (IL {-10*math.log10(max(tot,1e-12)):.2f} dB), "
      f"imbalance {imb*100:.2f}%")

# %% [markdown]
# ## Plot
# Mid-slab permittivity slice and the per-arm 3D transmission bars, saved
# to `output/c_y_splitter_3d.png`.

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
kmid = nz // 2
ax1.imshow(eps3[:, :, kmid].cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx3, 0, ny * dx3], aspect="auto")
ax1.set_title(f"v6 design extruded (SiN n={N_SIN3:.3f}, 150 nm)")
ax1.set_xlabel("x [um]")
ax2.bar(["S21 (up)", "S31 (dn)"], [T_up * 100, T_dn * 100], color="tab:blue")
ax2.set_ylabel("power [% of straight-guide ref]")
ax2.set_title(f"true-3D: total {tot*100:.1f}%, imbalance {imb*100:.2f}%")
fig.tight_layout()
fig.savefig(OUT / "c_y_splitter_3d.png", dpi=150)
plt.close(fig)
print("c_y_splitter_3d.png written")
