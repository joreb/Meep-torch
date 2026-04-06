# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""TIER-1 3D validation of the grating-to-atom focuser (lean stepper).

Extrudes the FDFD-optimized vertical-cut design (grating_atom_focus.py,
gdsfactory-seeded rho) uniformly across a 6 um-wide strip -- teeth
become real trenches -- and runs true 3D FDTD: quasi-TE slab mode in,
DFT of Ey at the atom's x-plane, against a bare-slab reference run
(same guide-peak normalization as 2D). The 3D beam also diverges
LATERALLY, which the 2D design never saw -- the honest question this
run answers is how much of the 2D figure (0.41x guide peak) survives.

Scope note: this validates the VERTICAL-CUT design extruded in width;
it is not the fdtdx original's plan-view-patterned 3D device (that is
Tier 2, full-3D inverse design -- wiki TODO).

Run after grating_atom_focus.py:
    uv run python examples_2/grating_atom_focus_3d.py
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
OUT = pathlib.Path(__file__).parent

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors_3d import DFTFieldsMonitor3D
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean
from meep_torch.ports.validate3d import ModePlaneSourceX

WL = 1.092
FCEN = 1 / WL
N_SIN = float(Si3N4.n(FCEN).real)
N_SIO = float(SiO2.n(FCEN).real)
RES2 = 30
H_FILM, L_DEV, X0_DEV, Z_SUB = 0.52, 18.0, 4.0, 2.5
LX, LZ = 25.0, 8.0

# rebuild the 2D eps exactly as the design script paints it
nx2, nz2 = int(LX * RES2), int(LZ * RES2)
k0, k1 = int(Z_SUB * RES2), int(Z_SUB * RES2) + int(H_FILM * RES2)
base2 = torch.ones(nx2, nz2, dtype=torch.float64)
base2[:, :k0] = N_SIO ** 2
base2[:, k0:k1] = N_SIN ** 2
rho = torch.from_numpy(np.load(OUT / "grating_atom_focus_rho.npy"))
mg = MaterialGrid(rho.shape, eps_min=1.0, eps_max=N_SIN ** 2)
region = DesignRegion(mg, ix0=int(X0_DEV * RES2), ix1=int((X0_DEV + L_DEV) * RES2),
                      iy0=k0, iy1=k1)
r = tanh_projection(conic_filter(rho, 0.12, 1 / RES2), 64.0)
eps2 = region.paint(base2, r).detach()

# --- 3D: extrude across a 6 um strip, downsample to RES3 ---
RES3 = 20
dx3 = 1.0 / RES3
W_DEV, PAD_Y = 6.0, 2.0
NPML = RES3
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT3 = torch.float32

nx = int(LX * RES3)
nz = int(LZ * RES3)
ny = int((W_DEV + 2 * PAD_Y) * RES3)
j0, j1 = int(PAD_Y * RES3), int((PAD_Y + W_DEV) * RES3)


def to3d(e2):
    ds = F.interpolate(e2[None, None], size=(nx, nz), mode="bilinear",
                       align_corners=False)[0, 0]
    bg = torch.ones(nz, dtype=torch.float64)
    bg[:int(Z_SUB * RES3)] = N_SIO ** 2                    # substrate everywhere
    e3 = bg[None, None, :].repeat(nx, ny, 1).clone()
    e3[:, j0:j1, :] = ds[:, None, :]                       # strip = 2D cut
    return e3.to(DEV, dtype=DT3)


bare2 = base2
eps3 = to3d(eps2)
eps3_ref = to3d(bare2)
print(f"grating_atom_focus_3d  grid {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.1f}M cells "
      f"on {DEV}")

IX_SRC = NPML + 6
ia_x = int((X0_DEV + L_DEV - 1.4) * RES3)
ia_z = int((Z_SUB + H_FILM + 3.5) * RES3)
ia_y = ny // 2
icol = int((X0_DEV + 2.0) * RES3)
kf0, kf1 = int(Z_SUB * RES3), int((Z_SUB + H_FILM) * RES3)

T_RUN = 2 * (5 / 0.2) + LX * 2.1 + 40


def run_case(e3):
    c = build_lean(nx, ny, nz, dx3, e3, NPML, dtype=DT3, device=DEV)
    src = ModePlaneSourceX(e3, dx3, IX_SRC, FCEN, GaussianPulse(FCEN, 0.2), c.dt)
    mons = [DFTFieldsMonitor3D(c.dt, [FCEN], "Ey", plane=(0, ia_x)),
            DFTFieldsMonitor3D(c.dt, [FCEN], "Ey", plane=(0, icol))]
    _, accs = run_3d_lean(c, int(round(T_RUN / c.dt)), source=src,
                          monitors=mons, compile=(DEV == "cuda"))
    if DEV == "cuda":
        print(f"  peak VRAM {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    out = []
    for acc in accs:
        E = (acc[0, 0] + 1j * acc[0, 1]).cpu()             # (ny-ish, nz) plane
        out.append(E)
    del c, src
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


atom_pl, _ = run_case(eps3)
_, guide_pl = run_case(eps3_ref)
I_atom = float(abs(atom_pl[ia_y, ia_z]) ** 2)
I_guide = float((abs(guide_pl[j0:j1, kf0:kf1]) ** 2).max())
gain3 = I_atom / I_guide
print(f"grating_atom_focus_3d  atom/guide-peak (3D) = {gain3:.3f}  "
      f"(2D design figure: 0.41)")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.imshow((abs(atom_pl.numpy()) ** 2).T, origin="lower", cmap="inferno",
          extent=[0, ny * dx3, 0, nz * dx3], aspect="auto")
a1.plot([ia_y * dx3], [ia_z * dx3], "c*", ms=12)
a1.set_title(f"|Ey|^2 at the atom x-plane (3D), gain {gain3:.2f}")
a1.set_xlabel("y [um]"); a1.set_ylabel("z [um]")
a2.imshow(eps3[:, ia_y, :].cpu().numpy().T, origin="lower", cmap="viridis",
          extent=[0, nx * dx3, 0, nz * dx3], aspect="auto")
a2.plot([ia_x * dx3], [ia_z * dx3], "r*", ms=12)
a2.set_title("extruded design, center cut")
a2.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "grating_atom_focus_3d.png", dpi=150)
print("grating_atom_focus_3d.png written")
