# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2e): TRUE-3D validation of the inverse-designed mmi_tapered
Y coupler (674 nm, 98.5% in 2D).

Rebuilds the optimized design (c_y_coupler_mmi_tapered_rho.npy over the
tuned length_mmi=34 geometry), extrudes to the real stack (SiN n=2.0347,
150 nm, SiO2) and measures per-arm guided power vs a straight-guide 3D
reference. GPU, forward-only.

Run after c_y_coupler_mmi_tapered.py:
  uv run python examples_2/c_y_coupler_mmi_tapered_3d.py
"""
import math
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

src_txt = open(OUT / "c_y_coupler_mmi_tapered.py").read()
head = src_txt.split("# --- inverse-design optimization")[0]
head = head.split("def s_params_opt(")[0]
g = {"__file__": str(OUT / "c_y_coupler_mmi_tapered.py")}
exec(compile(head, "mmi_head", "exec"), g)

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.materials import Si3N4
from meep_torch.core.monitors_3d import FluxMonitor3D
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean
from meep_torch.ports.validate3d import ModePlaneSourceX, extrude

grid2 = g["grid"]
eps2_base = g["eps"].cpu()
RES2, dx2, FCEN, N_CLAD, NEFF = g["RES"], g["dx"], g["FCEN"], g["N_CLAD"], g["NEFF"]
nx2, ny2 = grid2.nx, grid2.ny
o1x, o1y, _ = g["ports"]["o1"]
o2x, o2y, _ = g["ports"]["o2"]
o3x, o3y, _ = g["ports"]["o3"]
# rebuild the SAME design region as the opt stage
ix0 = o1x + int((4.0 + 10.0) * RES2)
ix1 = min(o2x - int(2.0 * RES2), nx2 - RES2 - 4)
iy0 = int(max(o3y - 1.6 * RES2, RES2))
iy1 = int(min(o2y + 1.6 * RES2, ny2 - RES2))
rho = torch.from_numpy(np.load(OUT / "c_y_coupler_mmi_tapered_rho.npy"))
NRX, NRY = rho.shape
mg = MaterialGrid((NRX, NRY), eps_min=N_CLAD ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
rr = tanh_projection(conic_filter(rho.to(torch.float64), 0.12, dx2), 64.0)
eps2d = region.paint(eps2_base, rr).detach().cpu()
# the exec'd 2D head ran its characterization on the GPU -- free it all
g.clear()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
eps2d_ref = torch.full_like(eps2d, N_CLAD ** 2)
eps2d_ref[:, o1y - int(0.25 * RES2): o1y + int(0.25 * RES2)] = float(eps2d.max())

RES3 = 40   # 6 cells across the 150 nm core (RES=20 = unresolved slab)
N_SIN3 = float(Si3N4.n(FCEN).real)
dx3 = 1.0 / RES3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


DT3 = torch.float32   # 212M-cell grid: float32 to fit VRAM (validated dtype)


def to3d(e2):
    n3x = int(e2.shape[0] * RES3 / RES2)
    n3y = int(e2.shape[1] * RES3 / RES2)
    ds = F.interpolate(e2[None, None], size=(n3x, n3y),
                       mode="bilinear", align_corners=False)[0, 0]
    return extrude(ds, n_clad=N_CLAD, n_core=N_SIN3, h_core=0.15,
                   pad_z=1.5, res=RES3).to(DEV, dtype=DT3)


eps3, eps3_ref = to3d(eps2d), to3d(eps2d_ref)
nx, ny, nz = eps3.shape
NPML = RES3
T_RUN = 2 * (5 / 0.2) + nx * dx3 * 1.7 + 40.0
print(f"c_mmi_tapered_3d  grid {nx}x{ny}x{nz} on {DEV}, T_RUN={T_RUN:.0f}")

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
                          monitors=mons, compile=(DEV == "cuda"))  # auto-falls back to eager on OOM
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
print(f"c_mmi_tapered_3d  input TE0 neff={neff_in.real:.4f}")
print(f"c_mmi_tapered_3d  3D: |S21|^2={T_up*100:.1f}% |S31|^2={T_dn*100:.1f}% "
      f"total {tot*100:.1f}% (IL {-10*math.log10(max(tot,1e-12)):.2f} dB), "
      f"imbalance {imb*100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.imshow(eps3[:, :, nz // 2].cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx3, 0, ny * dx3], aspect="auto")
ax1.set_title(f"inverse-designed mmi_tapered extruded (SiN {N_SIN3:.3f})")
ax1.set_xlabel("x [um]")
ax2.bar(["S21", "S31"], [T_up * 100, T_dn * 100], color="tab:blue")
ax2.set_title(f"true-3D: total {tot*100:.1f}%, imbalance {imb*100:.2f}%")
ax2.set_ylabel("power [% of ref]")
fig.tight_layout()
fig.savefig(OUT / "c_y_coupler_mmi_tapered_3d.png", dpi=150)
plt.close(fig)
print("c_y_coupler_mmi_tapered_3d.png written")
