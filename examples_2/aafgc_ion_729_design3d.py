# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""TIER-2 demo: full-3D inverse design of a PLAN-VIEW grating
(run_3d_adjoint, checkpointed 3D gradients).

The 2.5D pipeline can only chirp teeth along x; a real AAFGC curves its
grating lines to focus laterally. This demo optimizes a free 2D film
pattern rho(x, y) (extruded through the 220 nm thickness -- the
litho-faithful parameterization) to maximize upward DFT power through
a window above the grating. Reduced scene for iterate-able adjoint
runtime, dx = 73 nm (10 pts/lam vacuum, 5 in SiN -- QUALITATIVE, the
same coarseness class the original fdtdx script ran; numbers here are
demo-grade, not paper-grade).

Run: uv run python examples_2/aafgc_ion_729_design3d.py
"""
import math
import pathlib
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).parent))
OUT = pathlib.Path(__file__).parent

from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.adjoint3d import run_3d_adjoint
from meep_torch.core.monitors_3d import DFTFieldsMonitor3D
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state
from meep_torch.ports.validate3d import ModePlaneSourceX

WL = 0.729
FCEN = 1 / WL
N_SIN, N_SIO = 2.0, 1.44
dx = 0.073
RES = 1.0 / dx
LX, LY = 20.0, 6.0
OX_BOT, H_FILM, OX_TOP, VAC = 1.5, 0.22, 2.0, 3.0
LZ = OX_BOT + H_FILM + OX_TOP + VAC
nx, ny, nz = int(LX / dx), int(LY / dx), int(LZ / dx)
NPML = 10
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.float32
X0_DEV, L_DEV = 4.0, 12.0
W_STRIP = 4.0
print(f"aafgc_design3d  grid {nx}x{ny}x{nz} = {nx*ny*nz/1e6:.1f}M cells on {DEV}")

k0f = int(OX_BOT / dx)
k1f = k0f + max(int(H_FILM / dx), 3)
k_surf = int((OX_BOT + H_FILM + OX_TOP) / dx)
j0 = int((LY / 2 - W_STRIP / 2) / dx)
j1 = int((LY / 2 + W_STRIP / 2) / dx)
ix0, ix1 = int(X0_DEV / dx), int((X0_DEV + L_DEV) / dx)

base = torch.ones(nx, ny, nz, dtype=DT, device=DEV)
base[:, :, :k_surf] = N_SIO ** 2
base[:, j0:j1, k0f:k1f] = N_SIN ** 2              # strip guide

# objective window: upward DFT power on a plane 2 um above the surface,
# x-window over the second half of the grating
k_ce = min(int((OX_BOT + H_FILM + OX_TOP + 2.0) / dx), nz - NPML - 3)
wx0, wx1 = int(10.0 / dx), int(18.0 / dx)

# analytic chirp seed (y-uniform), target above the window center
_, dummy = 0, 0
neff = 1.75                                        # coarse-grid estimate
xa, za = 14.0, 8.0
def phase(x):
    return neff * x + math.hypot(xa - x, za)
C0 = phase(X0_DEV)
NRX, NRY = int(L_DEV / 0.15), int(W_STRIP / 0.15)
xs_r = X0_DEV + (np.arange(NRX) + 0.5) * 0.15
seed = np.zeros(NRX)
m = 1
from scipy.optimize import brentq
while True:
    try:
        xm = brentq(lambda x: phase(x) - C0 - m * WL, X0_DEV - 0.5,
                    X0_DEV + L_DEV + 1.0)
    except ValueError:
        break
    if xm > X0_DEV + L_DEV:
        break
    seed[np.abs(xs_r - xm) < 0.18] = 1.0
    m += 1

rho = (0.3 + 0.4 * torch.from_numpy(seed)[:, None].repeat(1, NRY)).to(
    torch.float64).requires_grad_(True)


def paint3d(r2):
    """rho (NRX, NRY) -> film voxels of the strip, differentiable."""
    up = F.interpolate(r2[None, None].to(DT), size=(ix1 - ix0, j1 - j0),
                       mode="bilinear", align_corners=False)[0, 0].to(DEV)
    eps = base.clone()
    fill = N_SIO ** 2 + up * (N_SIN ** 2 - N_SIO ** 2)
    eps[ix0:ix1, j0:j1, k0f:k1f] = fill[:, :, None]
    return eps


class _Src:
    def __init__(self, inner):
        self.inner = inner
        self.src = inner.src

    def amplitude(self, t):
        return self.inner.amplitude(t)


T_RUN = 2 * (5 / 0.2) + (LX + LZ) * 1.8 + 30
pulse = GaussianPulse(FCEN, 0.2)
IX_SRC = NPML + 5
CHUNK = 64


def J_of(r2, beta):
    r = tanh_projection(conic_filter(r2, 0.15, 0.15), beta)
    eps = paint3d(r)
    c = build_3d_coeffs(nx, ny, nz, dx, eps=eps, npml=NPML, dtype=DT,
                        device=DEV)
    src = _Src(ModePlaneSourceX(eps.detach(), dx, IX_SRC, FCEN, pulse, c.dt))
    mon = DFTFieldsMonitor3D(c.dt, [FCEN], "Ey", plane=(2, k_ce), dtype=DT)
    st0 = init_3d_state(nx, ny, nz, npml=NPML, dtype=DT, device=DEV)

    def loss_fn(accs):
        # NEGATIVE window power: run_3d_adjoint backpropagates THIS and
        # Adam minimizes -- the first run returned +power, so the
        # optimizer minimized the objective (J fell 0.247 -> 0.07
        # monotonically). The FDFD scripts negate at .backward(); here
        # the backward lives inside the core call.
        E2 = accs[0][0, 0] ** 2 + accs[0][0, 1] ** 2
        return -E2.squeeze()[wx0:wx1, :].sum()

    L, _ = run_3d_adjoint(c, int(round(T_RUN / c.dt)), CHUNK, src, [mon],
                          loss_fn, st0)
    return -L                                     # report positive power


history = []
best = (-1.0, None)
adam = torch.optim.Adam([rho], lr=0.03)
t00 = time.time()
for it in range(8):
    BETA = [8, 16, 32, 64][min(it // 2, 3)]
    adam.zero_grad()
    t0 = time.time()
    L = J_of(rho.clamp(0, 1), BETA)
    # run_3d_adjoint accumulated grads on rho through paint3d
    adam.step()
    history.append(L)
    if L > best[0]:
        best = (L, rho.detach().clamp(0, 1).clone())
    print(f"aafgc_design3d  it={it} beta={BETA}: J={L:.4e} "
          f"({time.time()-t0:.0f}s, peak "
          f"{torch.cuda.max_memory_allocated()/1e9:.1f} GB)" if DEV == "cuda"
          else f"aafgc_design3d  it={it}: J={L:.4e}")
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()

rho_f = best[1]
np.save(OUT / "aafgc_ion_729_design3d_rho.npy", rho_f.numpy())
print(f"aafgc_design3d  done: J {history[0]:.3e} -> {best[0]:.3e} "
      f"({best[0]/max(history[0],1e-30):.1f}x) in {time.time()-t00:.0f}s")

r = tanh_projection(conic_filter(rho_f, 0.15, 0.15), 64.0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(history, "bo-")
a1.set_xlabel("iteration"); a1.set_ylabel("window DFT power [arb]")
a1.set_title(f"3D adjoint design: {best[0]/max(history[0],1e-30):.1f}x over seed")
a2.imshow(r.numpy().T, origin="lower", cmap="viridis",
          extent=[X0_DEV, X0_DEV + L_DEV, 0, W_STRIP])
a2.set_xlabel("x [um]"); a2.set_ylabel("y [um]")
a2.set_title("plan-view film pattern (projected)")
fig.tight_layout()
fig.savefig(OUT / "aafgc_ion_729_design3d.png", dpi=150)
print("aafgc_ion_729_design3d.png written")
