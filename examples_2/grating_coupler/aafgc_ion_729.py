# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # AAFGC: ion-addressing focusing grating coupler (2D)
#
# Rebuild of the npj Nanophotonics AAFGC (s44310-025-00102-4):
# adjoint-optimized focusing grating coupler addressing a trapped ion.
#
# Paper: Momenzadeh et al., "Individual trapped-ion addressing with
# adjoint-optimized multimode photonic circuits" -- 220 nm SiN core,
# SiO2 top/bottom oxide (5 um), lam = 729 nm (40Ca+ qubit transition),
# ion trapped H = 68 um ABOVE the chip surface, ~30 um lateral offset,
# grating: 40 teeth, FULL etch, min feature 200 nm. Their result:
# focusing efficiency (FE, power in a 5 um window at the ion plane over
# grating input power) = -3.8 dB after adjoint optimization.
#
# This rebuild (2D x-z, meep-torch, single TE00 lane of the paper's
# multimode scheme): analytic chirp seed (phase condition to the ion
# point) -> FDFD-pardiso exact-adjoint optimization of the tooth pattern
# -> FDTD cross-check. Omitted vs the paper: Au electrode layers + ITO
# (no metals in the 2D TM stack -- their ITO alone costs ~1 dB), and the
# TE10 crosstalk machinery (multimode addressing is 3D). FE here uses
# |Ez|^2 sums as a power proxy (same-medium ratios).
#
# Run: `uv run python examples_2/grating_coupler/aafgc_ion_729.py`
# (outputs land in `examples_2/grating_coupler/output/`)

# %%
import math
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import brentq

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

from meep_torch.adjoint import DesignRegion, MaterialGrid
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import DFTFieldsMonitor
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.fdfd import make_mode_source, port_mode_line, solve_fdfd_tm
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

# %% [markdown]
# ## Layer stack and grid
# 220 nm SiN film in SiO2 oxide (bottom oxide truncated into the PML, 5 um top
# oxide), vacuum above, and 68 um of free space up to the ion plane. RES=20
# gives 27 points per vacuum wavelength at 729 nm.

# %%
WL = 0.729
FCEN = 1 / WL
N_SIN, N_SIO = 2.0, 1.44          # paper's stated indices
RES = 20                          # 27 pts/lam in vacuum, 10 in oxide
dx = 1.0 / RES
NPML = RES

H_FILM = 0.22
OX_BOT, OX_TOP = 3.0, 5.0         # bottom oxide truncated into PML
H_ION = 68.0                      # above chip surface (top-oxide top)
D_LAT = 30.0                      # lateral grating->ion offset
L_DEV = 30.0                      # 40-teeth grating span
X0_DEV = 6.0
LX = X0_DEV + L_DEV + D_LAT + 14.0
LZ = OX_BOT + H_FILM + OX_TOP + H_ION + 3.0
nx, nz = int(LX * RES), int(LZ * RES)
print(f"aafgc_ion_729  grid {nx}x{nz} = {nx*nz/1e6:.1f}M cells, "
      f"lam=729nm SiN220/SiO2, ion at H={H_ION} um")

k0f = int(OX_BOT * RES)
k1f = k0f + max(int(H_FILM * RES), 4)
k_surf = int((OX_BOT + H_FILM + OX_TOP) * RES)
base = torch.full((nx, nz), 1.0, dtype=torch.float64, device=grid.device)         # vacuum
base[:, :k_surf] = N_SIO ** 2                                 # oxides
base[:, k0f:k1f] = N_SIN ** 2                                 # film

# %% [markdown]
# ## Ion target window and input port
# The ion sits 68 um above the chip, 30 um laterally offset; FE is measured in
# a 5 um window at that plane. The input TE0 mode enters through an x-normal
# port on the film.

# %%
ia_x = int((X0_DEV + L_DEV / 2 + D_LAT) * RES)
ia_z = int((OX_BOT + H_FILM + OX_TOP + H_ION) * RES)
WIN = int(2.5 * RES)                                          # 5 um window

SPAN = int(1.5 * RES)
zc = (k0f + k1f) // 2
p_in = Port(axis="x", index=NPML + 8, lo=zc - SPAN, hi=zc + SPAN, direction=+1)
_, neff = port_mode_line(base, p_in, FCEN, dx)

# %% [markdown]
# ## Analytic chirp seed
# Tooth positions from a stationary-phase condition to the ion point: each
# tooth adds one wavelength of optical path (guide leg at neff, then oxide and
# vacuum legs to the ion). Yields the paper's ~40-tooth chirped grating.

# %%
# analytic chirp seed: neff*x + n_oxide*R_ox(x) + R_vac(x) stationary;
# collapse to the effective path n_eff*x + OPL(tooth->ion) with the
# oxide leg at N_SIO (exact ray split is an unneeded refinement here)
xa = ia_x * dx
def opl(x):
    dxl = xa - x
    frac_ox = OX_TOP / (OX_TOP + H_ION)
    return (N_SIO * math.hypot(dxl * frac_ox, OX_TOP)
            + math.hypot(dxl * (1 - frac_ox), H_ION))
def phase(x):
    return neff * x + opl(x)
C0 = phase(X0_DEV)
teeth, m = [], 1
while True:
    try:
        xm = brentq(lambda x: phase(x) - C0 - m * WL, X0_DEV - 0.5,
                    X0_DEV + L_DEV + 2.0)
    except ValueError:
        break
    if xm > X0_DEV + L_DEV or len(teeth) >= 40:               # paper: 40 teeth
        break
    teeth.append(xm)
    m += 1
lam_loc = np.diff([X0_DEV] + teeth)
print(f"aafgc_ion_729  analytic: {len(teeth)} teeth, "
      f"period {lam_loc.min():.2f}..{lam_loc.max():.2f} um, neff={neff:.4f}")

# %% [markdown]
# ## Design region and FDFD objective
# The tooth pattern becomes a MaterialGrid over the 30 um grating span; FE is
# computed from an exact FDFD (pardiso) solve as the |Ez|^2 sum in the ion
# window over the input-line sum.

# %%
NRX = int(L_DEV / 0.1)
NRZ = max(int(H_FILM / 0.1), 2)
xs_r = X0_DEV + (np.arange(NRX) + 0.5) * 0.1
seed = np.zeros(NRX)
for i, xm in enumerate(teeth):
    w_loc = 0.5 * (lam_loc[i] if i < len(lam_loc) else lam_loc[-1])
    seed[(xs_r > xm - w_loc / 2) & (xs_r < xm + w_loc / 2)] = 1.0

mg = MaterialGrid((NRX, NRZ), eps_min=N_SIO ** 2, eps_max=N_SIN ** 2)
region = DesignRegion(mg, ix0=int(X0_DEV * RES), ix1=int((X0_DEV + L_DEV) * RES),
                      iy0=k0f, iy1=k1f)

jz = make_mode_source(base, p_in, FCEN, dx)
E_ref = solve_fdfd_tm(base, dx, FCEN, jz, NPML, backend="pardiso")
icol = int((X0_DEV - 2.0) * RES)
P_in = float((E_ref[icol, k0f - SPAN:k1f + SPAN].abs() ** 2).sum())


def FE_of(eps):
    """|Ez|^2-sum in the 5 um ion-plane window over the input-line sum."""
    E = solve_fdfd_tm(eps, dx, FCEN, jz, NPML, backend="pardiso")
    win = (E[ia_x - WIN:ia_x + WIN, ia_z].abs() ** 2).sum()
    return win / P_in, E


def J_of(rho, beta):
    r = tanh_projection(conic_filter(rho, 0.12, dx), beta)
    fe, _ = FE_of(region.paint(base, r))
    return fe


# %% [markdown]
# ## Seed evaluation and adjoint optimization
# Evaluate the analytic seed, then 40 Adam iterations on the tooth pattern
# with a beta ramp (8 -> 64); gradients come from the exact FDFD adjoint.
# The best post-step design at beta=64 is kept.

# %%
with torch.no_grad():
    seed_t = torch.from_numpy(seed)[:, None].repeat(1, NRZ)
    FE_seed = float(J_of(seed_t, 64.0))
print(f"aafgc_ion_729  analytic seed FE proxy = {FE_seed:.4f} "
      f"({10*math.log10(max(FE_seed,1e-12)):.1f} dB)")

rho = (0.3 + 0.4 * torch.from_numpy(seed))[:, None].repeat(1, NRZ).clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.03)
history, post64 = [], []
best = (-1.0, None)
t0 = time.time()
for it in range(40):
    BETA = [8, 16, 32, 64][min(it // 10, 3)]
    adam.zero_grad()
    J = J_of(rho.clamp(0, 1), BETA)
    (-J).backward()
    adam.step()
    history.append(float(J))
    with torch.no_grad():
        Jp = float(J_of(rho.clamp(0, 1), 64.0))
    post64.append(Jp)
    if Jp > best[0]:
        best = (Jp, rho.detach().clamp(0, 1).clone())
    print(f"aafgc_ion_729  it={it} beta={BETA}: FE={float(J):.4f} "
          f"post64={Jp:.4f}")

# %% [markdown]
# ## Final design: FE and ion-plane spot size
# Repaint the best rho at beta=64, save it, and report the FE proxy plus the
# 1/e^2 spot width at the ion plane (paper: 2.22 um focus).

# %%
rho_f = best[1]
r = tanh_projection(conic_filter(rho_f, 0.12, dx), 64.0)
eps_fin = region.paint(base, r).detach()
with torch.no_grad():
    FE_fin, E_fin = FE_of(eps_fin)
FE_fin = float(FE_fin)
np.save(OUT / "aafgc_ion_729_rho.npy", rho_f.cpu().numpy())
print(f"aafgc_ion_729  BEST FE proxy = {FE_fin:.4f} "
      f"({10*math.log10(max(FE_fin,1e-12)):.1f} dB; paper AAFGC: -3.8 dB "
      f"with metals+ITO omitted here)")

# spot size at the ion plane (1/e^2 of the intensity lobe)
I_line = (E_fin[:, ia_z].abs() ** 2).cpu().numpy()
pk = I_line.max()
above = np.where(I_line > pk / math.e ** 2)[0]
spot = (above.max() - above.min()) * dx if len(above) else float("nan")
print(f"aafgc_ion_729  ion-plane spot (1/e^2) = {spot:.2f} um "
      f"(paper: 2.22 um focus, DL 2 um)")

# %% [markdown]
# ## FDTD cross-check
# Independent time-domain runs (device + bare-stack reference) with the same
# mode source; the DFT'd Ez at FCEN should reproduce the FDFD FE.

# %%
# --- FDTD cross-check ---
DEV = "cuda" if torch.cuda.is_available() else "cpu"
grid = Grid(nx=nx, ny=nz, dx=dx, device=DEV)
coeffs = build_tm_coeffs(grid, eps_inf=eps_fin.to(DEV), npml=NPML)
src = ModeSourceTM(grid, coeffs, p_in, GaussianPulse(FCEN, 0.2), FCEN)
mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
T_RUN = 2 * (5 / 0.2) + (LX + LZ) * 1.6 + 60
with torch.no_grad():
    _, (acc,) = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                       source=src, monitors=[mon], checkpoint_every=400)
ez = (acc.cpu()[0, 0] + 1j * acc.cpu()[0, 1])
coeffs_r = build_tm_coeffs(grid, eps_inf=base.to(DEV), npml=NPML)
src_r = ModeSourceTM(grid, coeffs_r, p_in, GaussianPulse(FCEN, 0.2), FCEN)
mon_r = DFTFieldsMonitor(grid, [FCEN], "Ez")
with torch.no_grad():
    _, (acc_r,) = run_tm(init_tm_state(grid), coeffs_r,
                         int(round(T_RUN / grid.dt)), source=src_r,
                         monitors=[mon_r], checkpoint_every=400)
ez_r = (acc_r.cpu()[0, 0] + 1j * acc_r.cpu()[0, 1])
P_in_t = float((np.abs(ez_r.cpu().numpy()[icol, k0f - SPAN:k1f + SPAN]) ** 2).sum())
FE_t = float((np.abs(ez.cpu().numpy()[ia_x - WIN:ia_x + WIN, ia_z]) ** 2).sum()) / P_in_t
rel = abs(FE_fin - FE_t) / max(FE_t, 1e-12)
print(f"aafgc_ion_729  FDTD cross-check FE = {FE_t:.4f} "
      f"({10*math.log10(max(FE_t,1e-12)):.1f} dB, rel diff {rel*100:.1f}%)")

# %% [markdown]
# ## Plot
# Four panels: optimization curve vs the paper's -3.8 dB, the design (chip
# zoom), the FDFD |Ez| beam to the ion, and the ion-plane intensity profile.

# %%
fig, axs = plt.subplots(2, 2, figsize=(13, 8))
a1, a2, a3, a4 = axs.flat
a1.plot(10 * np.log10(np.maximum(post64, 1e-12)), "bo-", label="post-step @64")
a1.axhline(-3.8, color="r", lw=0.8, label="paper AAFGC -3.8 dB")
a1.set_xlabel("iteration"); a1.set_ylabel("FE proxy [dB]")
a1.set_title(f"FDFD {10*math.log10(max(FE_fin,1e-12)):.1f} dB | "
             f"FDTD {10*math.log10(max(FE_t,1e-12)):.1f} dB")
a1.legend(fontsize=7)
a2.imshow(eps_fin.cpu().numpy()[:, :int(10 * RES)].T, origin="lower", cmap="viridis",
          extent=[0, LX, 0, 10], aspect="auto")
a2.set_title("design (chip zoom, z<10 um)")
a3.imshow(E_fin.abs().cpu().numpy().T, origin="lower", cmap="inferno",
          extent=[0, LX, 0, LZ], aspect="auto", vmax=float(E_fin.abs().max()) * 0.25)
a3.plot([ia_x * dx], [ia_z * dx], "c*", ms=12)
a3.set_title("|Ez| FDFD -- beam to the ion (star)")
a4.plot(np.arange(nx) * dx, I_line / pk)
a4.axvline(ia_x * dx, color="r", lw=0.8)
a4.set_xlabel("x [um]"); a4.set_title(f"ion-plane profile, spot {spot:.1f} um")
for a in (a3,):
    a.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "aafgc_ion_729.png", dpi=150)
print("aafgc_ion_729.png written")
