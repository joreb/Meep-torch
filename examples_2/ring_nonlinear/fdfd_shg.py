# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Undepleted SHG via chained FDFD solves
#
# Undepleted SHG via chained FDFD solves (nonlinear line revived by user
# 2026-08-17 for an FDFD test).
#
# Two LINEAR solves replace the time-domain buildup entirely:
#   1. pump Ez at f0 in the pump-band eps,
#   2. SH at 2*f0 in the SH-band eps, driven by Jz = -i*w_SH * chi2 * Ep^2
#      (the nonlinear polarization as a frequency-domain source).
#
# Undepleted approximation is exact by construction (the pump solve never
# sees chi2), and SH ~ amp^4 is exact by linearity of both solves.
#
# Straight TFLN-like guide (stack-assumed indices, per-band material
# dispersion -- FDFD evaluates eps(f) exactly, no pole fitting):
#
#   - g2: QPM-poled chi2 (sign flip every Lambda/2, Lambda = 2*pi/dk from the
#     SOLVED mode indices) must grow ~monotonically and beat the unpoled
#     guide by orders of magnitude at the far end.
#   - g3: the unpoled SH power must oscillate with period Lambda (the
#     coherence beat) -- measured period vs 2*pi/dk, convention-free.
#
# Run: `uv run python examples_2/ring_nonlinear/fdfd_shg.py`
# (writes `examples_2/ring_nonlinear/output/fdfd_shg.png`)

# %%
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

from meep_torch.fdfd import (make_mode_source, port_mode_line, solve_fdfd_tm)
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.port import Port

# %% [markdown]
# ## Guide, ports, and QPM period
# Straight 1.2 um guide with per-band core indices; the pump and SH mode
# profiles give dk = k_SH - 2 k_pump and the analytic QPM period Lambda.

# %%
# TFLN-like platform, stack-assumed 2.5D indices per band (stated, not fitted)
N_P_CORE, N_S_CORE = 2.01, 2.21          # pump 1348 nm, SH 674 nm
N_CLAD = 1.44
W = 1.2
F0 = 1 / 1.348
RES = 40
dx = 1.0 / RES
LX, LY = 40.0, 6.0
nx, ny = int(LX * RES), int(LY * RES)
NPML = RES
CHI2 = 1e-3                               # arbitrary units; gates are ratios

jw = int(W / 2 * RES)
cy = ny // 2


def guide_eps(n_core):
    e = torch.full((nx, ny), N_CLAD ** 2, dtype=torch.float64)
    e[:, cy - jw: cy + jw] = n_core ** 2
    return e


eps_p, eps_s = guide_eps(N_P_CORE), guide_eps(N_S_CORE)
core = (eps_s > N_CLAD ** 2 + 1e-9)

p_in = Port(axis="x", index=NPML + 8, lo=cy - int(1.5 * RES),
            hi=cy + int(1.5 * RES), direction=+1)
prof_p, neff_p = port_mode_line(eps_p, p_in, F0, dx)
prof_s, neff_s = port_mode_line(eps_s, p_in, 2 * F0, dx)
dk = 2 * math.pi * (2 * F0) * (neff_s - neff_p)     # k_SH - 2 k_pump
LAM = 2 * math.pi / dk                              # QPM period = beat period
print(f"fdfd_shg  neff_p={neff_p:.4f} neff_s={neff_s:.4f} "
      f"dn={neff_s-neff_p:.4f} Lambda_QPM={LAM:.3f} um")

# %% [markdown]
# ## The two chained solves
# Solve 1: linear pump field. Solve 2: SH driven by the nonlinear
# polarization Jz = -i*w_SH*chi2*Ep^2, first unpoled to measure the grid's
# actual coherence beat.

# %%
# --- solve 1: pump ---
Ep = solve_fdfd_tm(eps_p, dx, F0, make_mode_source(eps_p, p_in, F0, dx), NPML)

# --- solve 2: SH, unpoled first (measures the GRID's actual beat) ---
x_um = (torch.arange(nx, dtype=torch.float64) - p_in.index) * dx
w_sh = 2 * math.pi * (2 * F0)


def sh_solve(sign_mask):
    chi2 = CHI2 * core.to(torch.float64) * sign_mask
    jz = -1j * w_sh * chi2.to(torch.complex128) * Ep ** 2
    return solve_fdfd_tm(eps_s, dx, 2 * F0, jz, NPML)

Es_unp = sh_solve(torch.ones_like(x_um)[:, None])

# --- SH mode power vs propagation distance (single-solve scan) ---
lo, hi = p_in.lo, p_in.hi
profc = prof_s.to(torch.complex128)
def a_of_x(Es):
    return (Es[:, lo:hi] * profc[None, :]).sum(dim=1) * dx

xs = np.arange(nx) * dx
i0 = p_in.index + int(1.0 * RES)                    # start past the source
i1 = nx - NPML - int(1.0 * RES)
P_unp = (a_of_x(Es_unp).abs() ** 2).cpu().numpy()

# %% [markdown]
# ## Self-calibrated poling
# Numerical dispersion shifts dk by ~8% at this resolution, so the poling
# period is measured from the unpoled beat itself (what a lab does too);
# poling at the analytic Lambda desyncs after ~pi/delta_k.

# %%
# measure the grid's actual beat period from the unpoled oscillation --
# numerical dispersion shifts dk by ~8% at 12 cells/lambda_SH, and poling
# at the analytic Lambda loses QPM sync after ~pi/delta_k (measured first
# failure mode of this script: gain 8x instead of hundreds). Poling at the
# MEASURED period is also what a lab does.
# Peer-verified mechanism (0.1% agreement): 2nd-order FD inflates
# n_num/n by arcsin(k dx/2)/(k dx/2); the SH at 12 cells/lambda gets
# x1.0118 while the pump at ~24 cells gets only x1.0029, and the beat
# depends on the DIFFERENCE, so the asymmetry widens dn 0.2355->0.2560
# (measured 0.2560). The error scales with the SH/pump resolution ratio:
# doubling RES shrinks it ~4x (24 cells/lambda_SH -> ~2% beat error).
# A fixed analytic-Lambda poling desyncs after ~1/(dn error) domains, so
# self-calibration is mandatory at ring scale (183 domains over 314 um).
seg = P_unp[i0:i1]
pk = [k for k in range(1, len(seg) - 1) if seg[k] > seg[k - 1] and seg[k] >= seg[k + 1]
      and seg[k] > 0.1 * seg.max()]
period = float(np.mean(np.diff(pk)) * dx) if len(pk) >= 3 else float("nan")
dk_grid = 2 * math.pi / period
poling = torch.sign(torch.cos(dk_grid * x_um))[:, None]
Es_pol = sh_solve(poling)
P_pol = (a_of_x(Es_pol).abs() ** 2).cpu().numpy()

# %% [markdown]
# ## Gates and plot
# g2: poled/unpoled contrast at the far end; g3: unpoled beat period vs
# Lambda; g1 is exact by linearity. Field map plus power-vs-x curves.

# %%
# g2: poled/unpoled contrast at the far end (average over the last beat)
nlast = max(int(LAM * RES), 8)
far_pol = float(P_pol[i1 - nlast: i1].mean())
far_unp = float(P_unp[i1 - nlast: i1].mean())
g2 = far_pol / far_unp
print(f"fdfd_shg  g2 poled/unpoled at {xs[i1]:.0f} um: {g2:.0f}x "
      f"(poled {far_pol:.3e}, unpoled {far_unp:.3e})")
print(f"fdfd_shg  g3 unpoled beat period {period:.3f} um vs Lambda {LAM:.3f} um "
      f"({abs(period-LAM)/LAM*100:.1f}% off, {len(pk)} peaks)")
print("fdfd_shg  g1 SH ~ amp^4: exact by linearity of both solves (pump solve "
      "is chi2-blind; doubling amp scales Ep by 2, Jz_SH by 4, P_SH by 16)")
assert g2 > 50, f"QPM gain too small: {g2}"
assert abs(period - LAM) / LAM < 0.15, f"beat period {period} vs {LAM}"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
ax1.imshow(np.abs(Es_pol.cpu().numpy()).T, origin="lower", cmap="inferno",
           extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax1.set_title(f"|E(2f0)| QPM-poled (Lambda={LAM:.2f} um)")
ax2.semilogy(xs[i0:i1], P_pol[i0:i1], label="poled (QPM)")
ax2.semilogy(xs[i0:i1], P_unp[i0:i1], label="unpoled")
ax2.set_xlabel("x [um]"); ax2.set_ylabel("SH mode power [arb]")
ax2.set_title(f"g2: {g2:.0f}x at {xs[i1]:.0f} um; "
              f"g3: beat {period:.2f} vs Lambda {LAM:.2f} um")
ax2.legend()
fig.tight_layout()
fig.savefig(OUT / "fdfd_shg.png", dpi=150)
print("fdfd_shg.png written")
