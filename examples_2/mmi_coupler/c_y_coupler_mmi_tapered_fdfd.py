# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # MMI Y coupler: FDFD inverse design
#
# FDFD inverse design of the tuned mmi_tapered Y coupler -- exact adjoint
# (scipy LU direct solve + one transposed triangular solve for the
# gradient) via meep_torch.fdfd, mirroring c_y_coupler_mmi_tapered.py's
# FDTD optimization stage. CPU only (solve_fdfd_tm's default "scipy"
# backend is the differentiable path).
#
# The exec'd head runs the FDTD characterization of the tuned geometry (a
# couple of runs, possibly on GPU) before this script ever gets control --
# that is expected and harmless.
#
# The head is split before the optimization banner (so the design region
# is rebuilt fresh here, not shared with the FDTD opt stage) AND before
# the head's own S-parameter function for safety, same double-split as
# c_y_coupler_mmi_tapered_3d.py. Its own objective ports use
# SPAN=int(1.8*RES) and, with outputs at +/-1.32 um, that window overlaps
# the neighboring arm -- so this script builds its OWN ports at
# SPAN=int(1.2*RES) (team spec) and uses them uniformly for both the FDFD
# objective and the FDTD cross-check; the exec'd head's ports are not
# reused anywhere below.
#
# Every FDFD percentage is a ratio against a_ref (see c_y_splitter_fdfd.py
# docstring); the SEED design is cross-checked against FDTD before the 24
# FDFD iterations run.
#
# Run: `uv run python examples_2/mmi_coupler/c_y_coupler_mmi_tapered_fdfd.py`

# %%
import math
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Reuse the 2D characterization head
# Exec the tuned-geometry setup as source text, bind the fast pardiso
# FDFD backend, and pull the shared constants out of the exec'd globals.

# %%
src_txt = open(HERE / "c_y_coupler_mmi_tapered.py").read()
head = src_txt.split("# --- inverse-design optimization")[0]
head = head.split("def s_params_opt(")[0]
g = {"__file__": str(HERE / "c_y_coupler_mmi_tapered.py")}
exec(compile(head, "mmi_head", "exec"), g)

from meep_torch.adjoint import DesignRegion, MaterialGrid, rho_from_eps
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
import functools
from meep_torch.fdfd import make_mode_source, mode_amplitude
from meep_torch.fdfd import solve_fdfd_tm as _solve
solve_fdfd_tm = functools.partial(_solve, backend="pardiso")  # 12x MKL
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

torch.set_num_threads(8)

grid, ports = g["grid"], g["ports"]
RES, dx, FCEN, N_CLAD, NEFF, NPML = (g["RES"], g["dx"], g["FCEN"],
                                     g["N_CLAD"], g["NEFF"], g["NPML"])
T_RUN = g["T_RUN"]
nx, ny = grid.nx, grid.ny
o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
o3x, o3y, _ = ports["o3"]

eps_base = g["eps"].detach().cpu().to(torch.float64)

# %% [markdown]
# ## Design region and non-overlapping ports
# The design region is rebuilt fresh (the split dropped the head's own),
# and this script's OWN SPAN=1.2*RES ports keep the two output windows
# from overlapping between the +/-1.32 um arms.

# %%
# design region: rebuilt fresh (the split dropped the head's own
# rebuild), exactly the box c_y_coupler_mmi_tapered_3d.py lines 51-54 use.
ix0 = o1x + int((4.0 + 10.0) * RES)
ix1 = min(o2x - int(2.0 * RES), nx - NPML - 4)
iy0 = int(max(o3y - 1.6 * RES, NPML))
iy1 = int(min(o2y + 1.6 * RES, ny - NPML))
NRX, NRY = int((ix1 - ix0) * dx / 0.08), int((iy1 - iy0) * dx / 0.08)
mg = MaterialGrid((NRX, NRY), eps_min=N_CLAD ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
seed = rho_from_eps(eps_base, region)
print(f"c_y_coupler_mmi_tapered_fdfd  design {NRX}x{NRY} rho pixels (~80 nm)")

# OWN objective ports (SPAN=1.2*RES, team spec) -- NOT g["p_in"]/p_up/p_dn,
# whose SPAN=1.8*RES windows overlap between the +/-1.32 um output arms.
SPAN = int(1.2 * RES)
p_in = Port(axis="x", index=max(o1x + int(2.0 * RES), NPML + 4),
            lo=o1y - SPAN, hi=o1y + SPAN, direction=+1)
p_up = Port(axis="x", index=min(o2x - int(2.0 * RES), nx - NPML - 4),
            lo=o2y - SPAN, hi=o2y + SPAN, direction=-1)
p_dn = Port(axis="x", index=min(o3x - int(2.0 * RES), nx - NPML - 4),
            lo=o3y - SPAN, hi=o3y + SPAN, direction=-1)
validate_ports([p_in, p_up, p_dn], grid, NPML)
assert p_up.lo >= p_dn.hi or p_dn.lo >= p_up.hi, \
    f"output port windows overlap: up=[{p_up.lo},{p_up.hi}) dn=[{p_dn.lo},{p_dn.hi})"
pulse = GaussianPulse(FCEN, 0.2)  # NOTE: no module-level pulse in the mmi script

# %% [markdown]
# ## Reference solve and normalization
# A straight guide with the exact input-port eps profile fixes a_ref, the
# amplitude every FDFD percentage is measured against.

# %%
# Reference straight guide: the exact eps profile at p_in's port line,
# tiled across all x (exact mode match to jz_in, see c_y_splitter_fdfd.py).
ref_eps = eps_base[p_in.index, :].unsqueeze(0).expand(nx, ny).clone()
p_ref = Port(axis="x", index=p_up.index, lo=o1y - SPAN, hi=o1y + SPAN, direction=-1)

jz_in = make_mode_source(eps_base, p_in, FCEN, dx)

t0 = time.time()
Ez_ref = solve_fdfd_tm(ref_eps, dx, FCEN, jz_in, NPML)
solve_t = time.time() - t0
print(f"c_y_coupler_mmi_tapered_fdfd  one FDFD solve: {solve_t:.1f}s "
      f"(grid {nx}x{ny}={nx*ny} cells, ~{48*solve_t/60:.0f} min budget for 24 it)")
a_ref = mode_amplitude(Ez_ref, ref_eps, p_ref, FCEN, dx).detach()

# %% [markdown]
# ## FDFD objective and FDTD cross-check
# One differentiable FDFD solve per objective evaluation; the FDTD
# cross-check uses the same SPAN=1.2*RES windows on both sides.

# %%
def fdfd_TuTd(eps):
    Ez = solve_fdfd_tm(eps, dx, FCEN, jz_in, NPML)
    a_up = mode_amplitude(Ez, eps, p_up, FCEN, dx)
    a_dn = mode_amplitude(Ez, eps, p_dn, FCEN, dx)
    ru, rd = a_up / a_ref, a_dn / a_ref
    return (ru.conj() * ru).real, (rd.conj() * rd).real


def total_T(rho, beta):
    r = tanh_projection(conic_filter(rho, 0.12, dx), beta)
    eps = region.paint(eps_base, r)
    Tu, Td = fdfd_TuTd(eps)
    return 2 * torch.sqrt(Tu * Td + 1e-12)


def fdtd_TuTd(eps_full):
    """Same shape as c_y_coupler_mmi_tapered.py's s_params_opt (lines
    159-172), but using OUR p_in/p_up/p_dn (SPAN=1.2*RES) throughout so
    the FDFD-vs-FDTD comparison measures the same windows on both sides."""
    dev_eps = eps_full.to(grid.device, dtype=grid.dtype)
    coeffs = build_tm_coeffs(grid, eps_inf=dev_eps, npml=NPML)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN)
    mons = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index,
                          lo=p.lo, hi=p.hi) for p in (p_in, p_up, p_dn)]
    with torch.no_grad():
        _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                         source=src, monitors=mons, checkpoint_every=400)
    out = []
    for p, acc in zip((p_in, p_up, p_dn), accs):
        el = dev_eps.detach().cpu()[p.index, p.lo:p.hi]
        out.append(decompose(acc.cpu(), el, dx, [FCEN], "tm", 1, axis_sign=-1.0))
    a_in = out[0][0][0, 0]
    s_up = out[1][0][0, 0] / a_in
    s_dn = out[2][0][0, 0] / a_in
    return float(abs(s_up) ** 2), float(abs(s_dn) ** 2)


# %% [markdown]
# ## Seed cross-check
# FDFD vs FDTD on the unoptimized seed, before spending 24 iterations.

# %%
# SEED cross-check first (same rationale as c_y_splitter_fdfd.py).
with torch.no_grad():
    Tu_s, Td_s = fdfd_TuTd(eps_base)
tot_fdfd_seed = float(Tu_s + Td_s)
Tu_fdtd_s, Td_fdtd_s = fdtd_TuTd(eps_base)
tot_fdtd_seed = Tu_fdtd_s + Td_fdtd_s
rel_seed = abs(tot_fdfd_seed - tot_fdtd_seed) / max(tot_fdtd_seed, 1e-12)
print(f"c_y_coupler_mmi_tapered_fdfd  SEED  FDFD total={tot_fdfd_seed*100:.1f}% "
      f"FDTD total={tot_fdtd_seed*100:.1f}% rel_diff={rel_seed*100:.1f}%"
      + ("  WARNING: >7%" if rel_seed > 0.07 else ""))

# %% [markdown]
# ## Optimization loop
# 24 Adam steps with beta continuation; the best post-step iterate at
# BETA=64 is kept.

# %%
rho = seed.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.03)
history, post64 = [], []
best = (-1.0, None)
for it in range(24):
    BETA = [8, 16, 32, 64][min(it // 6, 3)]
    adam.zero_grad()
    J = total_T(rho.clamp(0.0, 1.0), BETA)
    (-J).backward()
    adam.step()
    history.append(float(J))
    with torch.no_grad():
        Jp = float(total_T(rho.clamp(0.0, 1.0), 64.0))
    post64.append(Jp)
    if Jp > best[0]:
        best = (Jp, rho.detach().clamp(0, 1).clone())
    print(f"c_y_coupler_mmi_tapered_fdfd  it={it} beta={BETA}: "
          f"J={float(J)*100:.1f}% post64={Jp*100:.1f}%")

# %% [markdown]
# ## Final evaluation and FDTD cross-check
# The best design is re-measured in both engines; rho lands in
# `output/c_y_coupler_mmi_tapered_fdfd_rho.npy`.

# %%
rho_f = best[1]
r = tanh_projection(conic_filter(rho_f, 0.12, dx), 64.0)
eps_fin = region.paint(eps_base, r).detach()
with torch.no_grad():
    Tu, Td = fdfd_TuTd(eps_fin)
    Ez_fin = solve_fdfd_tm(eps_fin, dx, FCEN, jz_in, NPML)
Tu, Td = float(Tu), float(Td)
tot = Tu + Td
il_db = -10 * math.log10(max(tot, 1e-12))
imb = abs(Tu - Td) / max(tot, 1e-12)
np.save(OUT / "c_y_coupler_mmi_tapered_fdfd_rho.npy", rho_f.numpy())
print(f"c_y_coupler_mmi_tapered_fdfd  BEST Tu={Tu*100:.1f}% Td={Td*100:.1f}% "
      f"total={tot*100:.1f}% (IL {il_db:.2f} dB), imbalance {imb*100:.2f}%")

Tu_fdtd, Td_fdtd = fdtd_TuTd(eps_fin)
tot_fdtd = Tu_fdtd + Td_fdtd
rel = abs(tot - tot_fdtd) / max(tot_fdtd, 1e-12)
print(f"c_y_coupler_mmi_tapered_fdfd  FINAL FDFD total={tot*100:.1f}% "
      f"FDTD total={tot_fdtd*100:.1f}% rel_diff={rel*100:.1f}%"
      + ("  WARNING: >7%" if rel > 0.07 else ""))

# %% [markdown]
# ## Plots
# Convergence, seed and optimized eps, and the FDFD steady-state |Ez|.

# %%
fig, axs = plt.subplots(2, 2, figsize=(13, 7))
ax1, ax2, ax3, ax4 = axs.flat
ax1.plot(np.array(post64) * 100, "bo-", label="post-step @BETA=64")
ax1.plot(np.array(history) * 100, "c.--", lw=0.8, label="pre-step")
i_best = int(np.argmax(post64))
ax1.plot([i_best], [post64[i_best] * 100], "r*", ms=14, label=f"best it{i_best}")
ax1.set_xlabel("iteration"); ax1.set_ylabel("2 sqrt(Tu Td) [%]")
ax1.set_title(f"FDFD: total {tot*100:.1f}%, IL {il_db:.2f} dB, imbalance {imb*100:.2f}%")
ax1.legend(fontsize=7)
ax2.imshow(eps_base.numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax2.set_title("tuned mmi_tapered seed")
ax3.imshow(eps_fin.numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax3.set_title("FDFD-optimized")
ax4.imshow(Ez_fin.abs().numpy().T, origin="lower", cmap="inferno",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax4.set_title(r"$|E_z|$ (FDFD steady-state)")
for ax in (ax3, ax4):
    ax.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "c_y_coupler_mmi_tapered_fdfd.png", dpi=150)
plt.close(fig)
print("c_y_coupler_mmi_tapered_fdfd.png written")
