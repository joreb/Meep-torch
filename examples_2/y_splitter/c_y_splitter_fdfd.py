# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)

# %% [markdown]
# # Y splitter: FDFD inverse design
#
# FDFD inverse design of the Y splitter -- exact adjoint (scipy LU direct
# solve + one transposed triangular solve for the gradient) via
# meep_torch.fdfd, mirroring c_y_splitter.py's FDTD optimization. CPU only
# (solve_fdfd_tm's default "scipy" backend is the differentiable path;
# "torch-sla"/real-backend are not used here).
#
# Every FDFD percentage below is a ratio against a_ref, the steady-state
# mode amplitude of a straight-reference solve -- a plain FDFD current
# sheet radiates both directions, so absolute amplitudes mean nothing on
# their own (see fdfd.py's make_mode_source docstring). Because the whole
# scale rests on that one number, the SEED design is cross-checked against
# the FDTD path (c_y_splitter.py's S-parameter routine, reimplemented
# inline since the exec'd head stops before that function) before spending
# 24 iterations on it; the FINAL design gets the same cross-check.
#
# Run: `uv run python examples_2/y_splitter/c_y_splitter_fdfd.py`

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
# ## Reuse the 2D setup head
# c_y_splitter.py's setup (grid, ports, design region, seed) is exec'd as
# source text; the FDFD solver is bound to the fast pardiso backend.

# %%
src_txt = open(HERE / "c_y_splitter.py").read()
head = src_txt.split("def s_params")[0]
g = {"__file__": str(HERE / "c_y_splitter.py")}
exec(compile(head, "splitter_head", "exec"), g)

from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.stepper_tm import init_tm_state
import functools
from meep_torch.fdfd import make_mode_source, mode_amplitude
from meep_torch.fdfd import solve_fdfd_tm as _solve
solve_fdfd_tm = functools.partial(_solve, backend="pardiso")  # 12x MKL
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

torch.set_num_threads(8)

grid, region, ports = g["grid"], g["region"], g["ports"]
FCEN, dx, NPML, RES = g["FCEN"], g["dx"], g["NPML"], g["RES"]
p_in, p_up, p_dn = g["p_in"], g["p_up"], g["p_dn"]
SPAN, pulse, T_RUN = g["SPAN"], g["pulse"], g["T_RUN"]
nx, ny = grid.nx, grid.ny
o1x, o1y, _ = ports["o1"]

# region/MaterialGrid hold only index math + floats -- device/dtype come
# entirely from the tensors handed to paint(), so the GPU-built region and
# seed from the exec'd head are reused as-is, just moved to CPU float64.
eps_base = g["eps_base"].detach().cpu().to(torch.float64)
seed = g["seed"].detach().cpu().to(torch.float64)

# %% [markdown]
# ## Reference solve and normalization
# A straight guide with the exact eps profile of the input port line
# fixes a_ref -- the amplitude every later solve is measured against.

# %%
# Reference straight guide: literally the eps profile at p_in's port line
# (the same column make_mode_source solves its mode on) tiled across all
# x -- guarantees the reference guide's mode is an exact match to jz_in,
# not just "same nominal width".
ref_eps = eps_base[p_in.index, :].unsqueeze(0).expand(nx, ny).clone()
p_ref = Port(axis="x", index=p_up.index, lo=o1y - SPAN, hi=o1y + SPAN, direction=-1)

jz_in = make_mode_source(eps_base, p_in, FCEN, dx)

t0 = time.time()
Ez_ref = solve_fdfd_tm(ref_eps, dx, FCEN, jz_in, NPML)
solve_t = time.time() - t0
print(f"c_y_splitter_fdfd  one FDFD solve: {solve_t:.1f}s "
      f"(grid {nx}x{ny}={nx*ny} cells, ~{48*solve_t/60:.0f} min budget for 24 it)")
a_ref = mode_amplitude(Ez_ref, ref_eps, p_ref, FCEN, dx).detach()

# %% [markdown]
# ## FDFD objective and FDTD cross-check
# fdfd_TuTd measures both output arms in one differentiable solve;
# fdtd_TuTd reimplements the FDTD S-parameter path so the two engines can
# be compared on identical windows.

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
    """Inline copy of c_y_splitter.py s_params (lines 112-126) -- the
    exec'd head stops before that def, so it isn't in scope."""
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
# FDFD vs FDTD totals on the unoptimized seed -- catches an a_ref
# normalization bug before 24 iterations get spent on it.

# %%
# SEED cross-check first: catches an a_ref normalization bug before 24
# FDFD iterations get spent chasing a miscalibrated objective.
with torch.no_grad():
    Tu_s, Td_s = fdfd_TuTd(eps_base)
tot_fdfd_seed = float(Tu_s + Td_s)
Tu_fdtd_s, Td_fdtd_s = fdtd_TuTd(eps_base)
tot_fdtd_seed = Tu_fdtd_s + Td_fdtd_s
rel_seed = abs(tot_fdfd_seed - tot_fdtd_seed) / max(tot_fdtd_seed, 1e-12)
print(f"c_y_splitter_fdfd  SEED  FDFD total={tot_fdfd_seed*100:.1f}% "
      f"FDTD total={tot_fdtd_seed*100:.1f}% rel_diff={rel_seed*100:.1f}%"
      + ("  WARNING: >7%" if rel_seed > 0.07 else ""))

# %% [markdown]
# ## Optimization loop
# 24 Adam steps with beta continuation; best post-step iterate at BETA=64
# is kept (same best-iterate rule as the FDTD script).

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
    print(f"c_y_splitter_fdfd  it={it} beta={BETA}: J={float(J)*100:.1f}% "
          f"post64={Jp*100:.1f}%")

# %% [markdown]
# ## Final evaluation and FDTD cross-check
# The best design is re-measured in both engines; rho lands in
# `output/c_y_splitter_fdfd_rho.npy`.

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
np.save(OUT / "c_y_splitter_fdfd_rho.npy", rho_f.cpu().numpy())
print(f"c_y_splitter_fdfd  BEST Tu={Tu*100:.1f}% Td={Td*100:.1f}% "
      f"total={tot*100:.1f}% (IL {il_db:.2f} dB), imbalance {imb*100:.2f}%")

Tu_fdtd, Td_fdtd = fdtd_TuTd(eps_fin)
tot_fdtd = Tu_fdtd + Td_fdtd
rel = abs(tot - tot_fdtd) / max(tot_fdtd, 1e-12)
print(f"c_y_splitter_fdfd  FINAL FDFD total={tot*100:.1f}% "
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
ax2.imshow(eps_base.cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax2.set_title("seed eps")
ax3.imshow(eps_fin.cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax3.set_title("FDFD-optimized")
ax4.imshow(Ez_fin.abs().cpu().numpy().T, origin="lower", cmap="inferno",
           extent=[0, nx*dx, 0, ny*dx], aspect="auto")
ax4.set_title(r"$|E_z|$ (FDFD steady-state)")
for ax in (ax3, ax4):
    ax.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "c_y_splitter_fdfd.png", dpi=150)
plt.close(fig)
print("c_y_splitter_fdfd.png written")
