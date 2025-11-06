# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2): inverse-designed Y splitter on the REAL SiN platform, 674 nm.

User/peer spec 2026-08-15: W = 4.0 um waveguides, lambda = 674 nm (red diode),
SiN 150 nm slab lateral effective index, indices from the meep materials
library (Si3N4 Luke, SiO2 Horiba). Geometry (gds, symmetric about y=0):
  straight(4) input + mmi1x2(width=4, width_taper=4, width_mmi=9, length_mmi=12,
  gap_mmi=0.25) + bend_s(size=(4, +/-0.375)) -> arm centers +/-2.5 um (5 um sep)
  + straight(4) outputs. The mmi1x2 ports sit at +/-2.125 (two 4 um ports, 0.25
  um gap) and the bend_s raises them to +/-2.5.

Half-domain even-mirror (y_mirror=+1, msg 1f31b286): the device is symmetric
about y=0 (the input-guide centerline / splitter axis), so only the UPPER half
is simulated with a PMC wall at the cut (fd1d bc=pmc_lo, build_tm_coeffs
y_mirror=+1, init_tm_state y_mirror=+1). S31 = S21 by symmetry, total transmission
= 2 x |S21|^2, imbalance = 0 by construction. This halves ny -> ~2x faster + 2x
less memory. The full symmetric comp is rasterized, then the upper half
(j >= cy_full, where cy_full is the grid j of comp y=0) is cropped.

Normalization: decompose |a|^2 and FluxMonitor flux are DIFFERENT scales (coupler
measured |a|^2/flux ~ 1.9), so the objective + reported efficiency are
mode-overlap RATIOS (scale-free): S21 = a_up / a_in (both decompose, same
port/mode norm N -> ratio clean). total = 2 x |S21|^2.

Run: uv run python examples_2/c_y_splitter.py
"""
import math
import os
import pathlib
import sys
import time

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(24)   # peer: up to 24 cores while rings held (taskset 0-23)
# CUDA (peer 803d4094, GPU): component_to_eps rasterizes on CPU;
# the half-domain Grid is built with device=DEV and eps moved after the crop.
# decompose takes .cpu() tensors; solve_1d stays numpy/CPU.
DEV = os.environ.get("CYS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.adjoint import DesignRegion, MaterialGrid, rho_from_eps
from meep_torch.adjoint.filters import conic_filter, tanh_projection
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import DFTFieldsMonitor, FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

# --- 674 nm real platform: 150 nm SiN slab -> lateral effective index ---
WL = 0.674
FCEN = 1 / WL
N_SIN = float(Si3N4.n(FCEN).real)      # ~2.0347 (Luke, dispersive)
N_CLAD = float(SiO2.n(FCEN).real)      # ~1.4652 (Horiba)
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2     # 15 cells x 0.01 um = 150 nm slab
NEFF = float(solve_1d(eps_line, 0.01, FCEN, "te", k=1).neff[0])   # ~1.6321
W_WG = 4.0
print(f"c_y_splitter  SiN 150nm/SiO2 @{WL*1e3:.0f}nm: n_core={N_SIN:.4f} "
      f"n_clad={N_CLAD:.4f} -> slab NEFF={NEFF:.4f}, W={W_WG} um")

# --- gdsfactory symmetric Y: straight(4) in + mmi1x2 + bend_s + straight(4) out ---
xs = gf.get_cross_section("strip").copy(width=W_WG)
mmi = gf.components.mmi1x2(width=W_WG, width_taper=W_WG, width_mmi=9.0,
                           length_mmi=12.0, gap_mmi=0.25, cross_section=xs)
comp = gf.Component("y_splitter_v5")
mref = comp.add_ref(mmi)
inp = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
inp.connect("o2", mref.ports["o1"])                  # input straight -> mmi o1
comp.add_port("o1", port=inp.ports["o1"])
bup = comp.add_ref(gf.components.bend_s(size=(4.0, 0.375), cross_section=xs))
bup.connect("o1", mref.ports["o2"])                  # upper bend -> +2.5
oup = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
oup.connect("o1", bup.ports["o2"])
comp.add_port("o2", port=oup.ports["o2"])
bdn = comp.add_ref(gf.components.bend_s(size=(4.0, -0.375), cross_section=xs))
bdn.connect("o1", mref.ports["o3"])                  # lower bend -> -2.5 (mirror)
odn = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
odn.connect("o1", bdn.ports["o2"])
comp.add_port("o3", port=odn.ports["o2"])
comp.flatten()

# RES=20 (not the peer's >=40): the run.py checkpoint path is O(n_steps/chunk x
# state) with use_reentrant=False (see wiki bugs/checkpoint-boundary-state-
# accumulation), so a RES=40 half-domain (~500k cells) adjoint wants >31 GB ->
# infeasible on this box. RES=20 half-domain ~126k cells fits 16 G at CHKPT=128
# (proven on the 674 nm coupler, 0.69x those cells). The 0.25 um mmi gap = 5
# cells at dx=50 nm -- borderline but resolved; mmi body (9x12 um) is 180x240
# cells, fine. Optimize coarse RES=20; a forward-only RES=40 validation (no_grad,
# O(chunk)) can follow once the design converges. (peer informed.)
RES = 20
dx = 1.0 / RES                                        # 50 nm
NPML = RES                                            # 1 um (20 cells)
# rasterize the FULL symmetric comp, then crop the upper half at the splitter
# axis (comp y=0 = grid j=o1y_full, the input-guide centerline).
grid_f, eps_f, ports_f = component_to_eps(comp, res=RES, pad=3.0,
                                          n_core=NEFF, n_clad=N_CLAD)
nxf, nyf = grid_f.nx, grid_f.ny
o1xf, o1yf, _ = ports_f["o1"]                         # input, y=0 (axis)
o2xf, o2yf, _ = ports_f["o2"]                         # upper output, y=+2.5
cy_full = o1yf                                        # grid j of comp y=0 (cut plane)
# o2/o3 symmetric about the axis (o1): upper arm at +2.5, lower at -2.5 ->
# (o2y + o3y) / 2 == o1y. Integer-exact when the comp is symmetric and bbox
# aligns (2*cy_full == o2yf + o3y).
_o3yf = ports_f["o3"][1]
assert 2 * cy_full == o2yf + _o3yf, (
    f"o2/o3 not symmetric about o1: o1y={cy_full} o2y={o2yf} o3y={_o3yf}")
# upper-half crop -> half grid (PMC wall at j=0, the cut)
eps_base = eps_f[:, cy_full:].clone().to(DEV)
nx, ny = nxf, nyf - cy_full
grid = Grid(nx=nx, ny=ny, dx=dx, device=DEV)
o1x = o1xf
o2x, o2y = o2xf, o2yf - cy_full                        # upper-arm port in half grid
# extend the output arm INTO the right PML (same fix as the coupler, peer
# d79ef100): with symmetric pad the gds straight ends pad-NPML*dx short of the
# PML, so the arm tip reflects (Fabry-Perot) back into the TE0 monitor. Paint
# the W_WG core from o2x to the right edge -- past the monitor at o2x-2 um, and
# outside the design region, so an optimized rho stays valid.
_h = int(W_WG * RES / 2)
eps_base[o2x:, o2y - _h:o2y + _h] = NEFF ** 2
T_RUN = 130.0
print(f"c_y_splitter  full grid {nxf}x{nyf} -> half {nx}x{ny} "
      f"({nx*dx:.1f} x {ny*dx:.1f} um), cut at full-j={cy_full}, on {DEV}")

# design region: mmi body + bend (between the 4 um input/output straights),
# upper half y 0..5 um. input straight ends at o1x+4*RES (mmi o1); output
# straight starts at o2x-4*RES (after bend). Region x = mmi+bend = 16 um.
ix0 = o1x + int(4.0 * RES)                             # mmi o1 (after input straight)
ix1 = o2x - int(4.0 * RES)                             # bend end (before output straight)
iy0 = 0                                               # cut plane (PMC wall)
iy1 = int(5.0 * RES)                                  # cover mmi top (+4.5) + bend (+2.5)
NRX, NRY = int((ix1 - ix0) / 3.2), int((iy1 - iy0) / 3.2)   # ~160 nm rho pixels @ RES=20
mg = MaterialGrid((NRX, NRY), eps_min=N_CLAD ** 2, eps_max=NEFF ** 2)
region = DesignRegion(mg, ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1)
seed = rho_from_eps(eps_base, region)                 # imported layout as rho seed

# TE0 source 2 um from input outer end; TE0 monitor 2 um from output outer end.
# Half-domain: p_in hugs the cut (lo=0), bc=pmc_lo everywhere.
SPAN = int(2.5 * RES)                                 # 2.5 um half-window (4 um guide + tail)
p_in = Port(axis="x", index=o1x + int(2.0 * RES), lo=0, hi=SPAN, direction=+1)
p_up = Port(axis="x", index=o2x - int(2.0 * RES), lo=o2y - SPAN,
            hi=o2y + SPAN, direction=-1)
# Half-domain: lo=0 is the PMC cut (even mirror), NOT PML, so validate_ports
# (which assumes PML on all sides) mis-flags lo=0. Manual check: hi below the
# top PML, indices in x range. (test_y_mirror.py validates the y_mirror method.)
assert p_in.hi <= ny - NPML and p_up.hi <= ny - NPML, "port hi reaches top PML"
assert 0 <= p_in.index < nx and 0 <= p_up.index < nx, "port index out of x range"
# fwidth scaled with wavelength (peer: 0.2*1.55/0.674 ~ 0.46, keep relative bw)
pulse = GaussianPulse(FCEN, 0.2 * 1.55 / WL)
BC = "pmc_lo"                                         # even-mirror half-domain
YM = +1
# CHKPT sized for the RES=20 half grid (~126k cells): same O(n_steps/chunk x
# state) wall as the coupler, ~0.6x those cells -> ~7 GB at CHKPT=128, fits 16G.
CHKPT = 128


def s_params(eps):
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML, y_mirror=YM)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN, bc=BC)
    mons = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index,
                          lo=p.lo, hi=p.hi) for p in (p_in, p_up)]
    _, accs = run_tm(init_tm_state(grid, y_mirror=YM), coeffs,
                     int(round(T_RUN / grid.dt)), source=src, monitors=mons,
                     checkpoint_every=CHKPT)
    out = []
    for p, acc in zip((p_in, p_up), accs):
        el = eps.detach()[p.index, p.lo:p.hi]
        out.append(decompose(acc.cpu(), el.cpu(), dx, [FCEN], "tm", 1, axis_sign=-1.0, bc=BC))
    a_in = out[0][0][0, 0]
    return out[1][0][0, 0] / a_in, out[0][1][0, 0] / a_in   # (S21, S11)


def objective(rho):
    r = tanh_projection(conic_filter(rho, 0.15, dx), BETA)
    s21, _ = s_params(region.paint(eps_base, r))
    return (s21.conj() * s21).real                     # |S21|^2 (upper); total=2x by sym


# CYS_VALIDATE_ONLY=1 -> skip the optimize loop and re-eval/replot from the
# saved rho_it*.npy (the best-iterate block below reads them off disk anyway).
# Used for presentation replots without re-running or fighting for the GPU.
OPTIMIZE = os.environ.get("CYS_VALIDATE_ONLY") != "1"
rho = seed.clone().requires_grad_(True)
adam = torch.optim.Adam([rho], lr=0.03)
history = []
if OPTIMIZE:
    print(f"c_y_splitter  [starting 16-iter adjoint, CHKPT={CHKPT}]")
for it in range(16 if OPTIMIZE else 0):
    BETA = [8, 16, 32, 64][min(it // 4, 3)]
    adam.zero_grad()
    t0 = time.perf_counter()
    J = objective(rho.clamp(0.0, 1.0))
    t1 = time.perf_counter()
    (-J).backward()
    t2 = time.perf_counter()
    adam.step()
    history.append(float(J))
    np.save(OUT / f"c_y_splitter_rho_it{it:02d}.npy",
            rho.detach().clamp(0, 1).cpu().numpy())          # per-iter rho: evolution video
    tot = 2.0 * float(J)                               # total = 2x|S21|^2 by symmetry
    print(f"c_y_splitter  it={it} beta={BETA}: |S21|^2={float(J)*100:.2f}% "
          f"total={tot*100:.2f}% (IL {-10*math.log10(max(tot,1e-12)):.2f} dB) "
          f"[fwd {t1-t0:.1f}s bwd {t2-t1:.1f}s]")
    if it == 0:
        print(f"c_y_splitter  ITER0 REPORT: grid {nx}x{ny}, CHKPT={CHKPT}, "
              f"~{(t1-t0 + t2-t1)/60:.1f}min/iter -> 16it ~{(t1-t0+t2-t1)*16/60:.0f}min. "
              f"GRINDING (peer: report grid+timing after iter0).")

# --- best-iterate selection (same fix as the coupler, peer d79ef100): the
# curve regresses (pre-reboot CPU run: it14 37.68% -> it15 37.61%), so the last
# iterate is not the best one. Re-eval every saved rho_it*.npy forward-only at
# BETA=64 and take the argmax. On GPU each eval is seconds. T_seed is the seed
# at BETA=64 too, so g2 compares like with like (not the it0 pre-step number).
def _eval_rho(rho_arr):
    r = tanh_projection(conic_filter(rho_arr, 0.15, dx), 64)
    with torch.no_grad():
        s21, _ = s_params(region.paint(eps_base, r))
    return float(abs(s21) ** 2)


T_seed = _eval_rho(seed)
best_it, best_j, rho_f = -1, -1.0, None
hist64 = []
for _p in sorted(OUT.glob("c_y_splitter_rho_it*.npy")):
    _it = int(_p.stem.split("_it")[1])
    _rho = torch.tensor(np.load(_p), dtype=eps_base.dtype, device=DEV).clamp(0, 1)
    _j = _eval_rho(_rho)
    hist64.append((_it, _j))
    if _j > best_j:
        best_j, best_it, rho_f = _j, _it, _rho
print(f"c_y_splitter  BEST iterate = it{best_it:02d} (|S21|^2={best_j*100:.2f}%, "
      f"total={2*best_j*100:.2f}%) -- using its rho; seed@BETA64={T_seed*100:.2f}%")
BETA = 64
# mirror-symmetrize about the cut (j=0): rho_f is already the upper half; the
# lower half is its mirror. No flip needed inside the half -- symmetrize just
# averages pixel noise: rho_f = 0.5*(rho_f + rho_f.flip(1)) would fold across
# the cut axis (j=0), which is NOT the device symmetry (arms are at +2.5, not
# mirrored within the half). Skip the flip -- the half-domain IS the symmetry.
r = tanh_projection(conic_filter(rho_f, 0.15, dx), BETA)
eps_fin = region.paint(eps_base, r).detach()
with torch.no_grad():
    s21, s11 = s_params(eps_fin)
    coeffs = build_tm_coeffs(grid, eps_inf=eps_fin, npml=NPML, y_mirror=YM)
    src = ModeSourceTM(grid, coeffs, p_in, pulse, FCEN, bc=BC)
    mon = DFTFieldsMonitor(grid, [FCEN], "Ez")
    _, (acc,) = run_tm(init_tm_state(grid, y_mirror=YM), coeffs,
                       int(round(T_RUN / grid.dt)), source=src,
                       monitors=[mon], checkpoint_every=CHKPT)
ez = np.abs((acc[0, 0] + 1j * acc[0, 1]).cpu().numpy())
T_up = float(abs(s21) ** 2)
tot = 2.0 * T_up                                       # S31 = S21 by symmetry
il_db = -10 * math.log10(tot)
np.save(OUT / "c_y_splitter_rho.npy", rho_f.cpu().numpy())
print(f"c_y_splitter  |S21|^2={T_up*100:.2f}% |S31|^2={T_up*100:.2f}% (sym) "
      f"total={tot*100:.2f}% IL {il_db:.2f} dB, imbalance 0.00% (sym), "
      f"|S11|^2={float(abs(s11)**2)*100:.2f}%")
# gates: g2 final > seed, g3 total > 50% (a decent splitter passes >50% each... actually total~1)
g2 = T_up > T_seed
g3 = tot > 0.5
print(f"c_y_splitter  GATES: g1 seed@BETA64={T_seed*100:.2f}% | "
      f"g2 best(it{best_it:02d})={T_up*100:.2f}% > seed={g2} | g3 total>50%={g3}")
assert g2, f"final {T_up:.4f} not > seed {T_seed:.4f}"

# full-device view (peer f0c3248e): the sim is a half-domain with a PMC at the
# cut, but a full Y reads better. Mirror j about the cut for DISPLAY only --
# eps[:, ::-1] stacked under eps reconstructs the even-mirror lower half.
def _mirror(a):
    return np.concatenate([a[:, ::-1], a], axis=1)


eps_full = _mirror(eps_fin.cpu().numpy())
ez_full = _mirror(ez)
fig, axs = plt.subplots(3, 2, figsize=(12, 11))
ax1, ax2, ax3, ax4, ax5, ax6 = axs.flat
ax1.plot([h[0] for h in hist64], [h[1] * 200 for h in hist64], "bo-",
         label="total per iter (post-step BETA=64)")
ax1.plot(best_it, best_j * 200, "r*", ms=15, label=f"best it{best_it:02d}")
ax1.axhline(T_seed * 200, color="g", ls="--", lw=0.9,
            label=f"gf mmi seed={T_seed*200:.1f}%")
ax1.set_xlabel("iteration"); ax1.set_ylabel(r"total $2|S_{21}|^2$ [%]")
ax1.legend(fontsize=7)
ax2.imshow(eps_base.cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax2.set_title("gdsfactory mmi1x2+bend_s seed (W=4um, upper half, PMC at bottom)")
ax3.imshow(eps_fin.cpu().numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax3.set_title(f"optimized design (best it{best_it:02d}, SiN 150nm @{WL*1e3:.0f}nm)")
ax4.imshow(ez.T, origin="lower", cmap="inferno", extent=[0, nx * dx, 0, ny * dx],
           aspect="auto")
ax4.set_title(rf"$|E_z|$ at {WL*1e3:.0f} nm (half-domain)")
_yf = 2 * ny * dx
ax5.imshow(eps_full.T, origin="lower", cmap="viridis", extent=[0, nx * dx, 0, _yf],
           aspect="auto")
ax5.set_title("full device eps (mirrored about symmetry plane for display)")
ax6.imshow(ez_full.T, origin="lower", cmap="inferno", extent=[0, nx * dx, 0, _yf],
           aspect="auto")
ax6.set_title(rf"full device $|E_z|$ (mirrored about symmetry plane for display)")
for ax in (ax2, ax3, ax4, ax5, ax6):
    ax.set_xlabel("x [um]"); ax.set_ylabel("y [um]")
fig.suptitle(f"Y splitter @{WL*1e3:.0f}nm: best it{best_it:02d} total={tot*100:.1f}%, "
             f"IL {il_db:.2f} dB, imbalance 0% (half-domain sym)", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "c_y_splitter.png", dpi=150)
plt.close(fig)
print("c_y_splitter.png written")