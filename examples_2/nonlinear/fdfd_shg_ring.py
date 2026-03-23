# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""QPM-poled SHG RING via FDFD, cuDSS GPU backend (user request).

Chained linear solves on the full ring: pump solve at the ring's own
(numerically found) resonance, SH solve at 2*f_res driven by
Jz = -i*w_SH * chi2_poled * Ep^2. All solves go through
solve_fdfd_tm_real_backend(backend="cudss") -- complex system, 2N real
embedding, GPU direct factorization.

Ring: r = 10 um DEMO ring (same W=1.2/gap=0.25/TFLN-effective indices
as the b_ring design). The production r = 50 ring needs ~22M real
unknowns at an SH-resolving RES and cuDSS on a GPU tops out
near ~8M (measured ALLOC_FAILED at 9.4M) -- documented, not hidden.
strumpack's cuda12x wheel is GPU-mandatory and its dense kernels crash
at ring scale (cusolver internal error), so cuDSS it is.

Poling self-calibrated: dk measured from a straight-guide FDFD beat at
the SAME RES (grid numerical dispersion shifts dk; an analytic-Lambda
poling desyncs -- see fdfd_shg.py).

Run: uv run python examples_2/nonlinear/fdfd_shg_ring.py
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

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))
OUT = pathlib.Path(__file__).parent

import gdsfactory as gf
from gds_meep_torch import component_to_eps

from meep_torch.fdfd import (make_mode_source, port_mode_line,
                             solve_fdfd_tm, solve_fdfd_tm_real_backend)
from meep_torch.ports.port import Port

gf.gpdk.PDK.activate()

# TFLN-stack effective indices (b_ring_shg.py BANDS)
LAM_P, N_P, N_PC = 1.348, 2.0233, 1.7065
LAM_S, N_S, N_SC = 0.674, 2.2198, 2.0777
R_RING, W_WG, GAP = 15.0, 1.2, 0.25
RES = 40
dx = 1.0 / RES
NPML = RES
CHI2 = 1e-3
BE, DEV = "cudss", "cuda"

xs = gf.get_cross_section("strip").copy(width=W_WG)
# bend_circular: the default euler ring is a rounded square, which breaks
# the circular-annulus poling geometry (first-run png review)
comp = gf.components.ring_single(radius=R_RING, gap=GAP, length_x=0.01,
                                 bend=gf.components.bend_circular,
                                 cross_section=xs)


def band_eps(n_core, n_clad):
    grid, eps, ports = component_to_eps(comp, res=RES, pad=2.0,
                                        n_core=n_core, n_clad=n_clad)
    return eps, ports


eps_p, ports = band_eps(N_P, N_PC)
eps_s, _ = band_eps(N_S, N_SC)
nx, ny = eps_p.shape
print(f"fdfd_shg_ring  r={R_RING} grid {nx}x{ny} = {nx*ny/1e6:.1f}M complex "
      f"({2*nx*ny/1e6:.1f}M real), backend {BE}/{DEV}")

o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
SPAN = int(1.2 * RES)
p_in = Port(axis="x", index=max(o1x + 4, NPML + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
p_out = Port(axis="x", index=min(o2x - 4, nx - NPML - 4), lo=o2y - SPAN,
             hi=o2y + SPAN, direction=-1)
jz_p = make_mode_source(eps_p, p_in, 1 / LAM_P, dx)

# ring-core mask = actual core minus the bus band; ring center from the
# mask centroid (robust to the exact layout, first-run mask missed arcs)
core_s = (eps_s > 0.5 * (float(eps_s.min()) + float(eps_s.max()))).to(torch.float64)
ring_core = core_s.clone()
ring_core[:, :o1y + int((W_WG / 2 + GAP / 2) * RES)] = 0.0   # cut the bus
X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
w = ring_core.numpy()
cx = float((X * w).sum() / w.sum())
cy = float((Y * w).sum() / w.sum())


def circ_power(E):
    return float((E.abs() ** 2 * ring_core).sum())


def pump_solve(f):
    return solve_fdfd_tm_real_backend(eps_p, dx, f, jz_p, NPML,
                                      backend=BE, device=DEV)


# --- resonance hunt: coarse scan over ~1 FSR, then two refinements ---
f0 = 1 / LAM_P
n_g = 2.2
FSR = 1.0 / (n_g * 2 * math.pi * R_RING)
t0 = time.time()
hist = []


def scan(fs):
    out = []
    for f in fs:
        P = circ_power(pump_solve(f))
        hist.append((f, P))
        out.append(P)
    return np.array(out)


fs1 = np.linspace(f0 - 0.6 * FSR, f0 + 0.6 * FSR, 9)
P1 = scan(fs1)
fc = fs1[int(np.argmax(P1))]
fs2 = np.linspace(fc - 0.15 * FSR, fc + 0.15 * FSR, 7)
P2 = scan(fs2)
fc = fs2[int(np.argmax(P2))]
fs3 = np.linspace(fc - 0.03 * FSR, fc + 0.03 * FSR, 7)
P3 = scan(fs3)
f_res = fs3[int(np.argmax(P3))]
print(f"fdfd_shg_ring  pump resonance {1e3/f_res:.2f} nm "
      f"({len(hist)} solves, {time.time()-t0:.0f}s)")

Ep = pump_solve(f_res)
Ep_off = pump_solve(f_res + 0.5 * FSR)
enh = circ_power(Ep) / circ_power(Ep_off)
print(f"fdfd_shg_ring  g1 circulating power on/off resonance: {enh:.1f}x")

# --- dk self-calibration on a straight guide (scipy, small grid) ---
sg_nx, sg_ny = int(30 * RES), int(6 * RES)
e_sp = torch.full((sg_nx, sg_ny), N_PC ** 2, dtype=torch.float64)
e_ss = torch.full((sg_nx, sg_ny), N_SC ** 2, dtype=torch.float64)
jw = int(W_WG / 2 * RES)
e_sp[:, sg_ny // 2 - jw: sg_ny // 2 + jw] = N_P ** 2
e_ss[:, sg_ny // 2 - jw: sg_ny // 2 + jw] = N_S ** 2
sp_port = Port(axis="x", index=NPML + 8, lo=sg_ny // 2 - SPAN,
               hi=sg_ny // 2 + SPAN, direction=+1)
Esg = solve_fdfd_tm(e_sp, dx, f_res, make_mode_source(e_sp, sp_port, f_res, dx), NPML)
core1 = (torch.arange(sg_ny) >= sg_ny // 2 - jw) & (torch.arange(sg_ny) < sg_ny // 2 + jw)
jz_sh_sg = -1j * (2 * math.pi * 2 * f_res) * CHI2 * core1[None, :].to(
    torch.complex128) * Esg ** 2
Esh_sg = solve_fdfd_tm(e_ss, dx, 2 * f_res, jz_sh_sg, NPML)
prof_s, _ = port_mode_line(e_ss, sp_port, 2 * f_res, dx)
a_x = (Esh_sg[:, sp_port.lo:sp_port.hi]
       * prof_s.to(torch.complex128)[None, :]).sum(dim=1).abs() ** 2
seg = a_x[NPML + int(1.5 * RES): sg_nx - NPML - RES].numpy()
pk = [k for k in range(1, len(seg) - 1)
      if seg[k] > seg[k - 1] and seg[k] >= seg[k + 1] and seg[k] > 0.1 * seg.max()]
period = float(np.mean(np.diff(pk)) * dx)
dk_grid = 2 * math.pi / period
print(f"fdfd_shg_ring  grid beat {period:.3f} um -> Lambda_QPM {period:.3f} um "
      f"({len(pk)} peaks)")

# --- SH solves on the ring: QPM-poled vs unpoled ---
theta = np.arctan2(Y - cy, X - cx)
arc = theta * R_RING
poling = torch.from_numpy(np.sign(np.cos(dk_grid * arc)))
w_sh = 2 * math.pi * (2 * f_res)


def sh_solve(sign_mask):
    chi2 = CHI2 * ring_core * sign_mask
    jz = -1j * w_sh * chi2.to(torch.complex128) * Ep ** 2
    return solve_fdfd_tm_real_backend(eps_s, dx, 2 * f_res, jz, NPML,
                                      backend=BE, device=DEV)


Es_pol = sh_solve(poling)
Es_unp = sh_solve(torch.ones_like(poling))
prof_out, _ = port_mode_line(eps_s, p_out, 2 * f_res, dx)
def sh_out(E):
    return float(((E[p_out.index, p_out.lo:p_out.hi]
                   * prof_out.to(torch.complex128)).sum() * dx).abs() ** 2)
g2 = sh_out(Es_pol) / max(sh_out(Es_unp), 1e-300)
print(f"fdfd_shg_ring  g2 SH out (bus): poled/unpoled = {g2:.1f}x "
      f"(poled {sh_out(Es_pol):.3e})")

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
a1, a2, a3, a4 = axs.flat
h = np.array(sorted(hist))
a1.plot((1 / h[:, 0]) * 1e3, h[:, 1] / h[:, 1].max(), "o-")
a1.axvline(1e3 / f_res, color="r", lw=0.8)
a1.set_xlabel("pump wavelength [nm]"); a1.set_ylabel("circ. power [norm]")
a1.set_title(f"resonance hunt: on/off = {enh:.0f}x")
a2.imshow(Ep.abs().numpy().T, origin="lower", cmap="inferno",
          extent=[0, nx * dx, 0, ny * dx])
a2.set_title(f"pump |Ez| at {1e3/f_res:.1f} nm (cuDSS)")
a3.imshow(Es_pol.abs().numpy().T, origin="lower", cmap="inferno",
          extent=[0, nx * dx, 0, ny * dx])
a3.set_title(f"SH |Ez| QPM-poled (Lambda={period:.2f} um)")
a4.imshow((poling * ring_core).numpy().T, origin="lower", cmap="RdBu",
          extent=[0, nx * dx, 0, ny * dx])
a4.set_title(f"poling sign (poled/unpoled SH out: {g2:.0f}x)")
for a in axs.flat[1:]:
    a.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "fdfd_shg_ring.png", dpi=150)
print("fdfd_shg_ring.png written")
