# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Kerr (chi3) ring measurement via SELF-CONSISTENT FDFD (cuDSS GPU).

Time-domain measured Kerr by free ringdown (charge, cut, watch f drift).
FDFD replaces all of it with a fixed point: solve linear, update
eps_eff = eps + chi3*|E|^2 (core only), re-solve until converged --
the steady state of the Kerr-shifted cavity at that drive level. The
resonance is then re-hunted at several drive amplitudes:

  g_lin:  the resonance shift df must be LINEAR in circulating energy
          (Kerr signature; fit R^2 gate)
  g_cold: extrapolating the df-vs-U line to U=0 must recover the cold
          resonance found at negligible amplitude (same self-check the
          time-domain plan used, no waiting for decay)

Same r=15 demo ring / pump band as fdfd_shg_ring.py.
Run: uv run python examples_2/nonlinear/fdfd_kerr_ring.py
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

from meep_torch.fdfd import make_mode_source, solve_fdfd_tm_real_backend
from meep_torch.ports.port import Port

gf.gpdk.PDK.activate()

LAM_P, N_P, N_PC = 1.348, 2.0233, 1.7065
R_RING, W_WG, GAP = 15.0, 1.2, 0.25
RES = 40
dx = 1.0 / RES
NPML = RES
CHI3 = 5e-4          # arb units; gates are about LINEARITY, not magnitude
BE, DEV = "cudss", "cuda"

xs = gf.get_cross_section("strip").copy(width=W_WG)
comp = gf.components.ring_single(radius=R_RING, gap=GAP, length_x=0.01,
                                 bend=gf.components.bend_circular,
                                 cross_section=xs)
grid, eps0, ports = component_to_eps(comp, res=RES, pad=2.0,
                                     n_core=N_P, n_clad=N_PC)
nx, ny = eps0.shape
o1x, o1y, _ = ports["o1"]
SPAN = int(1.2 * RES)
p_in = Port(axis="x", index=max(o1x + 4, NPML + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
core = (eps0 > 0.5 * (float(eps0.min()) + float(eps0.max()))).to(torch.float64)
ring_core = core.clone()
ring_core[:, :o1y + int((W_WG / 2 + GAP / 2) * RES)] = 0.0
jz0 = make_mode_source(eps0, p_in, 1 / LAM_P, dx)
print(f"fdfd_kerr_ring  grid {nx}x{ny}, chi3={CHI3}, backend {BE}/{DEV}")


def kerr_solve(f, amp, n_fp=4):
    """Fixed-point Kerr solve at drive amplitude `amp`. Returns (E, U)."""
    eps = eps0.clone()
    E = None
    for it in range(n_fp):
        E = solve_fdfd_tm_real_backend(eps, dx, f, amp * jz0, NPML,
                                       backend=BE, device=DEV)
        eps_new = eps0 + CHI3 * core * (E.abs() ** 2)
        # damped mixing keeps the fixed point stable near resonance
        eps = 0.5 * eps + 0.5 * eps_new
    U = float((E.abs() ** 2 * ring_core).sum())
    return E, U


def hunt(amp, f_guess, spans=(0.10, 0.02), npts=(7, 7)):
    """Resonance of the SELF-CONSISTENT cavity at this drive level."""
    n_g = 2.2
    FSR = 1.0 / (n_g * 2 * math.pi * R_RING)
    fc = f_guess
    for s, npt in zip(spans, npts):
        fs = np.linspace(fc - s * FSR, fc + s * FSR, npt)
        Ps = [kerr_solve(f, amp)[1] for f in fs]
        fc = fs[int(np.argmax(Ps))]
    _, U = kerr_solve(fc, amp)
    return fc, U


t0 = time.time()
# cold resonance: tiny amplitude (Kerr term negligible), wider first span
f_cold, _ = hunt(1e-6, 1 / LAM_P, spans=(0.6, 0.10, 0.02), npts=(9, 7, 7))
print(f"fdfd_kerr_ring  cold resonance {1e3/f_cold:.2f} nm "
      f"({time.time()-t0:.0f}s)")

AMPS = [1.0, 1.5, 2.0, 3.0]
rows = []
for a in AMPS:
    f_a, U_a = hunt(a, f_cold)
    rows.append((a, f_a, U_a, f_a - f_cold))
    print(f"fdfd_kerr_ring  amp {a:.1f}: f_res {1e3/f_a:.3f} nm, "
          f"U {U_a:.3e}, df {f_a-f_cold:+.3e}")

U = np.array([r[2] for r in rows])
df = np.array([r[3] for r in rows])
# g_lin: df linear in U
coef = np.polyfit(U, df, 1)
pred = np.polyval(coef, U)
ss = 1 - np.sum((df - pred) ** 2) / max(np.sum((df - df.mean()) ** 2), 1e-300)
# g_cold: U=0 intercept back at the cold line, in units of the max shift
icpt = coef[1] / max(abs(df).max(), 1e-300)
print(f"fdfd_kerr_ring  g_lin: df vs U linear fit R^2 = {ss:.4f} "
      f"(slope {coef[0]:.3e})")
print(f"fdfd_kerr_ring  g_cold: U=0 intercept = {icpt*100:.1f}% of max shift")
assert ss > 0.98, f"Kerr shift not linear in U: R^2={ss}"
assert abs(icpt) < 0.15, f"cold-frequency intercept off: {icpt}"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
a1.plot(U, df * 1e6, "o-")
a1.plot(U, pred * 1e6, "r--", lw=0.8, label=f"linear fit R$^2$={ss:.3f}")
a1.set_xlabel("circulating energy U [arb]")
a1.set_ylabel("resonance shift df [1e-6 f-units]")
a1.set_title("Kerr: df linear in U (self-consistent FDFD)")
a1.legend()
a2.plot([r[0] for r in rows], df * 1e6, "s-")
a2.set_xlabel("drive amplitude")
a2.set_title(f"cold {1e3/f_cold:.2f} nm; intercept {icpt*100:.1f}% of max shift")
fig.tight_layout()
fig.savefig(OUT / "fdfd_kerr_ring.png", dpi=150)
print("fdfd_kerr_ring.png written")
