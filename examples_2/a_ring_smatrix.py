# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (a): SiN ring resonator, HALF-RING coupler extraction.

A full r = 50 um weakly-guided ring needs >= 10 round trips (~6000 tu on
a 3M-cell grid) for resonances to form in FDTD. Instead (user directive)
only the LOWER HALF of the gdsfactory ring_single is simulated: the ring
side-guides run vertically at the cut plane and exit cleanly into the
top PML. Two short runs extract the full coupler physics:

  run 1 (bus-driven):  t(f) = bus through amplitude, kappa(f) = amplitude
                       coupled into the ring (right arc port),
  run 2 (arc-driven):  A(f) = arc-end to arc-end amplitude = t * a(f),
                       a(f) = BARE half-loop propagation (bend loss+phase).

The ring spectrum is then assembled analytically at arbitrary resolution
(all-pass: S21 = (t - a^2)/(1 - t a^2), a^2 = bare full loop = (A/t)^2),
giving exact FSR and Q without ever waiting for FDTD ring buildup.

Platform: SiN 150 nm / SiO2, meep materials-library indices, lateral
effective index; r = 50 um, W = 4 um, gap = 0.6 um.

Run: uv run python examples_2/a_ring_smatrix.py
"""
import math
import pathlib
import sys

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(16)

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

N_SIN = Si3N4.n(1 / 1.55).real
N_CLAD = SiO2.n(1 / 1.55).real
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / 1.55, "te", k=1).neff[0])
R_RING, W_WG, GAP = 50.0, 4.0, 0.6
RES = 15
FCEN = 1 / 1.55
dx = 1.0 / RES
print(f"a_ring  half-ring method: n_core={NEFF:.4f} n_clad={N_CLAD:.4f} "
      f"r={R_RING} W={W_WG} gap={GAP}")

xs = gf.get_cross_section("strip").copy(width=W_WG)
c = gf.components.ring_single(radius=R_RING, gap=GAP, cross_section=xs)
grid_f, eps_f, ports = component_to_eps(c, res=RES, pad=3.0,
                                        n_core=NEFF, n_clad=N_CLAD)
o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]

# --- crop at the ring-center height (side guides vertical there) and
# rebuild a half grid with clad + top PML above the cut ---
NPML = RES
j_center = o1y + int((W_WG / 2 + GAP + W_WG / 2 + R_RING) * RES)
PAD_TOP = int(2.0 * RES) + NPML
ny_h = j_center + PAD_TOP
nx_h = grid_f.nx
grid = Grid(nx=nx_h, ny=ny_h, dx=dx)
eps = torch.full((nx_h, ny_h), N_CLAD ** 2, dtype=grid.dtype)
eps[:, :j_center] = eps_f[:, :j_center]
eps[:, j_center:] = eps[:, j_center - 1: j_center]   # extend guides straight up
print(f"a_ring  half grid {nx_h}x{ny_h} ({nx_h * dx:.0f} x {ny_h * dx:.0f} um)")

coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
SPAN = int(3.5 * RES)
p_in = Port(axis="x", index=max(o1x + 4, NPML + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
p_thru = Port(axis="x", index=min(o2x - 4, nx_h - NPML - 4), lo=o2y - SPAN,
              hi=o2y + SPAN, direction=-1)
# arc ports: vertical guides at x = x_center +/- (R + W/2), y-normal planes
x_cen = nx_h // 2
ix_arm = int(R_RING * RES)
j_arc = ny_h - NPML - int(2.0 * RES)                 # 2 um below the top edge
p_arcL = Port(axis="y", index=j_arc, lo=x_cen - ix_arm - SPAN,
              hi=x_cen - ix_arm + SPAN, direction=-1)
p_arcR = Port(axis="y", index=j_arc, lo=x_cen + ix_arm - SPAN,
              hi=x_cen + ix_arm + SPAN, direction=-1)
validate_ports([p_in, p_thru, p_arcL, p_arcR], grid, NPML)

NFREQ = 60
DF_BAND = 0.006
freqs = np.linspace(FCEN - DF_BAND / 2, FCEN + DF_BAND / 2, NFREQ).tolist()
T_RUN = 500.0                                       # single-pass: no buildup
PULSE = GaussianPulse(FCEN, 0.04)


def eline(p):
    if p.axis == "x":
        return eps[p.index, p.lo:p.hi]
    return eps[p.lo:p.hi, p.index]


def run_ports(src_port, mons_ports):
    src = ModeSourceTM(grid, coeffs, src_port, PULSE, FCEN)
    mons = [FluxMonitorTM(grid, freqs, x_normal=(p.axis == "x"), idx=p.index,
                          lo=p.lo, hi=p.hi) for p in mons_ports]
    with torch.no_grad():
        _, accs = run_tm(init_tm_state(grid), coeffs,
                         int(round(T_RUN / grid.dt)), source=src,
                         monitors=mons, checkpoint_every=400)
    out = {}
    for p, acc, name in zip(mons_ports, accs, range(len(mons_ports))):
        sign = -1.0 if p.axis == "x" else +1.0
        out[name] = decompose(acc, eline(p), dx, freqs, "tm", 1,
                              axis_sign=sign)
    return out

# run 1: bus-driven
r1 = run_ports(p_in, [p_in, p_thru, p_arcR])
a_in = r1[0][0][:, 0]
t = (r1[1][0][:, 0] / a_in).numpy()                  # through (+x out)
kap = (r1[2][0][:, 0] / a_in).numpy()                # into ring, +y out at R

# run 2: arc-driven (down the right side, through coupler, up the left)
r2 = run_ports(p_arcR, [p_arcR, p_arcL])
a_inR = r2[0][1][:, 0]                               # -y incident (a_minus)
A = (r2[1][0][:, 0] / a_inR).numpy()                 # +y outgoing at L
a_half = A / t                                       # bare half loop
loop = a_half ** 2                                   # bare full loop a^2

print(f"a_ring  at 1550: |t|^2={abs(t[NFREQ//2])**2:.3f} "
      f"|kappa|^2={abs(kap[NFREQ//2])**2:.3f} "
      f"|a_loop|={abs(loop[NFREQ//2]):.4f} "
      f"(|t|^2+|k|^2={abs(t[NFREQ//2])**2+abs(kap[NFREQ//2])**2:.3f})")

# --- analytic assembly on a dense grid ---
fd = np.linspace(freqs[0], freqs[-1], 6000)
ti = np.interp(fd, freqs, t.real) + 1j * np.interp(fd, freqs, t.imag)
# interpolate loop magnitude and UNWRAPPED phase (phase winds ~2pi/FSR)
lm = np.interp(fd, freqs, np.abs(loop))
lp = np.interp(fd, freqs, np.unwrap(np.angle(loop)))
loop_d = lm * np.exp(1j * lp)
S21 = (ti - loop_d) / (1 - ti * loop_d)
T = np.abs(S21) ** 2
lam_d = 1e3 / fd

i0 = int(np.argmin(T))
half = (T[i0] + np.median(T)) / 2
i_lo = i0
while i_lo > 0 and T[i_lo] < half:
    i_lo -= 1
i_hi = i0
while i_hi < len(fd) - 1 and T[i_hi] < half:
    i_hi += 1
Q = fd[i0] / max(fd[i_hi] - fd[i_lo], 1e-12)
dips = [i for i in range(2, len(fd) - 2)
        if T[i] < 0.6 * np.median(T) and T[i] <= T[i - 1] and T[i] < T[i + 1]]
fsr_nm = (abs(1e3 / fd[dips[0]] - 1e3 / fd[dips[-1]]) / (len(dips) - 1)
          if len(dips) >= 2 else float("nan"))
print(f"a_ring  deepest dip lambda={lam_d[i0]:.2f} nm |S21|^2={T[i0]:.3f} "
      f"Q={Q:.0f} FSR={fsr_nm:.2f} nm extinction "
      f"{10 * math.log10(max(np.median(T), 1e-12) / max(T[i0], 1e-12)):.1f} dB")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax1.plot(lam_d, T, "b-", lw=0.9)
ax1.axvline(lam_d[i0], color="gray", ls=":", lw=0.8)
ax1.set_xlabel("wavelength [nm]"); ax1.set_ylabel(r"$|S_{21}|^2$")
ax1.set_title(f"r=50 um SiN ring (half-ring method): "
              f"Q={Q:.0f}, FSR={fsr_nm:.2f} nm")
ax1.set_ylim(0, 1.05)
step = max(1, nx_h // 900)
ax2.imshow(eps.numpy()[::step, ::step].T, origin="lower", cmap="viridis",
           extent=[0, nx_h * dx, 0, ny_h * dx])
ax2.set_title("simulated half domain (gf ring_single, cropped at midline)")
ax2.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "a_ring_smatrix.png", dpi=150)
plt.close(fig)
np.savez(OUT / "a_ring_coupler_data.npz", freqs=freqs, t=t, kappa=kap,
         A=A, loop=loop)
print("a_ring_smatrix.png written (+ a_ring_coupler_data.npz)")
