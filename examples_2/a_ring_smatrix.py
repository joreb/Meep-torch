# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Example (a): SiN ring resonator, HALF-RING coupler extraction.

At 674 nm (red diode, USER spec) with W = 1.0 um: r = 50 um bends are
lossless and the ring is effectively single-mode. W = 4.0 um was tried
and is PHYSICS-VETOED at 674 nm: supermode beat length 1-3 mm (no
practical gap/coupler couples), and the wide bend scrambles TE0 into
higher lateral modes (measured 1.8 dB/rt apparent TE0 loss on lossless
bends). A full ring still needs many round trips to resonate, so the
half-ring coupler extraction stays: Instead (user directive)
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

LAM = 0.674                                         # red diode (USER spec)
N_SIN = Si3N4.n(1 / LAM).real
N_CLAD = SiO2.n(1 / LAM).real
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / LAM, "te", k=1).neff[0])
R_RING, W_WG, GAP = 50.0, 1.0, 0.2
LC = 4.0                                            # racetrack straight coupler
RES = 50                                            # 20 cells per in-medium wavelength (674 nm)
FCEN = 1 / LAM
dx = 1.0 / RES
print(f"a_ring  half-ring method: n_core={NEFF:.4f} n_clad={N_CLAD:.4f} "
      f"r={R_RING} W={W_WG} gap={GAP}")

xs = gf.get_cross_section("strip").copy(width=W_WG)
c = gf.components.ring_single(radius=R_RING, gap=GAP, length_x=LC,
                              cross_section=xs)   # racetrack: straight coupler section
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
DEV = "cuda" if torch.cuda.is_available() else "cpu"
grid = Grid(nx=nx_h, ny=ny_h, dx=dx, device=DEV)
eps = torch.full((nx_h, ny_h), N_CLAD ** 2, dtype=grid.dtype)
eps[:, :j_center] = eps_f[:, :j_center]
eps[:, j_center:] = eps[:, j_center - 1: j_center]   # extend guides straight up
eps = eps.to(DEV)
print(f"a_ring  half grid {nx_h}x{ny_h} ({nx_h * dx:.0f} x {ny_h * dx:.0f} um) on {DEV}")

coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
SPAN = int(1.5 * RES)   # 0.5 um half-width + 1 um tails
p_in = Port(axis="x", index=max(o1x + 4, NPML + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
p_thru = Port(axis="x", index=min(o2x - 4, nx_h - NPML - 4), lo=o2y - SPAN,
              hi=o2y + SPAN, direction=-1)
# arc ports: locate the two vertical side guides from the eps row itself
# (the ring center is NOT exactly the domain center -- pads/bus offsets)
j_arc = ny_h - NPML - int(2.0 * RES)                 # 2 um below the top edge
row = eps[:, j_arc]
core_cols = (row > 0.5 * (float(row.min()) + float(row.max()))).nonzero().flatten()
splits = (core_cols.diff() > 1).nonzero().flatten()
assert len(splits) == 1, f"expected 2 side guides, got {len(splits) + 1} clusters"
cl = core_cols[: splits[0] + 1]
cr = core_cols[splits[0] + 1:]
xL = int(cl.float().mean().round())
xR = int(cr.float().mean().round())
print(f"a_ring  arc guide centers x = {xL * dx:.1f}, {xR * dx:.1f} um")
p_arcL = Port(axis="y", index=j_arc, lo=xL - SPAN, hi=xL + SPAN, direction=-1)
p_arcR = Port(axis="y", index=j_arc, lo=xR - SPAN, hi=xR + SPAN, direction=-1)
validate_ports([p_in, p_thru, p_arcL, p_arcR], grid, NPML)

NFREQ = 60
DF_BAND = 0.008                                     # ~3.6 nm, ~4 FSR
freqs = np.linspace(FCEN - DF_BAND / 2, FCEN + DF_BAND / 2, NFREQ).tolist()
# T_RUN per geometry (silent-truncation trap): gaussian tail 2*(5/fwidth)
# + arc transit pi*R*n + straight/coupler margin
T_RUN = 2 * (5 / 0.05) + math.pi * R_RING * 1.7 + 80.0
PULSE = GaussianPulse(FCEN, 0.05)


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
        out[name] = decompose(acc.cpu(), eline(p).cpu(), dx, freqs, "tm", 1,
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
# magnitude-only normalization: dividing by complex t would strip the
# BUS propagation phase (a path A never traveled) -- that error tripled
# the apparent FSR. A's own phase IS the half-loop propagation phase.
a_half = A / np.abs(t)                               # bare half loop
loop = a_half ** 2                                   # bare full loop a^2 (phase = full loop)
# passive ring: no round-trip gain. Small excess = A-noise / |t|~1; large = broken.
ex = float(np.abs(loop).max())
assert ex < 1.05, f"|a_loop| max {ex:.4f} -- extraction broken, not noise"
loop = np.where(np.abs(loop) > 1.0, loop / np.abs(loop), loop)

unit = abs(t[NFREQ//2])**2 + abs(kap[NFREQ//2])**2
print(f"a_ring  at band center ({1e3/freqs[NFREQ//2]:.1f} nm): |t|^2={abs(t[NFREQ//2])**2:.3f} "
      f"|kappa|^2={abs(kap[NFREQ//2])**2:.4f} "
      f"|a_loop|={abs(loop[NFREQ//2]):.4f} (|t|^2+|k|^2={unit:.3f})")
if unit < 0.8:
    print("a_ring  WARNING: |t|^2+|kappa|^2 < 0.8 -- extraction INVALID "
          "(T_RUN truncation or port mis-capture); Q/FSR below untrustworthy")

# --- analytic assembly on a dense grid ---
fd = np.linspace(freqs[0], freqs[-1], 6000)
# |t| only: the measured t phase is dominated by bus propagation between
# the ports (an EXTERNAL phase); putting it in the resonance denominator
# produced |S21| > 1 lobes. The ring round-trip phase lives in `loop`.
ti = np.interp(fd, freqs, np.abs(t))
# interpolate loop magnitude and UNWRAPPED phase (phase winds ~2pi/FSR)
lm = np.interp(fd, freqs, np.abs(loop))
lp = np.interp(fd, freqs, np.unwrap(np.angle(loop)))
loop_d = lm * np.exp(1j * lp)
S21 = (ti - loop_d) / (1 - ti * loop_d)
T = np.abs(S21) ** 2
assert T.max() <= 1.0 + 1e-6, f"|S21|^2 max {T.max():.4f} > 1 -- assembly bug"
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
ax2.imshow(eps.cpu().numpy()[::step, ::step].T, origin="lower", cmap="viridis",
           extent=[0, nx_h * dx, 0, ny_h * dx])
ax2.set_title("simulated half domain (gf ring_single, cropped at midline)")
ax2.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "a_ring_smatrix.png", dpi=150)
plt.close(fig)
np.savez(OUT / "a_ring_coupler_data.npz", freqs=freqs, t=t, kappa=kap,
         A=A, loop=loop)
print("a_ring_smatrix.png written (+ a_ring_coupler_data.npz)")
