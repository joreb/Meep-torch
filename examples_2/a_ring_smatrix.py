# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (a): SiN ring resonator on the REAL platform -- gdsfactory
geometry, FDTD, mode-overlap S-matrix, detuning response.

Platform: SiN 150 nm on SiO2 (user spec): lateral effective index
n_core = neff(150 nm slab TE0), indices from the meep materials
library (Si3N4 Luke, SiO2 Horiba) at 1550 nm. Ring r = 50 um,
waveguide width 4.0 um, bus gap 0.8 um -- weakly guided, so fields
extend far and the ring is large (FSR ~ 5 nm). RES 15 (16 cells per
in-medium wavelength, adequate at this contrast).

|S21|^2 / |S11|^2: TE0 mode source + mode-overlap decomposition at the
two bus ports, single broadband run. Transient: CW mode drive at three
detunings around the deepest interior resonance.

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

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM, PointProbe
from meep_torch.core.run import run_tm
from meep_torch.core.sources import ContinuousSource, GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

from meep_torch.core.materials import Si3N4, SiO2

# refractive indices from the meep materials library (Lorentz fits):
# Si3N4 K. Luke et al. (2015), SiO2 Horiba TN08 -- evaluated at 1550 nm
N_SIN = Si3N4.n(1 / 1.55).real                      # 1.9963
N_CLAD = SiO2.n(1 / 1.55).real                      # 1.4577
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, 1 / 1.55, "te", k=1).neff[0])
R_RING, W_WG, GAP = 50.0, 4.0, 0.8
print(f"a_ring  SiN 150nm/SiO2: n_core = {NEFF:.4f}, n_clad = {N_CLAD}; "
      f"r = {R_RING}, W = {W_WG}, gap = {GAP}")

RES = 15
FCEN = 1 / 1.55
dx = 1.0 / RES
xs = gf.get_cross_section("strip").copy(width=W_WG)
c = gf.components.ring_single(radius=R_RING, gap=GAP, cross_section=xs)
grid, eps, ports = component_to_eps(c, res=RES, pad=3.0,
                                    n_core=NEFF, n_clad=N_CLAD)
nx, ny = grid.nx, grid.ny
NPML = RES
print(f"a_ring  grid {nx}x{ny} ({nx * dx:.0f} x {ny * dx:.0f} um)")

o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
SPAN = int(3.5 * RES)
p_in = Port(axis="x", index=max(o1x + 4, NPML + 4), lo=o1y - SPAN,
            hi=o1y + SPAN, direction=+1)
p_out = Port(axis="x", index=min(o2x - 4, nx - NPML - 4), lo=o2y - SPAN,
             hi=o2y + SPAN, direction=-1)
validate_ports([p_in, p_out], grid, NPML)
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)

NFREQ = 160
DF_BAND = 0.0125                                    # ~30 nm, ~6 FSR
freqs = np.linspace(FCEN - DF_BAND / 2, FCEN + DF_BAND / 2, NFREQ).tolist()
FW = 0.04                                           # pulse: t_end = 250
T_RUN = 650.0
mons = [FluxMonitorTM(grid, freqs, x_normal=True, idx=p.index, lo=p.lo, hi=p.hi)
        for p in (p_in, p_out)]
src = ModeSourceTM(grid, coeffs, p_in, GaussianPulse(FCEN, FW), FCEN)
with torch.no_grad():
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=400)
e1 = eps[p_in.index, p_in.lo:p_in.hi]
e2 = eps[p_out.index, p_out.lo:p_out.hi]
a1p, a1m = decompose(accs[0], e1, dx, freqs, "tm", 1, axis_sign=-1.0)
a2p, a2m = decompose(accs[1], e2, dx, freqs, "tm", 1, axis_sign=-1.0)
s21 = (a2p[:, 0] / a1p[:, 0]).numpy()
s11 = (a1m[:, 0] / a1p[:, 0]).numpy()
lam = 1.0 / np.array(freqs)
T = np.abs(s21) ** 2
R = np.abs(s11) ** 2

EDGE = NFREQ // 10
i0 = EDGE + int(np.argmin(T[EDGE:NFREQ - EDGE]))
f0 = freqs[i0]
half = (T[i0] + np.median(T)) / 2
i_lo = i0
while i_lo > 0 and T[i_lo] < half:
    i_lo -= 1
i_hi = i0
while i_hi < NFREQ - 1 and T[i_hi] < half:
    i_hi += 1
Q = f0 / max(freqs[i_hi] - freqs[i_lo], 1e-9)
dips = [i for i in range(EDGE, NFREQ - EDGE)
        if T[i] < 0.8 * np.median(T) and T[i] <= T[i - 1] and T[i] < T[i + 1]]
fsr_nm = None
if len(dips) >= 2:
    fsr_nm = abs(1 / freqs[dips[0]] - 1 / freqs[dips[-1]]) / (len(dips) - 1) * 1e3
print(f"a_ring  deepest resonance lambda={1e3 / f0:.2f} nm |S21|^2={T[i0]:.3f} "
      f"Q~{Q:.0f}" + (f" FSR~{fsr_nm:.2f} nm" if fsr_nm else " (single dip)"))


def transient(detune_kappa):
    kappa = f0 / Q / 2
    f_d = f0 + detune_kappa * kappa
    src_cw = ModeSourceTM(grid, coeffs, p_in, ContinuousSource(f_d, width=20.0),
                          f_d)
    probe = PointProbe(nx // 2, ny - NPML - int(3.0 * RES))  # ring top
    n = int(round(400.0 / grid.dt))
    with torch.no_grad():
        _, (series,) = run_tm(init_tm_state(grid), coeffs, n, source=src_cw,
                              monitors=[probe], checkpoint_every=400)
    s = series.numpy() ** 2
    w = max(1, int(round(1.0 / f_d / grid.dt)))
    env = np.convolve(s, np.ones(w) / w, mode="valid")
    return np.arange(len(env)) * grid.dt, env


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
ax1.plot(lam * 1e3, T, "b-", lw=1.1, label=r"$|S_{21}|^2$")
ax1.plot(lam * 1e3, R, "r-", lw=0.8, alpha=0.6, label=r"$|S_{11}|^2$")
ax1.axvline(1e3 / f0, color="gray", ls=":", lw=0.8)
ax1.set_xlabel("wavelength [nm]"); ax1.set_ylabel("power")
ax1.set_title(f"r=50 um SiN ring: Q~{Q:.0f}"
              + (f", FSR {fsr_nm:.1f} nm" if fsr_nm else ""))
ax1.legend(fontsize=8); ax1.set_ylim(0, 1.1)

step = max(1, nx // 800)
ax2.imshow(eps.numpy()[::step, ::step].T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx])
ax2.set_title(f"ring_single r={R_RING:.0f}, W={W_WG}, gap={GAP} um")
ax2.set_xlabel("x [um]")

env0_max = None
for dk, col in ((0.0, "tab:red"), (1.0, "tab:orange"), (2.0, "tab:blue")):
    t, env = transient(dk)
    if env0_max is None:
        env0_max = env.max()
    ax3.plot(t, env / env0_max, color=col, lw=1.0,
             label=rf"$\delta = {dk:.0f}\,\kappa$")
ax3.set_xlabel("time [um/c]"); ax3.set_ylabel("ring intensity (norm.)")
ax3.set_title("ring buildup vs detuning (CW mode drive)")
ax3.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "a_ring_smatrix.png", dpi=150)
plt.close(fig)
print("a_ring_smatrix.png written")
