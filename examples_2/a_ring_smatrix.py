# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (a): SiN ring resonator -- gdsfactory geometry through the
meep-torch interface, FDTD, S-matrix spectrum, detuning response.

Redone at the fdtdx quality bar: RES 30, TE0 mode source on the bus and
mode-overlap decomposition at both bus ports (ports machinery), so
|S21|^2 / |S11|^2 are true mode-to-mode S-parameters from a SINGLE
broadband run (self-normalizing, no reference sweep). Transient panel:
CW mode drive at three detunings around the deepest resonance.

Effective-index 2D, n_core = 2.0 nominal (see gds_meep_torch notes).

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
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

NEFF, N_CLAD = 2.0, 1.44
RES = 30
FCEN, DF = 1 / 1.55, 0.05
c = gf.components.ring_single(radius=5.0, gap=0.2)
grid, eps, ports = component_to_eps(c, res=RES, pad=2.5,
                                    n_core=NEFF, n_clad=N_CLAD)
NPML = RES
print(f"a_ring  grid {grid.nx}x{grid.ny} at RES={RES}, n_core={NEFF}")
o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
SPAN = int(0.7 * RES)
p_in = Port(axis="x", index=max(o1x, NPML + 4), lo=o1y - SPAN, hi=o1y + SPAN,
            direction=+1)
p_out = Port(axis="x", index=min(o2x, grid.nx - NPML - 4),
             lo=o2y - SPAN, hi=o2y + SPAN, direction=-1)
validate_ports([p_in, p_out], grid, NPML)
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)

NFREQ = 120
freqs = np.linspace(FCEN - DF / 2, FCEN + DF / 2, NFREQ).tolist()
T_RUN = 400.0                                     # pulse t_end=100 + ring decay
mons = [FluxMonitorTM(grid, freqs, x_normal=True, idx=p.index, lo=p.lo, hi=p.hi)
        for p in (p_in, p_out)]
src = ModeSourceTM(grid, coeffs, p_in, GaussianPulse(FCEN, 2 * DF), FCEN)
with torch.no_grad():
    # checkpoint_every bounds the per-chunk monitor-sample buffer -- without
    # it _run_chunk holds every step's 120-freq samples (tens of GB, OOM)
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=400)
e1 = eps[p_in.index, p_in.lo:p_in.hi]
e2 = eps[p_out.index, p_out.lo:p_out.hi]
a1p, a1m = decompose(accs[0], e1, grid.dx, freqs, "tm", 1, axis_sign=-1.0)
a2p, a2m = decompose(accs[1], e2, grid.dx, freqs, "tm", 1, axis_sign=-1.0)
s21 = (a2p[:, 0] / a1p[:, 0]).numpy()
s11 = (a1m[:, 0] / a1p[:, 0]).numpy()
lam = 1.0 / np.array(freqs)
T = np.abs(s21) ** 2
R = np.abs(s11) ** 2

# deepest resonance + local-FWHM Q + FSR (interior only: band edges carry
# pulse-spectrum noise)
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
fwhm = freqs[i_hi] - freqs[i_lo]
Q = f0 / max(fwhm, 1e-9)
dips = [i for i in range(EDGE, NFREQ - EDGE)
        if T[i] < 0.8 * np.median(T) and T[i] <= T[i - 1] and T[i] < T[i + 1]]
fsr_nm = None
if len(dips) >= 2:
    fsr_nm = abs(1 / freqs[dips[0]] - 1 / freqs[dips[-1]]) / (len(dips) - 1) * 1e3
print(f"a_ring  deepest resonance lambda={1e3 / f0:.1f} nm |S21|^2={T[i0]:.3f} "
      f"Q~{Q:.0f}" + (f" FSR~{fsr_nm:.1f} nm" if fsr_nm else ""))


def transient(detune_kappa):
    kappa = f0 / Q / 2
    f_d = f0 + detune_kappa * kappa
    src_cw = ModeSourceTM(grid, coeffs, p_in, ContinuousSource(f_d, width=10.0),
                          f_d)
    probe = PointProbe(grid.nx // 2, o1y + int(5.0 * RES))
    n = int(round(200.0 / grid.dt))
    with torch.no_grad():
        _, (series,) = run_tm(init_tm_state(grid), coeffs, n, source=src_cw,
                              monitors=[probe], checkpoint_every=400)
    s = series.numpy() ** 2
    w = max(1, int(round(1.0 / f_d / grid.dt)))
    env = np.convolve(s, np.ones(w) / w, mode="valid")
    return np.arange(len(env)) * grid.dt, env


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))
ax1.plot(lam * 1e3, T, "b-", lw=1.2, label=r"$|S_{21}|^2$")
ax1.plot(lam * 1e3, R, "r-", lw=0.8, alpha=0.6, label=r"$|S_{11}|^2$")
ax1.axvline(1e3 / f0, color="gray", ls=":", lw=0.8)
ax1.set_xlabel("wavelength [nm]"); ax1.set_ylabel("power")
ax1.set_title(f"S-parameters (mode overlap), Q~{Q:.0f}"
              + (f", FSR {fsr_nm:.0f} nm" if fsr_nm else ""))
ax1.legend(fontsize=8); ax1.set_ylim(0, 1.1)

im = ax2.imshow(eps.numpy().T, origin="lower", cmap="viridis",
                extent=[0, grid.nx * grid.dx, 0, grid.ny * grid.dx])
ax2.plot([p_in.index * grid.dx], [o1y * grid.dx], "r>", ms=8)
ax2.plot([p_out.index * grid.dx], [o2y * grid.dx], "rs", ms=6)
ax2.set_title("gdsfactory ring_single (r=5, gap=200 nm)")
ax2.set_xlabel("x [um]")

for dk, col in ((0.0, "tab:red"), (1.0, "tab:orange"), (2.0, "tab:blue")):
    t, env = transient(dk)
    ax3.plot(t, env / env.max() if dk == 0.0 else env / env0_max,
             color=col, lw=1.0, label=rf"$\delta = {dk:.0f}\,\kappa$")
    if dk == 0.0:
        env0_max = env.max()
ax3.set_xlabel("time [um/c]"); ax3.set_ylabel("ring intensity (norm.)")
ax3.set_title("ring buildup vs detuning (CW mode drive)")
ax3.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "a_ring_smatrix.png", dpi=150)
plt.close(fig)
print("a_ring_smatrix.png written")
