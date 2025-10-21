# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (a2): ADD-DROP ring resonator -- gdsfactory ring_double on
the real SiN platform, 4-port mode-overlap S-matrix.

Platform: SiN 150 nm on SiO2, indices from the meep materials library
(Si3N4 Luke, SiO2 Horiba at 1550 nm), lateral effective index. Ring
r = 50 um, W = 4 um, both gaps 0.8 um. Ports: o1 in, o2 through,
o3 drop (top-left, counter-propagating pickup), o4 add.
At resonance: through dips, drop peaks -- |S21|^2 and |S31|^2 spectra
from one broadband TE0 mode-source run.

Run: uv run python examples_2/a_ring_adddrop.py
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
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.core.materials import Si3N4, SiO2
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
R_RING, W_WG, GAP = 50.0, 4.0, 0.8
print(f"a_ring_adddrop  n_core={NEFF:.4f} n_clad={N_CLAD:.4f}; "
      f"r={R_RING} W={W_WG} gaps={GAP}")

RES = 15
FCEN = 1 / 1.55
dx = 1.0 / RES
xs = gf.get_cross_section("strip").copy(width=W_WG)
c = gf.components.ring_double(radius=R_RING, gap=GAP, cross_section=xs)
grid, eps, ports = component_to_eps(c, res=RES, pad=3.0,
                                    n_core=NEFF, n_clad=N_CLAD)
nx, ny = grid.nx, grid.ny
NPML = RES
print(f"a_ring_adddrop  grid {nx}x{ny} ({nx * dx:.0f} x {ny * dx:.0f} um)")

SPAN = int(3.5 * RES)


def mk_port(name, direction):
    px, py, _ = ports[name]
    return Port(axis="x", index=min(max(px + (4 if direction > 0 else -4),
                                        NPML + 4), nx - NPML - 5),
                lo=py - SPAN, hi=py + SPAN, direction=direction)


p_in = mk_port("o1", +1)
p_thru = mk_port("o2", -1)
p_drop = mk_port("o3", +1)      # inward +x; the drop wave leaves at -x (a_m)
p_add = mk_port("o4", -1)
validate_ports([p_in, p_thru, p_drop, p_add], grid, NPML)
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)

NFREQ = 160
DF_BAND = 0.0125
freqs = np.linspace(FCEN - DF_BAND / 2, FCEN + DF_BAND / 2, NFREQ).tolist()
T_RUN = 650.0
plist = [p_in, p_thru, p_drop, p_add]
mons = [FluxMonitorTM(grid, freqs, x_normal=True, idx=p.index, lo=p.lo, hi=p.hi)
        for p in plist]
src = ModeSourceTM(grid, coeffs, p_in, GaussianPulse(FCEN, 0.04), FCEN)
with torch.no_grad():
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons, checkpoint_every=400)
amp = {}
for p, acc, name in zip(plist, accs, ("in", "thru", "drop", "add")):
    el = eps[p.index, p.lo:p.hi]
    amp[name] = decompose(acc, el, dx, freqs, "tm", 1, axis_sign=-1.0)
a_in = amp["in"][0][:, 0]
S21 = (amp["thru"][0][:, 0] / a_in).numpy()         # +x outgoing at o2
S31 = (amp["drop"][1][:, 0] / a_in).numpy()         # -x outgoing at o3
S41 = (amp["add"][0][:, 0] / a_in).numpy()          # +x outgoing at o4
S11 = (amp["in"][1][:, 0] / a_in).numpy()
lam = 1e3 / np.array(freqs)
T, D = np.abs(S21) ** 2, np.abs(S31) ** 2

EDGE = NFREQ // 10
i0 = EDGE + int(np.argmax(D[EDGE:NFREQ - EDGE]))    # strongest drop peak
f0 = freqs[i0]
half = D[i0] / 2
i_lo = i0
while i_lo > 0 and D[i_lo] > half:
    i_lo -= 1
i_hi = i0
while i_hi < NFREQ - 1 and D[i_hi] > half:
    i_hi += 1
Q = f0 / max(freqs[i_hi] - freqs[i_lo], 1e-9)
print(f"a_ring_adddrop  drop peak lambda={1e3 / f0:.2f} nm "
      f"D={D[i0] * 100:.1f}% T={T[i0] * 100:.1f}% Q~{Q:.0f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
ax1.plot(lam, T, "b-", lw=1.1, label=r"through $|S_{21}|^2$")
ax1.plot(lam, D, "g-", lw=1.1, label=r"drop $|S_{31}|^2$")
ax1.plot(lam, np.abs(S41) ** 2, "r-", lw=0.7, alpha=0.6,
         label=r"add $|S_{41}|^2$")
ax1.plot(lam, np.abs(S11) ** 2, "k-", lw=0.6, alpha=0.5,
         label=r"$|S_{11}|^2$")
ax1.axvline(1e3 / f0, color="gray", ls=":", lw=0.8)
ax1.set_xlabel("wavelength [nm]"); ax1.set_ylabel("power")
ax1.set_title(f"add-drop ring (r=50 um): drop Q~{Q:.0f}")
ax1.legend(fontsize=8); ax1.set_ylim(0, 1.1)
step = max(1, nx // 800)
ax2.imshow(eps.numpy()[::step, ::step].T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx])
ax2.set_title(f"gdsfactory ring_double r={R_RING:.0f} W={W_WG} gaps={GAP}")
ax2.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "a_ring_adddrop.png", dpi=150)
plt.close(fig)
print("a_ring_adddrop.png written")
