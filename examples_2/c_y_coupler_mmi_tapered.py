# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (c2d): Y coupler from gdsfactory mmi_tapered -- FULL-domain
characterization (no symmetry exploited, per user spec), 674 nm.

Component: gdsfactory mmi_tapered at DEFAULT parameters (user spec),
native 0.5 um strip (the W=4 variant self-overlaps)
plus 4 um straights on all three ports (into the PML). TE0 mode source
2 um from the input-straight end; TE0 overlap monitors 2 um from each
output-straight end. Platform: SiN 150 nm / SiO2, meep materials-
library indices at 674 nm, lateral effective index. GPU.

Run: uv run python examples_2/c_y_coupler_mmi_tapered.py
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
from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.core.monitors import DFTFieldsMonitor, FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.modes.fd1d import solve_1d
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

WL = 0.674
FCEN = 1 / WL
N_SIN = Si3N4.n(FCEN).real
N_CLAD = SiO2.n(FCEN).real
eps_line = np.full(400, N_CLAD ** 2)
eps_line[200 - 7: 200 + 8] = N_SIN ** 2
NEFF = float(solve_1d(eps_line, 0.01, FCEN, "te", k=1).neff[0])
W_WG = 0.5                                          # gf default strip width
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"c_y_coupler_mmi_tapered  @{WL*1e3:.0f}nm n_core={NEFF:.4f} "
      f"n_clad={N_CLAD:.4f} W={W_WG}")

xs = gf.get_cross_section("strip")            # DEFAULT cross-section (0.5 um)
# FULLY default mmi_tapered (user spec) -- note the W=4 cross-section
# variant is geometrically degenerate (4 um output tapers at +/-1.125 um
# centers overlap -> self-intersecting polygon), so default means the
# native 0.5 um strip throughout.
# length_mmi=34 = 3*Lpi/8 for this platform (Lpi = 4 n W_e^2 / 3 lam,
# W_e ~ 5.3 um incl. Goos-Haenchen penetration, n=1.632 @674nm); outputs
# moved onto the two-fold image positions +/- W_e/4 = +/-1.32 um.
mmi = gf.components.mmi_tapered(length_mmi=34.0,
                                output_positions=[-1.32, 1.32])
comp = gf.Component("y_coupler_mmi_tapered")
ref = comp.add_ref(mmi)
for pname in ("o1", "o2", "o3"):
    st = comp.add_ref(gf.components.straight(length=4.0, cross_section=xs))
    st.connect("o1", ref.ports[pname])
    comp.add_port(pname, port=st.ports["o2"])
comp.flatten()

RES = 30
dx = 1.0 / RES
grid_c, eps_c, ports = component_to_eps(comp, res=RES, pad=2.5,
                                        n_core=NEFF, n_clad=N_CLAD)
nx, ny = grid_c.nx, grid_c.ny
grid = Grid(nx=nx, ny=ny, dx=dx, device=DEV)
eps = eps_c.to(DEV)
NPML = RES
T_RUN = 2 * (5 / 0.2) + (nx * dx) * 1.7 + 60.0      # tail + transit + margin
print(f"c_y_coupler_mmi_tapered  grid {nx}x{ny} "
      f"({nx*dx:.1f} x {ny*dx:.1f} um) on {DEV}, T_RUN={T_RUN:.0f}")

o1x, o1y, _ = ports["o1"]
o2x, o2y, _ = ports["o2"]
o3x, o3y, _ = ports["o3"]
SPAN = int(1.8 * RES)                               # 0.25 um half-width + tails
p_in = Port(axis="x", index=max(o1x + int(2.0 * RES), NPML + 4),
            lo=o1y - SPAN, hi=o1y + SPAN, direction=+1)
p_up = Port(axis="x", index=min(o2x - int(2.0 * RES), nx - NPML - 4),
            lo=o2y - SPAN, hi=o2y + SPAN, direction=-1)
p_dn = Port(axis="x", index=min(o3x - int(2.0 * RES), nx - NPML - 4),
            lo=o3y - SPAN, hi=o3y + SPAN, direction=-1)
validate_ports([p_in, p_up, p_dn], grid, NPML)
coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=NPML)
src = ModeSourceTM(grid, coeffs, p_in, GaussianPulse(FCEN, 0.2), FCEN)
mons = [FluxMonitorTM(grid, [FCEN], x_normal=True, idx=p.index, lo=p.lo,
                      hi=p.hi) for p in (p_in, p_up, p_dn)]
mon_ez = DFTFieldsMonitor(grid, [FCEN], "Ez")
with torch.no_grad():
    _, accs = run_tm(init_tm_state(grid), coeffs, int(round(T_RUN / grid.dt)),
                     source=src, monitors=mons + [mon_ez],
                     checkpoint_every=400)
out = []
for p, acc in zip((p_in, p_up, p_dn), accs[:3]):
    el = eps.cpu()[p.index, p.lo:p.hi] if p.axis == "x" else None
    out.append(decompose(acc.cpu(), el, dx, [FCEN], "tm", 1, axis_sign=-1.0))
a_in = out[0][0][0, 0]
S21 = out[1][0][0, 0] / a_in
S31 = out[2][0][0, 0] / a_in
S11 = out[0][1][0, 0] / a_in
T_up, T_dn = float(abs(S21) ** 2), float(abs(S31) ** 2)
tot = T_up + T_dn
il_db = -10 * math.log10(max(tot, 1e-12))
imb = abs(T_up - T_dn) / max(tot, 1e-12)
print(f"c_y_coupler_mmi_tapered  |S21|^2={T_up*100:.1f}% |S31|^2={T_dn*100:.1f}% "
      f"total={tot*100:.1f}% (IL {il_db:.2f} dB), imbalance {imb*100:.2f}%, "
      f"|S11|^2={float(abs(S11)**2)*100:.2f}%")

acc_ez = accs[3].cpu()
ez = np.abs((acc_ez[0, 0] + 1j * acc_ez[0, 1]).numpy())
core = (eps_c > 0.5 * (float(eps_c.min()) + float(eps_c.max()))).numpy()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
ax1.imshow(eps_c.numpy().T, origin="lower", cmap="viridis",
           extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax1.set_title("gdsfactory mmi_tapered 1x2 (length_mmi=34, outputs on image positions, 674 nm)")
ax1.set_ylabel("y [um]")
im = ax2.imshow(ez.T, origin="lower", cmap="inferno",
                extent=[0, nx * dx, 0, ny * dx], aspect="auto")
ax2.contour(np.linspace(0, nx * dx, nx), np.linspace(0, ny * dx, ny),
            core.T.astype(float), levels=[0.5], colors="w", linewidths=0.4)
ax2.set_title(f"$|E_z|$ at 674 nm: total {tot*100:.1f}%, "
              f"IL {il_db:.2f} dB, imbalance {imb*100:.2f}%")
ax2.set_xlabel("x [um]"); ax2.set_ylabel("y [um]")
fig.tight_layout()
fig.savefig(OUT / "c_y_coupler_mmi_tapered.png", dpi=150)
plt.close(fig)
print("c_y_coupler_mmi_tapered.png written")
