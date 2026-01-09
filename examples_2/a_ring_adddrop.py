# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""device-suite (a2): ADD-DROP ring -- assembled from the half-ring coupler
extraction (see a_ring_smatrix.py).

gdsfactory ring_double(r=50, W=1, both gaps 0.2, 674 nm) has two IDENTICAL
couplers, each equal to the ring_single coupler measured by the two
half-ring FDTD runs (t, kappa, and the bare half-loop a = A/t, which is
also exactly the coupler-to-coupler path). Two-coupler formulas:

    D       = 1 - t^2 a^2
    S_thru  = t - kappa^2 t a^2 / D          (port o1 -> o2)
    S_drop  = -kappa^2 a / D                 (port o1 -> o3)

evaluated on a dense frequency grid (magnitude + unwrapped-phase
interpolation of the measured coupler quantities). No additional FDTD.

Run AFTER a_ring_smatrix.py:  uv run python examples_2/a_ring_adddrop.py
"""
import math
import pathlib
import sys

import gdsfactory as gf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from gds_meep_torch import component_to_eps

from meep_torch.core.materials import Si3N4, SiO2
from meep_torch.modes.fd1d import solve_1d

OUT = pathlib.Path(__file__).parent
gf.gpdk.PDK.activate()

DATA = OUT / "a_ring_coupler_data.npz"
if not DATA.exists():
    sys.exit("run examples_2/a_ring_smatrix.py first (writes the coupler npz)")
d = np.load(DATA)
freqs, t, kap, loop = d["freqs"], d["t"], d["kappa"], d["loop"]
a_half = np.sqrt(np.abs(loop)) * np.exp(1j * np.unwrap(np.angle(loop)) / 2)  # loop already |t|-normalized

fd = np.linspace(freqs[0], freqs[-1], 6000)


def interp_c(z):
    m = np.interp(fd, freqs, np.abs(z))
    p = np.interp(fd, freqs, np.unwrap(np.angle(z)))
    return m * np.exp(1j * p)


# |t|, |kappa| real (external bus phases stripped); ring phase in a_half
ti = np.interp(fd, freqs, np.abs(t))
ki = np.interp(fd, freqs, np.abs(kap))
ai = interp_c(a_half)
D = 1 - ti ** 2 * ai ** 2
S_thru = ti - ki ** 2 * ti * ai ** 2 / D
S_drop = -ki ** 2 * ai / D
T, Dr = np.abs(S_thru) ** 2, np.abs(S_drop) ** 2
lam = 1e3 / fd

i0 = int(np.argmax(Dr))
half = Dr[i0] / 2
i_lo = i0
while i_lo > 0 and Dr[i_lo] > half:
    i_lo -= 1
i_hi = i0
while i_hi < len(fd) - 1 and Dr[i_hi] > half:
    i_hi += 1
Q = fd[i0] / max(fd[i_hi] - fd[i_lo], 1e-12)
pks = [i for i in range(2, len(fd) - 2)
       if Dr[i] > 3 * np.median(Dr) and Dr[i] >= Dr[i - 1] and Dr[i] > Dr[i + 1]]
fsr_nm = (abs(1e3 / fd[pks[0]] - 1e3 / fd[pks[-1]]) / (len(pks) - 1)
          if len(pks) >= 2 else float("nan"))
print(f"a_ring_adddrop  drop peak lambda={lam[i0]:.2f} nm "
      f"D={Dr[i0] * 100:.1f}% T_thru={T[i0] * 100:.1f}% Q={Q:.0f} "
      f"FSR={fsr_nm:.2f} nm")

# geometry panel: gf ring_double (coarse rasterization for the figure)
N_SIN = Si3N4.n(1 / 0.674).real
N_CLAD = SiO2.n(1 / 0.674).real
xs = gf.get_cross_section("strip").copy(width=1.0)
c = gf.components.ring_double(radius=50.0, gap=0.2, cross_section=xs)
grid_g, eps_g, _ = component_to_eps(c, res=4, pad=3.0,
                                    n_core=N_SIN, n_clad=N_CLAD)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
ax1.plot(lam, T, "b-", lw=0.9, label=r"through $|S_{21}|^2$")
ax1.plot(lam, Dr, "g-", lw=0.9, label=r"drop $|S_{31}|^2$")
ax1.axvline(lam[i0], color="gray", ls=":", lw=0.8)
ax1.set_xlabel("wavelength [nm]"); ax1.set_ylabel("power")
ax1.set_title(f"add-drop (half-ring method): drop Q={Q:.0f}, "
              f"FSR={fsr_nm:.2f} nm")
ax1.legend(fontsize=8); ax1.set_ylim(0, 1.05)
ax2.imshow(eps_g.numpy().T, origin="lower", cmap="viridis",
           extent=[0, grid_g.nx / 4, 0, grid_g.ny / 4])
ax2.set_title("gdsfactory ring_double r=50 W=1 gaps=0.2 (674 nm)")
ax2.set_xlabel("x [um]")
fig.tight_layout()
fig.savefig(OUT / "a_ring_adddrop.png", dpi=150)
plt.close(fig)
print("a_ring_adddrop.png written")
