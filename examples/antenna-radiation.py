# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/antenna-radiation.py.

Radiation pattern of a dipole antenna in free space via the 2D near-to-far
transform (W4): Ez (TM, isotropic in 2D) and Ey (TE, cos^2 lobes) point
dipoles, radial Poynting flux over [0, 2 pi) at r = 1000 lambda, plotted
against the analytic patterns.

SUBSTITUTIONS:
  * meep's get_farfield exact green2d -> meep_torch near2far (exact Hankel
    kernels, far H from the local plane-wave relation, O(1/kR)).
  * Ex polarization is skipped (maps to the same TE physics as Ey, rotated).

Resolution 20 (meep tutorial uses 50); noted.

Run: uv run python examples/antenna-radiation.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp

OUT = pathlib.Path(__file__).parent

FREQ = 1.0                       # wavelength 1 um
CELL = 4.0
R_FAR = 1000.0
NUM_POLAR = 100
polar_rad = np.linspace(0, 2 * math.pi, NUM_POLAR, endpoint=False)


def radiation_pattern(component):
    sim = mp.Simulation(cell_size=mp.Vector3(6, 6, 0), resolution=20,
        sources=[mp.Source(mp.GaussianSource(FREQ, fwidth=0.2), component=component,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    n2f = sim.add_near2far(FREQ, 0, 1,
        mp.Near2FarRegion(center=mp.Vector3(0, +CELL / 2), size=mp.Vector3(CELL, 0), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(0, -CELL / 2), size=mp.Vector3(CELL, 0), weight=-1),
        mp.Near2FarRegion(center=mp.Vector3(+CELL / 2, 0), size=mp.Vector3(0, CELL), weight=+1),
        mp.Near2FarRegion(center=mp.Vector3(-CELL / 2, 0), size=mp.Vector3(0, CELL), weight=-1))
    sim.run(until=80)
    ff = np.array([sim.get_farfield(
        n2f, mp.Vector3(R_FAR * math.cos(t), R_FAR * math.sin(t))) for t in polar_rad])
    e, h = ff[:, :3], ff[:, 3:]
    sx = np.real(np.conj(e[:, 1]) * h[:, 2] - np.conj(e[:, 2]) * h[:, 1])
    sy = np.real(np.conj(e[:, 2]) * h[:, 0] - np.conj(e[:, 0]) * h[:, 2])
    return np.sqrt(sx ** 2 + sy ** 2)


flux_ez = radiation_pattern(mp.Ez)
flux_ey = radiation_pattern(mp.Ey)

fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"}, figsize=(10, 4.6))
for ax, flux, analytic, title in (
    (axes[0], flux_ez, np.ones(NUM_POLAR), "Ez dipole (isotropic)"),
    (axes[1], flux_ey, np.cos(polar_rad) ** 2, "Ey dipole (cos^2)"),
):
    ax.plot(polar_rad, flux / flux.max(), "b-", lw=1.2, label="meep-torch n2f")
    ax.plot(polar_rad, analytic, "r--", lw=1.0, label="analytic")
    ax.set_title(title)
    ax.set_rmax(1.05)
axes[0].legend(loc="lower left", bbox_to_anchor=(-0.15, -0.15), fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "antenna-radiation.png", dpi=150)
plt.close(fig)

rms_ez = float(np.sqrt(np.mean((flux_ez / flux_ez.max() - 1.0) ** 2)))
rms_ey = float(np.sqrt(np.mean((flux_ey / flux_ey.max() - np.cos(polar_rad) ** 2) ** 2)))
print(f"antenna-radiation.png  Ez isotropy rms={rms_ez:.4f}  Ey cos^2 rms={rms_ey:.4f}")
