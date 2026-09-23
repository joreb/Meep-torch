# From the Meep tutorial: plotting permittivity and fields of a bent waveguide
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep as mp

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from _plotdata import savefig as _savefig  # PNG + .plotdata.json sidecar

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Real meep: CPU only, no device concept. Kept for the print below.
DEVICE = "meep-cpu"

cell = mp.Vector3(16, 16, 0)
geometry = [
    mp.Block(
        mp.Vector3(12, 1, mp.inf),
        center=mp.Vector3(-2.5, -3.5),
        material=mp.Medium(epsilon=12),
    ),
    mp.Block(
        mp.Vector3(1, 12, mp.inf),
        center=mp.Vector3(3.5, 2),
        material=mp.Medium(epsilon=12),
    ),
]
pml_layers = [mp.PML(1.0)]
resolution = 10

sources = [
    mp.Source(
        mp.ContinuousSource(wavelength=2 * (11**0.5), width=20),
        component=mp.Ez,
        center=mp.Vector3(-7, -3.5),
        size=mp.Vector3(0, 1),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# output_* files go to output/ instead of the working directory
sim.use_output_directory(str(OUT))

sim.run(
    mp.at_beginning(mp.output_epsilon),
    mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)),
    until=200,
)

# Upstream's product is the ez.h5 animation (kept above); the twins
# additionally export the final steady Ez over epsilon as their parity
# artifact (png + csv), as the straight-waveguide twin does.
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

fig, (ax_eps, ax_ez) = plt.subplots(1, 2, figsize=(9, 4.5))
ax_eps.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
ax_eps.set_title("permittivity")
ax_eps.axis("off")
ax_ez.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
ax_ez.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9)
ax_ez.set_title("time-domain fields ($E_z$)")
ax_ez.axis("off")
fig.tight_layout()
_savefig(fig, OUT / "bent-waveguide_meep.png", dpi=150)

# Axes from the returned shape and the cell. PML sits inside the 16x16 cell
# here, so the full span is the cell size -- the chirped_pulse 10%-compression
# trap does not apply, but the same shape-derived rule does.
nx, ny = ez_data.shape
x = (np.arange(nx) + 0.5) * (cell.x / nx)
y = (np.arange(ny) + 0.5) * (cell.y / ny)
X, Y = np.meshgrid(x, y, indexing="ij")
data = np.column_stack((X.ravel(), Y.ravel(), ez_data.ravel()))
np.savetxt(OUT / "bent-waveguide_meep.csv", data, delimiter=",")
print(np.shape(data))

peak = float(np.max(np.abs(ez_data)))
print(f"bent-waveguide_meep.png  [{DEVICE}]  max|Ez| after t=200 = {peak:.6e}")