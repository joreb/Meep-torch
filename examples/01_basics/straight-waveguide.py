# From the Meep tutorial: plotting permittivity and fields of a straight waveguide
import os
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
import torch

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Examples default to the GPU: they are demonstrations, and the difference is
# not marginal (this cell is 3 s on CUDA against >550 s on CPU). The engine
# itself defaults to CPU -- a library should not pick a device for you -- so
# the script asks explicitly. MEEP_TORCH_DEVICE overrides; falls back to CPU
# on a machine without a card.
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")

cell = mp.Vector3(16, 8, 0)

geometry = [
    mp.Block(
        mp.Vector3(mp.inf, 1, mp.inf),
        center=mp.Vector3(),
        material=mp.Medium(epsilon=12),
    )
]

sources = [
    mp.Source(
        mp.ContinuousSource(frequency=0.15), component=mp.Ez, center=mp.Vector3(-7, 0)
    )
]

pml_layers = [mp.PML(1.0)]

resolution = 10

sim = mp.Simulation(
    cell_size=cell,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    resolution=resolution,
)

# output_* files go to output/ instead of the working directory
sim.use_output_directory(str(OUT))

sim.run(until=200)

eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

# Upstream shows two separate figures with plt.show(); the twins are headless,
# so both go into one saved png instead. Same imshow code.
fig, (ax_eps, ax_ez) = plt.subplots(1, 2, figsize=(10, 3))
ax_eps.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
ax_eps.set_title("permittivity")
ax_eps.axis("off")
ax_ez.imshow(eps_data.transpose(), interpolation="spline36", cmap="binary")
ax_ez.imshow(ez_data.transpose(), interpolation="spline36", cmap="RdBu", alpha=0.9)
ax_ez.set_title("time-domain fields ($E_z$)")
ax_ez.axis("off")
fig.tight_layout()
fig.savefig(OUT / "straight-waveguide.png", dpi=150)
plt.close(fig)

# Axes from the returned shape and the cell. PML sits inside the 16x8 cell
# here, so the full span is the cell size -- the chirped_pulse 10%-compression
# trap does not apply, but the same shape-derived rule does.
nx, ny = ez_data.shape
x = (np.arange(nx) + 0.5) * (cell.x / nx)
y = (np.arange(ny) + 0.5) * (cell.y / ny)
X, Y = np.meshgrid(x, y, indexing="ij")
data = np.column_stack((X.ravel(), Y.ravel(), ez_data.ravel()))
np.savetxt(OUT / "straight-waveguide.csv", data, delimiter=",")
print(np.shape(data))

peak = float(np.max(np.abs(ez_data)))
print(f"straight-waveguide.png  [{DEVICE}]  max|Ez| after t=200 = {peak:.6e}")