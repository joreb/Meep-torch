## linear-chirped pulse planewave with higher frequencies at the front (down-chirp)
import numpy as np

import meep_torch.mp as mp
import os
import pathlib
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# Examples default to the GPU: they are demonstrations, and the difference is
# not marginal (this cell is 3 s on CUDA against >550 s on CPU). The engine
# itself defaults to CPU -- a library should not pick a device for you -- so
# the script asks explicitly. MEEP_TORCH_DEVICE overrides; falls back to CPU
# on a machine without a card.
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")

resolution = 40

dpml = 2
pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

sx = 40
sy = 6
cell_size = mp.Vector3(sx + 2 * dpml, sy)

v0 = 1.0  # pulse center frequency
a = 0.2  # Gaussian envelope half-width
b = -0.5  # linear chirp rate (positive: up-chirp, negative: down-chirp)
t0 = 15  # peak time

chirp = lambda t: np.exp(1j * 2 * np.pi * v0 * (t - t0)) * np.exp(
    -a * (t - t0) ** 2 + 1j * b * (t - t0) ** 2
)

sources = [
    mp.Source(
        src=mp.CustomSource(src_func=chirp),
        center=mp.Vector3(-0.5 * sx),
        size=mp.Vector3(y=sy),
        component=mp.Ez,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    resolution=resolution,
    k_point=mp.Vector3(),
    sources=sources,
    symmetries=[mp.Mirror(mp.Y)],
)

# output_* files go to output/ instead of the working directory
sim.use_output_directory(str(OUT))

sim.run(
    mp.in_volume(
        mp.Volume(center=mp.Vector3(), size=mp.Vector3(sx, sy)),
        mp.at_every(2.7, mp.output_efield_z),
    ),
    until=t0 + 50,
)


ez = sim.get_array(component=mp.Ez)
ez_slice = ez[:, ez.shape[1] // 2]              # centre row
# x from the returned array and the cell, not sim.grid -- meep exposes no
# .grid, so reaching for one is a shim-ism that would not run upstream.
x = (np.arange(ez.shape[0]) + 0.5) * (sx / ez.shape[0])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, ez_slice, lw=0.8)
ax.set_title("Chirped pulse -- Ez(x) at final time (down-chirp)")
ax.set_xlabel("x (um)")
ax.set_ylabel("Ez")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "chirped_pulse.png", dpi=150)
plt.close(fig)

peak = float(np.max(np.abs(ez_slice)))
print(f"chirped_pulse.png  [{DEVICE}]  max|Ez| in snapshot = {peak:.6e}")
