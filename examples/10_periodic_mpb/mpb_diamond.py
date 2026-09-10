import math

import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb
from _mpb_plots import band_figure
import pathlib

OUT = pathlib.Path(__file__).parent / "output/mpb_diamond/"
OUT.mkdir(exist_ok=True)


# Dielectric spheres in a diamond (fcc) lattice.  This file is used in
# the "Data Analysis Tutorial" section of the MPB manual.

sqrt_half = math.sqrt(0.5)
geometry_lattice = mp.Lattice(
    basis_size=mp.Vector3(sqrt_half, sqrt_half, sqrt_half),
    basis1=mp.Vector3(0, 1, 1),
    basis2=mp.Vector3(1, 0, 1),
    basis3=mp.Vector3(1, 1),
)

# Corners of the irreducible Brillouin zone for the fcc lattice,
# in a canonical order:
vlist = [
    mp.Vector3(0, 0.5, 0.5),  # X
    mp.Vector3(0, 0.625, 0.375),  # U
    mp.Vector3(0, 0.5, 0),  # L
    mp.Vector3(0, 0, 0),  # Gamma
    mp.Vector3(0, 0.5, 0.5),  # X
    mp.Vector3(0.25, 0.75, 0.5),  # W
    mp.Vector3(0.375, 0.75, 0.375),  # K
]

k_points = mp.interpolate(4, vlist)

# define a couple of parameters (which we can set from the command_line)
eps = 11.56  # the dielectric constant of the spheres
r = 0.25  # the radius of the spheres

diel = mp.Medium(epsilon=eps)

# A diamond lattice has two "atoms" per unit cell:
geometry = [
    mp.Sphere(r, center=mp.Vector3(0.125, 0.125, 0.125), material=diel),
    mp.Sphere(r, center=mp.Vector3(-0.125, -0.125, -0.125), material=diel),
]

# (A simple fcc lattice would have only one sphere/object at the origin.)

resolution = 16  # use a 16x16x16 grid
mesh_size = 5
num_bands = 5

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_bands,
    mesh_size=mesh_size,
)
ms.device = "cuda"
ms.filename_prefix = str(OUT /"mpb_diamond")


def main():
    # Not ported yet, and left to raise where it raises (rule 11). The
    # blocker is the fcc lattice's OBLIQUE basis (basis1.basis2 = 0.5)
    # in 3D: raster.py sends 1D/2D through the metric-aware
    # lattice-coordinate path, but 3D still rasterizes on a cartesian
    # box, so _check_basis_orthogonal stops it rather than silently
    # solving a sheared structure. A rectangular supercell or the
    # planewave eigensolver unblocks it.

    # run calculation, outputting electric_field energy density at the U point:
    ms.run(mpb.output_at_kpoint(mp.Vector3(0, 0.625, 0.375), mpb.output_dpwr))
    fig, table, header = band_figure(
        ms, "Diamond lattice of dielectric spheres",
        corner_labels=["X", "U", "L", r"$\Gamma$", "X", "W", "K"])
    fig.savefig(OUT / "mpb_diamond_bands.png", bbox_inches="tight")
    np.savetxt(OUT / "mpb_diamond_bands.csv", table, delimiter=",",
               header=header, comments="")

if __name__ == "__main__":
    main()
