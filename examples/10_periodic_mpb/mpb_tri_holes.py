import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import band_figure, epsilon_figure, save

OUT = pathlib.Path(__file__).parent / "output/mpb_tri_holes"
OUT.mkdir(parents=True, exist_ok=True)

# 2d system: triangular lattice of air holes in dielectric
# This structure has a complete band gap (i.e. a gap in both TE and TM
# simultaneously) for a hole radius of 0.45a and a dielectric constant of
# 12.   (See, e.g., the book "Photonic Crystals" by Joannopoulos et al.)

# first, define the lattice vectors and k-points for a triangular lattice:

geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

kz = 0  # use non-zero kz to consider vertical propagation

k_points = [
    mp.Vector3(z=kz),  # Gamma
    mp.Vector3(0, 0.5, kz),  # M
    mp.Vector3(1 / -3, 1 / 3, kz),  # K
    mp.Vector3(z=kz),  # Gamma
]

k_interp = 4
k_points = mp.interpolate(k_interp, k_points)

# Now, define the geometry, etcetera:

eps = 12  # the dielectric constant of the background
r = 0.45  # the hole radius

default_material = mp.Medium(epsilon=eps)
geometry = [mp.Cylinder(r, material=mp.air)]

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    default_material=default_material,
    resolution=resolution,
    num_bands=num_bands,
)
# No ms.device here. This is a 2D lattice, and ModeSolver.device is read
# only by the 3D planewave backend -- a 2D solver handed device="cuda"
# warns and runs on the CPU anyway. Setting it would report a GPU that
# never touched the calculation.


# Gamma-M-K-Gamma, the path the k_points above walk.
GMKG = [r"$\Gamma$", "M", "K", r"$\Gamma$"]


def main():
    gaps = {}
    if kz == 0:
        for pol, run in (("te", ms.run_te), ("tm", ms.run_tm)):
            run()
            save(*band_figure(
                ms, f"Triangular lattice of air holes: {pol.upper()} bands",
                GMKG), OUT, f"mpb_tri_holes_{pol}")
            gaps[pol] = max((g[0] for g in ms.gap_list), default=0.0)
    else:
        # kz != 0 mixes the polarizations, so there is one band set, not two
        ms.run()
        save(*band_figure(ms, f"Triangular lattice of air holes: "
                              f"bands at $k_z$ = {kz}", GMKG),
             OUT, "mpb_tri_holes_all")
        gaps["all"] = max((g[0] for g in ms.gap_list), default=0.0)

    # Oblique lattice: rectify or the rhombic cell is drawn as a square and
    # the round holes as ellipses.
    save(*epsilon_figure(ms, f"Air holes r = {r} in $\\varepsilon$ = {eps}",
                         rectify=True, periods=3),
         OUT, "mpb_tri_holes_eps")

    # This structure is the textbook COMPLETE gap: holes in dielectric open
    # a gap in TE and TM at once, which is why both numbers matter rather
    # than the larger one.
    freqs = np.asarray(ms.all_freqs)
    summary = ", ".join(f"{k.upper()} {v:.4f}%" for k, v in gaps.items())
    print(f"mpb_tri_holes_eps.png  [{ms.active_backend}]  largest gap: "
          f"{summary}  ({freqs.shape[0]} k-points, {freqs.shape[1]} bands)")


if __name__ == "__main__":
    main()
