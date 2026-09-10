# Band structure of a square lattice of dielectric rods in air (mpb example).
import pathlib
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb


from _mpb_plots import band_figure, epsilon_figure

OUT = pathlib.Path(__file__).parent / "output/mpb_sq_rods"
OUT.mkdir(exist_ok=True)

# Define various parameters with define_param so that they are
# settable from the command_line (with mpb <param>=<value>):
r = 0.2  # radius of the rods
eps = 11.56  # dielectric constant
k_interp = 4  # number of k points to interpolate

GaAs = mp.Medium(epsilon=eps)

geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))  # 2d cell

geometry = [mp.Cylinder(r, material=GaAs)]

Gamma = mp.Vector3()
X = mp.Vector3(0.5, 0)
M = mp.Vector3(0.5, 0.5)
k_points = mp.interpolate(k_interp, [Gamma, X, M, Gamma])
GXMG = [r"$\Gamma$", "X", "M", r"$\Gamma$"]

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
)


def main():
    # Compute the TE and TM bands and report the total elapsed time:
    t0 = time.time()
    gaps = {}
    # Upstream runs both polarizations and prints timings. The twins need
    # something a reader can diff, so each run also leaves a band diagram
    # and the CSV behind it (rule 4 of examples/example_template.py).
    for pol, run in (("te", ms.run_te), ("tm", ms.run_tm)):
        run()
        gaps[pol] = [g[0] for g in ms.gap_list]
        fig, table, header = band_figure(
            ms, f"Square lattice of rods: {pol.upper()} bands", GXMG)
        fig.savefig(OUT / f"mpb_sq_rods_{pol}.png", bbox_inches="tight")
        plt.close(fig)
        np.savetxt(OUT / f"mpb_sq_rods_{pol}.csv", table, delimiter=",",
                   header=header, comments="")
    print(f"total time for both TE and TM bands: {time.time() - t0:.2f} seconds")

    fig, eps_table, eps_header = epsilon_figure(ms, "Square lattice of rods")
    fig.savefig(OUT / "mpb_sq_rods_eps.png", bbox_inches="tight")
    plt.close(fig)
    np.savetxt(OUT / "mpb_sq_rods_eps.csv", eps_table, delimiter=",",
               header=eps_header, comments="# ")

    ms.display_eigensolver_stats()

    best = max(gaps["tm"], default=0.0)
    print(f"mpb_sq_rods.png  [{ms.active_backend}]  largest TM gap = {best:.4f}%")


if __name__ == "__main__":
    main()
