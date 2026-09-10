import pathlib
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import band_figure, eps_profile, save

OUT = pathlib.Path(__file__).parent / "output/mpb_bragg_sine"
OUT.mkdir(exist_ok=True)

# Compute the band structure for a Bragg mirror consisting of a
# sinusoidally-varying dielectric index.

# The index will vary sinusoidally between index-min and index-max:
index_min = 1
index_max = 3


# Define a function of position p (in the lattice basis) that returns
# the material at that position.  In this case, we use the function:
#        index-min + 0.5 * (index-max - index-min)
#                        * (1 + cos(2*pi*x))
# This is periodic, and also has inversion symmetry.
def eps_func(p):
    return mp.Medium(
        index=index_min
        + 0.5 * (index_max - index_min) * (1 + math.cos(2 * math.pi * p.x))
    )


geometry_lattice = mp.Lattice(size=mp.Vector3(1))  # 1d cell

# We'll just make it the default material, so that it goes everywhere.
default_material = eps_func

k_points = mp.interpolate(9, [mp.Vector3(), mp.Vector3(x=0.5)])

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    num_bands=num_bands,
    k_points=k_points,
    geometry_lattice=geometry_lattice,
    resolution=resolution,
    default_material=default_material,
)


def main():
    # the TM and TE bands are degenerate, so we only need TM:
    ms.run_tm()

    # Upstream stops there. The path is a single Gamma->X segment, so unlike
    # mpb_bragg (one k-point, band LIST) this one has a real band diagram to
    # draw, and the seven gaps are what the graded stack is for.
    save(*band_figure(
        ms, "Sinusoidal Bragg mirror: TM bands", [r"$\Gamma$", "X"]),
        OUT, "mpb_bragg_sine_bands")

    # The lattice is defined by a FUNCTION, not by geometry objects, so the
    # useful epsilon panel is the sampled grid against the n(x)^2 it was
    # drawn from -- that is what shows default_material=<callable> reached
    # the solver. epsilon_figure() is not used here: it imshows the grid,
    # and a (32, 1) cell renders as a one-pixel strip.
    eps = np.asarray(ms.get_epsilon())[:, 0]
    x = (np.arange(eps.size) + 0.5) / eps.size - 0.5   # cell centres
    analytic = (index_min + 0.5 * (index_max - index_min)
                * (1 + np.cos(2 * np.pi * x))) ** 2

    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    table, header = eps_profile(
        ax, ms, f"index {index_min} to {index_max}, resolution {resolution}",
        reference=analytic, mode="points")
    fig.tight_layout()
    save(fig, table, header, OUT, "mpb_bragg_sine_eps")

    dev = float(np.abs(eps - analytic).max())
    first_gap = ms.gap_list[0][0] if ms.gap_list else 0.0
    print(f"mpb_bragg_sine_bands.png  [{ms.active_backend}]  first gap = "
          f"{first_gap:.4f}%  max|eps - n(x)^2| = {dev:.3e}")


if __name__ == "__main__":
    main()
