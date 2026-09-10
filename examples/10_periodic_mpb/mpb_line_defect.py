import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import band_figure, epsilon_figure

OUT = pathlib.Path(__file__).parent / "output/mpb_line_defect"
OUT.mkdir(exist_ok=True)

# A line_defect waveguide in a 2d triangular lattice of dielectric
# rods (c.f. tri_rods.ctl), formed by a row of missing rods along the
# "x" direction.  (Here, "x" and "y" refer to the first and second
# basis directions.)  This structure supports a single guided band
# within the band gap, much like the analogous waveguide in a square
# lattice of rods (see "Photonic Crystals" by Joannopoulos et al.).

supercell_y = 7  # the (odd) number of lateral supercell periods

geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, supercell_y),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

eps = 12  # the dielectric constant of the rods
r = 0.2  # the rod radius in the bulk crystal

geometry = [mp.Cylinder(r, material=mp.Medium(epsilon=eps))]

# duplicate the bulk crystal rods over the supercell:
geometry = mp.geometric_objects_lattice_duplicates(geometry_lattice, geometry)

# add a rod of air, to erase a row of rods and form a waveguide:
geometry += [mp.Cylinder(r, material=mp.air)]

Gamma = mp.Vector3()
K_prime = mp.lattice_to_reciprocal(
    mp.Vector3(0.5), geometry_lattice
)  # edge of Brillouin zone.
k_points = mp.interpolate(4, [Gamma, K_prime])

# the bigger the supercell, the more bands you need to compute to get
# to the defect modes (the lowest band is "folded" supercell_y times):
extra_bands = 5  # number of extra bands to compute above the gap
num_bands = supercell_y + extra_bands

resolution = 32

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    num_bands=num_bands,
    resolution=resolution,
)


def main():
    # Compute the TM modes, outputting the Ez field in the *middle* of the
    # band.  (In general, the guided mode in such an air defect may have
    # exited the gap by the time it reaches the edge of the Brillouin
    # zone at K_prime.)
    # Upstream writes `ms.fix_efield_phase` here, which is an upstream typo:
    # real pymeep has only the module-level `mpb.fix_efield_phase` plus the
    # `ms.fix_field_phase()` method, so that line raises AttributeError on
    # meep too. `mpb.` is the form upstream's own siblings use
    # (mpb_tri_rods.py, mpb_data_analysis.py), so this restores consistency
    # with upstream rather than departing from it.
    # output_efield_z writes h5; without a prefix it lands in the working
    # directory rather than output/ (rule 2).
    ms.filename_prefix = str(OUT / "mpb_line_defect")
    ms.run_tm(
        mpb.output_at_kpoint(k_points[len(k_points) // 2]),
        mpb.fix_efield_phase,
        mpb.output_efield_z,
    )

    fig, table, header = band_figure(
        ms, "Line defect in a triangular lattice of rods: TM bands",
        [r"$\Gamma$", "K'"])
    fig.savefig(OUT / "mpb_line_defect_bands.png", bbox_inches="tight")
    plt.close(fig)
    np.savetxt(OUT / "mpb_line_defect_bands.csv", table, delimiter=",",
               header=header, comments="")

    fig, eps_table, eps_header = epsilon_figure(
        ms, f"Line defect, {supercell_y}-period supercell")
    fig.savefig(OUT / "mpb_line_defect_eps.png", bbox_inches="tight")
    plt.close(fig)
    np.savetxt(OUT / "mpb_line_defect_eps.csv", eps_table, delimiter=",",
               header=eps_header, comments="# ")

    best = max((g[0] for g in ms.gap_list), default=0.0)
    print(f"mpb_line_defect_bands.png  [{ms.active_backend}]  {num_bands} TM bands, "
          f"largest gap = {best:.4f}%")


if __name__ == "__main__":
    main()
