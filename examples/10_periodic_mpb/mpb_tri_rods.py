import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import band_figure, epsilon_figure, save

OUT = pathlib.Path(__file__).parent / "output/mpb_tri_rods"
OUT.mkdir(parents=True, exist_ok=True)

# Gamma-M-K-Gamma, the path the k_points below walk.
GMKG = [r"$\Gamma$", "M", "K", r"$\Gamma$"]

# A triangular lattice of dielectric rods in air.  (This structure has
# a band_gap for TM fields.)  This file is used in the "Data Analysis
# Tutorial" section of the MPB manual.

num_bands = 8

geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

geometry = [mp.Cylinder(0.2, material=mp.Medium(epsilon=12))]

k_points = [
    mp.Vector3(),  # Gamma
    mp.Vector3(y=0.5),  # M
    mp.Vector3(1 / -3, 1 / 3),  # K
    mp.Vector3(),  # Gamma
]

k_points = mp.interpolate(4, k_points)

resolution = 32

ms = mpb.ModeSolver(
    geometry=geometry,
    geometry_lattice=geometry_lattice,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
)
# Rule 2: output_efield_z below writes h5, and a bare prefix is relative to
# the CWD -- without this the files land wherever the script was launched.
ms.filename_prefix = str(OUT / "tri_rods")  # meep adds the "-"


def main():
    ms.run_tm(
        mpb.output_at_kpoint(
            mp.Vector3(1 / -3, 1 / 3), mpb.fix_efield_phase, mpb.output_efield_z
        )
    )
    tm = np.asarray(ms.all_freqs)
    save(*band_figure(ms, "Triangular lattice of rods: TM bands", GMKG),
         OUT, "mpb_tri_rods_tm")
    tm_gap = max((g[0] for g in ms.gap_list), default=0.0)

    ms.run_te()
    te = np.asarray(ms.all_freqs)
    save(*band_figure(ms, "Triangular lattice of rods: TE bands", GMKG),
         OUT, "mpb_tri_rods_te")
    te_gap = max((g[0] for g in ms.gap_list), default=0.0)

    # rectify=True because this lattice is OBLIQUE: the solver's grid is in
    # lattice coordinates, so an imshow of the raw array draws the rhombic
    # cell as a square and the round rods as ellipses. Correct data,
    # misleading picture -- MPBData maps it back to cartesian.
    save(*epsilon_figure(ms, "Triangular lattice of rods: $\\varepsilon$",
                         rectify=True, periods=3),
         OUT, "mpb_tri_rods_eps")

    # The TM gap is the point of this structure -- rods in air confine the
    # TM field in the high index, which is what opens it -- and TE having
    # essentially none on the same lattice is the contrast worth printing.
    print(f"mpb_tri_rods_tm.png  [{ms.active_backend}]  "
          f"largest gap: TM {tm_gap:.4f}%, TE {te_gap:.4f}%  "
          f"(TM band 1 max {tm[:, 0].max():.6f}, TE {te[:, 0].max():.6f})")


if __name__ == "__main__":
    main()
