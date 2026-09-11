import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import (band_figure, epsilon_3d_figure,
                        epsilon_figure, save)

OUT = pathlib.Path(__file__).parent / "output/mpb_hole_slab"
OUT.mkdir(parents=True, exist_ok=True)

# Photonic crystal slab consisting of a triangular lattice of air
# holes in a finite_thickness dielectric slab, optionally with a
# substrate on one side of the slab.  See the paper: S. G. Johnson,
# S. Fan, P. R. Villeneuve, J. D. Joannopoulos, L. A. Kolodziejski,
# "Guided modes in photonic crystal slabs," PRB 60, 5751 (August
# 1999).

# Note that this structure has mirror symmetry throught the z=0 plane,
# and we are looking at k_vectors in the xy plane only.  Thus, we can
# break up the modes into even and odd (analogous to TE and TM), using
# the run_zeven and run_zodd functions.

h = 0.5  # the thickness of the slab
eps = 12.0  # the dielectric constant of the slab
loweps = 1.0  # the dielectric constant of the substrate
r = 0.3  # the radius of the holes
supercell_h = 4  # height of the supercell

# triangular lattice with vertical supercell:
geometry_lattice = mp.Lattice(
    size=mp.Vector3(1, 1, supercell_h),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)

geometry = [
    mp.Block(
        material=mp.Medium(epsilon=loweps),
        center=mp.Vector3(z=0.25 * supercell_h),
        size=mp.Vector3(mp.inf, mp.inf, 0.5 * supercell_h),
    ),
    mp.Block(material=mp.Medium(epsilon=eps), size=mp.Vector3(mp.inf, mp.inf, h)),
    mp.Cylinder(r, material=mp.air, height=supercell_h),
]

# 1st Brillouin zone of a triangular lattice:
Gamma = mp.Vector3()
M = mp.Vector3(y=0.5)
K = mp.Vector3(1 / -3, 1 / 3)

only_K = False  # run with only_K=true to only do this k_point
# Upstream parameters kept as is: 16 k-points x 2 parities at 32x32x16 is
# ~43 s per k-point on an RTX A4000 (~25 min total), ~2 min per k-point on
# CPU. only_K=True is the quick look.
k_interp = 4  # the number of k points to interpolate
k_points = [K] if only_K else mp.interpolate(k_interp, [Gamma, M, K, Gamma])
resolution = mp.Vector3(32, 32, 16)
num_bands = 9

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    resolution=resolution,
    num_bands=num_bands,
    k_points=k_points,
)
ms.device = "cuda"
# Rule 2: output_hfield_z/output_dfield_z below write h5, and without a
# prefix under OUT they land in the launch directory.
ms.filename_prefix = str(OUT / "hole_slab")


# The light line, and it is the whole point of this example. A slab is
# open above and below, so a mode is only GUIDED if it cannot radiate into
# the cladding: omega < c |k_parallel|, which with c = 1 and k in units of
# 2 pi/a is simply f < |k|. Everything above that line is a leaky
# resonance, not a bound state (Johnson et al., PRB 60, 5751).
#
# |k| has to be CARTESIAN. The k_points are in the reciprocal basis, where
# Gamma->M and Gamma->K are not the same length in k, so plotting against
# the reciprocal components would bend a straight line.
def _light_line():
    kc = [mp.reciprocal_to_cartesian(k, geometry_lattice) for k in k_points]
    return np.array([math.hypot(float(v.x), float(v.y)) for v in kc])


def _bands_figure(pol, title):
    """Band diagram with the light cone shaded, plus the guided count."""
    kmag = _light_line()
    fig, table, header = band_figure(ms, title, [r"$\Gamma$", "M", "K",
                                                 r"$\Gamma$"],
                                     gap_floor=np.inf)
    ax = fig.axes[0]
    x = np.arange(len(k_points))
    top = ax.get_ylim()[1]
    ax.fill_between(x, kmag, top, color="0.85", zorder=0,
                    label="light cone (radiative)")
    ax.plot(x, kmag, "-", color="0.4", lw=1.2)
    ax.set_ylim(0, top)
    ax.legend(fontsize=8, loc="lower right")
    freqs = np.asarray(ms.all_freqs)
    guided = int((freqs < kmag[:, None]).sum())
    save(fig, table, header, OUT, f"mpb_hole_slab_{pol}")
    return guided, freqs


def main():
    # run_zeven/run_zodd used to raise here -- z-parity was a 2D-only path
    # and 3D had no eigenvectors at all. The 3D planewave backend solves
    # inside the mirror sector now, so both run; the mirrored axis is
    # forced to an odd grid so the sector is exactly invariant.

    # Run even and odd bands, outputting fields only at the K point:
    counts = {}
    if loweps == 1.0:
        # we only have even/odd classification for symmetric structure
        ms.run_zeven(mpb.output_at_kpoint(K, mpb.output_hfield_z))
        counts["zeven"], zeven = _bands_figure(
            "zeven", "Hole slab: z-even (TE-like) bands")
        ms.run_zodd(mpb.output_at_kpoint(K, mpb.output_dfield_z))
        counts["zodd"], zodd = _bands_figure(
            "zodd", "Hole slab: z-odd (TM-like) bands")
    else:
        ms.run(mpb.output_at_kpoint(K, mpb.output_hfield_z), mpb.display_zparities)
        counts["all"], zeven = _bands_figure(
            "all", f"Hole slab: bands (substrate eps = {loweps})")

    # A VERTICAL cut, not a mid-plane: this is a slab in a supercell, so
    # the picture worth having is the section showing slab, hole and the
    # air above and below. axis 1 is the second lattice direction; the cut
    # therefore runs along x and z.
    save(*epsilon_figure(ms, f"Hole slab: h = {h}, r = {r}, "
                             f"supercell {supercell_h}a", slice_axis=1),
         OUT, "mpb_hole_slab_eps")

    # And the structure itself. A section shows a slab with a gap in it;
    # only the isosurface shows that the gap is a round hole in a
    # triangular lattice, which is what the band structure is about.
    save(*epsilon_3d_figure(ms, f"Hole slab: r = {r}, h = {h}"),
         OUT, "mpb_hole_slab_3d")

    ms.display_eigensolver_stats()

    # Rule 7. The guided-mode count is the number that says the light-line
    # argument above held; the first z-even band at K is what the committed
    # oracle (hole_slab_mpb.json) is compared against.
    k_index = k_points.index(K) if K in k_points else len(k_points) // 2
    summary = ", ".join(f"{k} {v}" for k, v in counts.items())
    print(f"mpb_hole_slab_eps.png  [{ms.active_backend}]  guided modes "
          f"below the light line: {summary}  |  first z-even band at K = "
          f"{float(zeven[k_index, 0]):.6f}")


if __name__ == "__main__":
    main()
