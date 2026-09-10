# Field maps for the MPB "Data Analysis Tutorial" (mpb example).
import os
import pathlib
import sys

examples_dir = os.path.realpath(os.path.dirname(__file__))
sys.path.insert(0, examples_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import field_grid_figure, save

OUT = pathlib.Path(__file__).parent / "output/mpb_data_analysis"
OUT.mkdir(exist_ok=True)


def tri_rods():
    # Import the ModeSolver defined in the mpb_tri_rods.py example
    from mpb_tri_rods import ms as tr_ms

    efields = []

    # Band function to collect the efields
    def get_efields(tr_ms, band):
        efields.append(tr_ms.get_efield(band))

    tr_ms.run_tm(
        mpb.output_at_kpoint(
            mp.Vector3(1 / -3, 1 / 3), mpb.fix_efield_phase, get_efields
        )
    )

    # Create an MPBData instance to transform the efields
    md = mpb.MPBData(rectify=True, resolution=32, periods=3)

    converted = []
    for f in efields:
        # Get just the z component of the efields
        f = f[..., 0, 2]
        # NB: the slice must stay an MPBArray -- it carries the lattice that
        # convert() needs, and np.asarray() around it strips the subclass and
        # gets you "Couldn't find 'lattice'".
        converted.append(md.convert(f))

    tr_ms.run_te()

    eps = tr_ms.get_epsilon()
    rectangular_data = md.convert(eps)

    # Upstream shows the raw and the rectified epsilon in two separate
    # windows; the twins are headless, so both go in one png (rules 1, 8).
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=150)
    for ax, arr, lab in ((axes[0], eps, f"lattice grid {tuple(eps.shape)}"),
                         (axes[1], rectangular_data,
                          f"rectified, 3 periods {tuple(rectangular_data.shape)}")):
        ax.imshow(np.asarray(arr).T, interpolation="spline36", cmap="binary",
                  origin="lower")
        ax.set_title(lab, fontsize=9)
        ax.axis("off")
    fig.suptitle("Triangular lattice of rods: dielectric")
    fig.tight_layout()
    save(fig, np.asarray(rectangular_data),
         "rectified epsilon, 3 periods, on the output lattice",
         OUT, "mpb_data_analysis_eps")

    save(*field_grid_figure(
             converted, rectangular_data,
             r"Triangular lattice of rods: TM $E_z$ at K, bands 1-8"),
         OUT, "mpb_data_analysis_efields")

    peak = float(max(np.abs(f).max() for f in converted))
    print(f"mpb_data_analysis_efields.png  [{tr_ms.active_backend}]  "
          f"{len(converted)} bands on a "
          f"{rectangular_data.shape[0]}x{rectangular_data.shape[1]} "
          f"rectified grid, max|Ez| = {peak:.6e}")


def diamond():
    # Import the ModeSolver from the mpb_diamond.py example
    from mpb_diamond import ms as d_ms

    dpwr = []

    def get_dpwr(ms, band):
        dpwr.append(ms.get_dpwr(band))

    # This used to raise: the fcc primitive cell is OBLIQUE
    # (basis1.basis2 = 0.5) and the Bloch-FDTD backend rasterizes on a
    # cartesian grid, so _check_basis_orthogonal stopped it rather than
    # let it solve a different structure silently. The 3D planewave
    # backend sees only the reciprocal duals, so it solves the primitive
    # cell directly and the guard no longer applies on that path.
    d_ms.run(mpb.output_at_kpoint(mp.Vector3(0, 0.625, 0.375), get_dpwr))

    md = mpb.MPBData(rectify=True, periods=2, resolution=32)
    converted_dpwr = [md.convert(d) for d in dpwr]

    # dpwr is a SCALAR field, and in 3D that array is 3D: a mid-plane
    # slice is what makes it image data. Taking the plane through the
    # middle of the third axis rather than index 0 keeps the spheres in
    # section instead of catching a cell boundary.
    mid = [np.asarray(d)[:, :, np.asarray(d).shape[2] // 2]
           if np.asarray(d).ndim == 3 else np.asarray(d)
           for d in converted_dpwr]
    save(*field_grid_figure(mid, mid[0],
                            "Diamond lattice of spheres: D power at U, "
                            "mid-plane slice"),
         OUT, "mpb_data_analysis_diamond")

    # Rule 7. This half had no closing line at all: it wrote a figure and
    # reported no number, which is the one thing a reader can diff against
    # a twin. The energy fraction in the mid-plane is the number that says
    # the slice caught the spheres rather than a gap between them.
    peak = float(max(np.abs(m).max() for m in mid))
    share = float(sum(m.sum() for m in mid)
                  / sum(np.asarray(d).sum() for d in converted_dpwr))
    print(f"mpb_data_analysis_diamond.png  [{d_ms.active_backend}]  "
          f"{len(mid)} bands, mid-plane holds {share:.1%} of the D energy, "
          f"max = {peak:.6e}")


if __name__ == "__main__":
    tri_rods()
    diamond()
