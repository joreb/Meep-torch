# Bands at X for a quarter-wave Bragg stack (mpb example).
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import eps_profile, save


OUT = pathlib.Path(__file__).parent / "output/mpb_bragg"
OUT.mkdir(exist_ok=True)

# Compute the bands at the X point for a quarter-wave stack Bragg
# mirror (this is the point that defines the band gap edges).

# the high and low indices:
n_lo = 1.0
n_hi = 3.0

w_hi = n_lo / (n_hi + n_lo)  # a quarter_wave stack

geometry_lattice = mp.Lattice(size=mp.Vector3(1))  # 1d cell
default_material = mp.Medium(index=n_lo)
geometry = mp.Cylinder(
    material=mp.Medium(index=n_hi),
    center=mp.Vector3(),
    axis=mp.Vector3(1),
    radius=mp.inf,
    height=w_hi,
)

kx = 0.5
k_points = [mp.Vector3(kx)]

resolution = 32
num_bands = 8

ms = mpb.ModeSolver(
    num_bands=num_bands,
    k_points=k_points,
    geometry_lattice=geometry_lattice,
    geometry=[geometry],
    resolution=resolution,
    default_material=default_material,
)


def main():
    # h5 output goes to output/, not the working directory
    ms.filename_prefix = str(OUT / "mpb_bragg")
    ms.run_tm(
        mpb.output_hfield_y
    )  # note that TM and TE bands are degenerate, so we only need TM

    # Upstream stops there. One k-point makes a band DIAGRAM meaningless,
    # so the export is the structure and the band list instead: the whole
    # point of the quarter-wave stack is where the gap edges land, and the
    # epsilon profile is what shows the stack is a stack at all -- the
    # geometry is a Cylinder along X with radius=inf, i.e. a slab, and
    # rasterizing it as a disc would fill the cell and still give bands.
    freqs = [float(f) for f in ms.get_freqs()]

    fig, (ax_eps, ax_b) = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
    eps_table, eps_header = eps_profile(
        ax_eps, ms, f"quarter-wave stack, $w_{{hi}}$ = {w_hi:.3f}")
    ax_b.plot(np.arange(1, len(freqs) + 1), freqs, "o-", color="tab:red")
    ax_b.set_xlabel("band")
    ax_b.set_ylabel("frequency (c/a)")
    ax_b.set_title(f"bands at X ($k_x$ = {kx})")
    ax_b.grid(alpha=0.3)
    fig.tight_layout()
    # NOT save(): that writes one CSV per figure, and this figure is two
    # panels with a CSV each (the stack and the band list). eps_profile
    # still does the drawing and hands back the numbers behind its panel.
    fig.savefig(OUT / "mpb_bragg.png", bbox_inches="tight")
    plt.close(fig)
    np.savetxt(OUT / "mpb_bragg_eps.csv", eps_table, delimiter=",",
               header=eps_header, comments="")
    np.savetxt(OUT / "mpb_bragg_bands.csv",
               np.column_stack((np.arange(1, len(freqs) + 1), freqs)),
               delimiter=",", header="band,frequency", comments="")

    gap = (freqs[1] - freqs[0]) / (0.5 * (freqs[0] + freqs[1])) * 100
    print(f"mpb_bragg.png  [{ms.active_backend}]  gap at X = {gap:.4f}%  "
          f"(bands {freqs[0]:.6f}, {freqs[1]:.6f})")


if __name__ == "__main__":
    main()
