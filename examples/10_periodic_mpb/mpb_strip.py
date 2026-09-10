import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch import mpb

from _mpb_plots import band_figure, epsilon_figure, save

OUT = pathlib.Path(__file__).parent / "output/mpb_strip"
OUT.mkdir(exist_ok=True)
# Compute modes of a rectangular Si strip waveguide on top of oxide.
# Note that you should only pay attention, here, to the guided modes,
# which are the modes whose frequency falls under the light line --
# that is, frequency < beta / 1.45, where 1.45 is the SiO2 index.

# Since there's no special lengthscale here, I'll just
# use microns.  In general, if you use units of x, the frequencies
# output are equivalent to x/lambda# so here, the freqeuncies will be
# output as um/lambda, e.g. 1.5um would correspond to the frequency
# 1/1.5 = 0.6667.

w = 0.3  # Si width (um)
h = 0.25  # Si height (um)

Si = mp.Medium(index=3.45)
SiO2 = mp.Medium(index=1.45)

# Define the computational cell.  We'll make x the propagation direction.
# the other cell sizes should be big enough so that the boundaries are
# far away from the mode field.
sc_y = 2  # supercell width (um)
sc_z = 2  # supercell height (um)
geometry_lattice = mp.Lattice(size=mp.Vector3(0, sc_y, sc_z))

# define the 2d blocks for the strip and substrate
geometry = [
    mp.Block(
        size=mp.Vector3(mp.inf, mp.inf, 0.5 * (sc_z - h)),
        center=mp.Vector3(z=0.25 * (sc_z + h)),
        material=SiO2,
    ),
    mp.Block(size=mp.Vector3(mp.inf, w, h), material=Si),
]

# The k (i.e. beta, i.e. propagation constant) points to look at, in
# units of 2*pi/um.  We'll look at num_k points from k_min to k_max.
num_k = 9
k_min = 0.1
k_max = 3.0
k_points = mp.interpolate(num_k, [mp.Vector3(k_min), mp.Vector3(k_max)])

resolution = 32  # pixels/um

# Increase this to see more modes.  (The guided ones are the ones below the
# light line, i.e. those with frequencies < kmag / 1.45, where kmag
# is the corresponding column in the output if you grep for "freqs:".)
num_bands = 4

# Rule 2: everything under output/. A bare prefix like upstream's
# "strip-" is relative to the CWD, so find_k's Poynting files land
# wherever the script was launched from -- they turned up at the
# repo root.
filename_prefix = str(OUT / "strip-")

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
    filename_prefix=filename_prefix,
)
ms.device="cuda"


def main():
    # compute num_bands lowest frequencies as a function of k. Also display
    # "parities", i.e. whether the mode is symmetric or anti_symmetric
    # through the y=0 and z=0 planes.
    ms.run(mpb.display_yparities, mpb.display_zparities)

    # Upstream prints the dispersion and stops. The whole point of the
    # comment at the top of this file is the LIGHT LINE -- only modes below
    # it are guided -- and that is a statement about a picture, so draw it.
    #
    # band_figure is given the physical x axis here rather than its default
    # k-point index: beta is one swept propagation constant, not a walk
    # between symmetry points, and the light lines are straight only in
    # beta. gap_floor is off because a waveguide dispersion has no gaps to
    # shade -- the bands simply rise.
    betas = np.array([float(k.x) for k in k_points])
    fig, table, header = band_figure(
        ms, "Si strip on oxide: guided modes below the light line",
        x=betas, xlabel=r"$\beta$  ($2\pi/\mu$m)", gap_floor=np.inf)
    ax = fig.axes[0]
    for index, style, label in ((1.45, "--", "SiO$_2$ light line, $\\beta/1.45$"),
                                (1.0, ":", "air light line, $\\beta$")):
        ax.plot(betas, betas / index, style, color="0.4", lw=1.2, label=label)
    ax.set_ylim(0, float(np.asarray(ms.all_freqs).max()) * 1.1)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, table, header, OUT, "mpb_strip_dispersion")

    # The cross-section the modes live in. size=(0, y, z) makes this a 3D
    # vector problem on a single-planewave x axis, so get_epsilon_tensor
    # returns (1, ny, nz) and epsilon_figure squeezes the collapsed axis.
    save(*epsilon_figure(ms, f"Si strip {w} x {h} um on oxide"),
         OUT, "mpb_strip_eps")

    freqs = np.asarray(ms.all_freqs)
    guided = freqs < betas[:, None] / 1.45

    ###########################################################################

    # Above, we outputted the dispersion relation: frequency (omega) as a
    # function of wavevector kx (beta).  Alternatively, you can compute
    # beta for a given omega -- for example, you might want to find the
    # modes and wavevectors at a fixed wavelength of 1.55 microns.  You
    # can do that using the find_k function:

    omega = 1 / 1.55  # frequency corresponding to 1.55um

    # Output the x component of the Poynting vector for num_bands bands at omega
    ms.find_k(
        mp.NO_PARITY,
        omega,
        1,
        num_bands,
        mp.Vector3(1),
        1e-3,
        omega * 3.45,
        omega * 0.1,
        omega * 4,
        mpb.output_poynting_x,
        mpb.display_yparities,
        mpb.display_group_velocities,
    )

    # Rule 7: one closing line carrying a number a reader can diff. The
    # guided-mode COUNT is the one that says the light-line argument above
    # actually held, and neff of the fundamental is what a designer wants.
    print(f"mpb_strip_dispersion.png  [{ms.active_backend}]  "
          f"{int(guided.sum())}/{guided.size} modes below the SiO2 light "
          f"line, neff(band 1, beta={betas[-1]:g}) = "
          f"{betas[-1] / float(freqs[-1, 0]):.4f}")


if __name__ == "__main__":
    main()
