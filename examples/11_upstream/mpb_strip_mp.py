# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_strip.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the meep import and the generated ModeSolver lines, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE = os.environ.get("MEEP_TORCH_DEVICE")`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
if not hasattr(_np, "trapz") and hasattr(_np, "trapezoid"):
    _np.trapz = _np.trapezoid        # upstream predates numpy 2.0, which removed it
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/mpb_strip/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import meep_torch.mp as mp
from meep_torch.mp import mpb

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

filename_prefix = "strip-"  # use this prefix for output files

ms = mpb.ModeSolver(
    geometry_lattice=geometry_lattice,
    geometry=geometry,
    k_points=k_points,
    resolution=resolution,
    num_bands=num_bands,
    filename_prefix=filename_prefix,
)
ms.device = DEVICE   # generated: 3D planewave backend only
ms.filename_prefix = str(OUT / "mpb_strip")


def main():
    # compute num_bands lowest frequencies as a function of k. Also display
    # "parities", i.e. whether the mode is symmetric or anti_symmetric
    # through the y=0 and z=0 planes.
    ms.run(mpb.display_yparities, mpb.display_zparities)

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


if __name__ == "__main__":
    main()
