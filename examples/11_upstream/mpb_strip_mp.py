# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/mpb_strip.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the meep import and the generated ModeSolver lines, so that a difference between
# this file and its twin is a difference between the ENGINES. See that
# script's docstring for why nothing here rewrites a call site.
#
# The literal spellings below (`matplotlib.use("Agg")`, `OUT = ...`,
# `OUT.mkdir`, `DEVICE`) are the ones
# examples/example_template.py greps for. An earlier version used private
# `_mpl` / `_OUT` names, which satisfied the RULES and failed the CHECKER on
# 216 counts -- two dialects for one convention, and the checker was right to
# complain: a rule nothing can verify is a comment.
import os
import pathlib

import matplotlib
import numpy as _np

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
OUT = pathlib.Path(__file__).parent / "output/mpb_strip/"
OUT.mkdir(exist_ok=True)
DEVICE = "cuda"
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/


def _dump_plot_data(_fig, _path):
    """Write the NUMBERS behind a figure next to it, as <name>.plotdata.json.

    Several tutorials compute a result and only plot it -- binary_grating_
    phasemap, gaussian-beam and refl-quartz print nothing numeric and write no
    data file, so a stdout-and-files comparison has nothing to say about them
    at all. The numbers are right there in the axes.

    This lives in the generated preamble and not in the body, which is the
    whole point: it is emitted identically into both twins, so the
    byte-comparability that makes a twin difference mean anything survives. A
    record() call in the body would not.
    """
    import json as _json
    out, n = {}, 0
    for ai, ax in enumerate(_fig.get_axes()):
        for li, ln in enumerate(ax.get_lines()):
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs):
                out[f"ax{ai}.line{li}.x"] = [float(v) for v in xs]
                out[f"ax{ai}.line{li}.y"] = [float(v) for v in ys]
                n += 1
        for ii, im in enumerate(ax.get_images()):
            a = im.get_array()
            if a is not None and a.size:
                import numpy as _np
                a = _np.asarray(a, dtype=float)
                out[f"ax{ai}.image{ii}.shape"] = list(a.shape)
                out[f"ax{ai}.image{ii}.stats"] = [float(a.min()), float(a.max()),
                                                  float(a.mean()),
                                                  float((a ** 2).mean() ** 0.5)]
                n += 1
    if n:
        # PREFIXED WITH THE SCRIPT STEM, because the figure's name comes from
        # upstream's body and is therefore IDENTICAL on both twins. Without the
        # prefix both write Ez_angle0.plotdata.json into one directory and the
        # second run overwrites the first -- the same defect that made the two
        # mpb twins clobber each other's h5 output.
        _p = pathlib.Path(str(_path))
        _p = _p.with_name("mpb_strip_mp-" + _p.stem)
        _p.with_suffix(".plotdata.json").write_text(
            _json.dumps(out, indent=1, sort_keys=True))


import matplotlib.figure as _mplfig   # submodule, not an attribute of
                                     # matplotlib until imported
_savefig = _mplfig.Figure.savefig


def _savefig_and_dump(self, fname, *a, **k):
    r = _savefig(self, fname, *a, **k)
    try:
        _dump_plot_data(self, fname)
    except Exception:
        pass          # a figure we cannot read must not kill the tutorial
    return r


_mplfig.Figure.savefig = _savefig_and_dump
# --- end preamble ----------------------------------------------------------

import meep_torch.mp as mp
from meep_torch import mpb

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
ms.device = "cuda"
ms.filename_prefix = str(OUT / "mpb_strip_mp")


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
