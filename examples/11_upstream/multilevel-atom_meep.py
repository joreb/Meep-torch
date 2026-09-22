# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/multilevel-atom.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the generated ModeSolver lines, so that a difference between
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
OUT = pathlib.Path(__file__).parent / "output/multilevel-atom/"
OUT.mkdir(exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
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
        _p = _p.with_name("multilevel-atom_meep-" + _p.stem)
        _p.with_suffix(".plotdata.json").write_text(
            _json.dumps(out, indent=1, sort_keys=True))


import matplotlib.figure as _mplfig   # submodule, not an attribute of
                                     # matplotlib until imported
_savefig = _mplfig.Figure.savefig


def _savefig_and_dump(self, fname, *a, **k):
    # SUFFIXED WITH THE TWIN TAG, for the same reason the plotdata is prefixed
    # with the stem: the figure's name comes from upstream's body and is
    # identical on both twins, so `_mp` and `_meep` wrote the same PNG into one
    # directory and whichever ran second won.
    _p = pathlib.Path(str(fname))
    r = _savefig(self, _p.with_name(_p.stem + "_meep" + _p.suffix), *a, **k)
    try:
        # the ORIGINAL name: the plotdata already carries the "<stem>-" prefix
        _dump_plot_data(self, fname)
    except Exception:
        pass          # a figure we cannot read must not kill the tutorial
    return r


_mplfig.Figure.savefig = _savefig_and_dump


# plt.show() IS A FIGURE TOO. Several tutorials build a figure and only call
# show(): upstream is interactive, the harness is headless, and under Agg
# show() is a no-op, so the figure -- and the numbers in it -- vanished. Those
# scripts scored "no comparable output" in the parity sweep while actually
# having plenty to compare. Route show() through the same savefig hook so they
# emit a PNG and a .plotdata.json like everything else.
import matplotlib.pyplot as _plt

_pltshow = _plt.show
_shown = set()


def _show_and_dump(*a, **k):
    for _i, _num in enumerate(_plt.get_fignums()):
        _f = _plt.figure(_num)
        if id(_f) in _shown:
            continue      # already written by an explicit savefig
        _shown.add(id(_f))
        try:
            _savefig_and_dump(_f, f"figure{_i}.png")
        except Exception:
            pass          # a figure we cannot write must not kill the tutorial
    return None


_plt.show = _show_and_dump
# --- end preamble ----------------------------------------------------------

import math

import meep as mp

# This file realizes a 1D, one-sided Fabry-Perot laser, as described in Fig. 2 of Optics Express, Vol. 20, pp. 474-88, 2012.

# Cavity definitions
resolution = 400
ncav = 1.5  # cavity refractive index
Lcav = 1  # cavity length
dpad = 1  # padding thickness
dpml = 1  # PML thickness
sz = Lcav + dpad + dpml
cell_size = mp.Vector3(z=sz)
dimensions = 1
pml_layers = [mp.PML(dpml, side=mp.High)]

# For defining laser properties in MEEP, the transition rates / frequencies are specified in units of 2*pi*a/c.
# gamma_21 in MEEP is the Full-Width Half-Max, as opposed to gamma_perp, which is the HWHM in SALT.
# Additionally, the classical coupling element sigma = 2*theta^2*omega_a/hbar, where
# theta is the off-diagonal dipole matrix element.

# These different conventions can cause a bit of confusion when comparing against SALT, so here we perform
# this transformation explicitly.

omega_a = 40  # omega_a in SALT
freq_21 = omega_a / (2 * math.pi)  # emission frequency  (units of 2πc/a)

gamma_perp = 4  # HWHM in angular frequency, SALT
gamma_21 = (2 * gamma_perp) / (
    2 * math.pi
)  # FWHM emission linewidth in sec^-1 (units of 2πc/a)
# Note that 2*pi*gamma_21 = 2*gamma_perp in SALT.

theta = 1  # theta, the off-diagonal dipole matrix element, in SALT
sigma_21 = 2 * theta * theta * omega_a  # dipole coupling strength (hbar = 1)

# The gain medium in MEEP is allowed to have an arbitrary number of levels, and is not
# restricted to a two-level gain medium, as it simulates the populations of every individual
# atomic energy level.

# If you are using a 2 level gain model, you can compare against
# results which only simulate the atomic inversion, using the definitions
# gamma_parallel = pumping_rate + rate_21
# D_0 = (pumping_rate - rate_21)/(pumping_rate + rate_21) * N0

# In fact, even if you arn't using a 2 level gain model, you can compare against an effective
# two level model using the formula provided in Cerjan et al., Opt. Express 20, 474 (2012).

# Here, D_0 as written above is not yet in "SALT" units. To make this conversion,
# D_0 (SALT) = theta^2/(hbar*gamma_perp) * D_0 (as written above)

# Finally, note the lack of 4*pi in the above conversion that is written in many published SALT papers.
# This 4*pi comes from using Gaussian units, in which the displacement field, D = E + 4*pi*P, whereas
# in SI units, D = eps0*E + P, which is what MEEP uses.

# Gain medium pump and decay rates are specified in units of c/a.

rate_21 = 0.005  # non-radiative rate  (units of c/a)
N0 = 37  # initial population density of ground state
Rp = 0.0051  # pumping rate of ground to excited state
# so for example, these parameters have D_0 (SALT) = 0.0693.

# Make the actual medium in MEEP:
transitions = [
    mp.Transition(
        1,
        2,
        pumping_rate=Rp,
        frequency=freq_21,
        gamma=gamma_21,
        sigma_diag=mp.Vector3(sigma_21, 0, 0),
    ),
    mp.Transition(2, 1, transition_rate=rate_21),
]
ml_atom = mp.MultilevelAtom(sigma=1, transitions=transitions, initial_populations=[N0])
two_level = mp.Medium(index=ncav, E_susceptibilities=[ml_atom])

# Specify the cavity geometry:
geometry = [
    mp.Block(
        center=mp.Vector3(z=-0.5 * sz + 0.5 * Lcav),
        size=mp.Vector3(mp.inf, mp.inf, Lcav),
        material=two_level,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=resolution,
    boundary_layers=pml_layers,
    geometry=geometry,
    dimensions=dimensions,
)

sim.init_sim()


def field_func(p):
    return 1 if p.z == -0.5 * sz + 0.5 * Lcav else 0


sim.fields.initialize_field(mp.Ex, field_func)

# Specify the end time:
endt = 7000
# Note that the total number of time steps run is endt*resolution*2. This is the origin of the extra
# factor of 2 in the definition of dt in fieldfft_meep.m.


def print_field(sim):
    fp = sim.get_field_point(
        mp.Ex, mp.Vector3(z=(-0.5 * sz) + Lcav + (0.5 * dpad))
    ).real
    print(f"field:, {sim.meep_time()}, {fp}")


sim.run(mp.after_time(endt - 250, print_field), until=endt)
