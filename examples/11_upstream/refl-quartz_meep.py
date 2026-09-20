# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/refl-quartz.py by scripts/port_upstream_example.py.
# The body below is upstream's, unchanged but for the generated ModeSolver lines, so that a difference between
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
OUT = pathlib.Path(__file__).parent / "output/refl-quartz/"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import math

import matplotlib.pyplot as plt
import numpy as np
from meep.materials import fused_quartz

import meep as mp

resolution = 200  # pixels/μm

dpml = 1.0
sz = 10 + 2 * dpml
cell_size = mp.Vector3(z=sz)
pml_layers = [mp.PML(dpml)]

wvl_min = 0.4
wvl_max = 0.8
fmin = 1 / wvl_max
fmax = 1 / wvl_min
fcen = 0.5 * (fmax + fmin)
df = fmax - fmin
nfreq = 50

sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=df),
        component=mp.Ex,
        center=mp.Vector3(z=-0.5 * sz + dpml),
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    dimensions=1,
    resolution=resolution,
)

refl_fr = mp.FluxRegion(center=mp.Vector3(z=-0.25 * sz))
refl = sim.add_flux(fcen, df, nfreq, refl_fr)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-9))

empty_flux = mp.get_fluxes(refl)
empty_data = sim.get_flux_data(refl)
sim.reset_meep()

geometry = [
    mp.Block(
        mp.Vector3(mp.inf, mp.inf, 0.5 * sz),
        center=mp.Vector3(z=0.25 * sz),
        material=fused_quartz,
    )
]

sim = mp.Simulation(
    cell_size=cell_size,
    boundary_layers=pml_layers,
    geometry=geometry,
    sources=sources,
    dimensions=1,
    resolution=resolution,
)

refl = sim.add_flux(fcen, df, nfreq, refl_fr)
sim.load_minus_flux_data(refl, empty_data)

sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ex, mp.Vector3(), 1e-9))

refl_flux = mp.get_fluxes(refl)
R_meep = -1 * np.divide(refl_flux, empty_flux)

freqs = mp.get_flux_freqs(refl)
wvls = np.divide(1, freqs)

eps_quartz = (
    lambda l: 1
    + 0.6961663 * math.pow(l, 2) / (pow(l, 2) - pow(0.0684043, 2))
    + 0.4079426 * pow(l, 2) / (pow(l, 2) - pow(0.1162414, 2))
    + 0.8974794 * pow(l, 2) / (pow(l, 2) - pow(9.896161, 2))
)
R_fresnel = lambda l: math.pow(
    math.fabs(1 - math.sqrt(eps_quartz(l))) / (1 + math.sqrt(eps_quartz(l))), 2
)
R_analytic = [R_fresnel(i) for i in wvls]

plt.figure()
plt.plot(wvls, R_meep, "bo-", label="meep")
plt.plot(wvls, R_analytic, "rs-", label="analytic")
plt.xlabel("wavelength (μm)")
plt.ylabel("reflectance")
plt.axis([0.4, 0.8, 0.0340, 0.0365])
plt.xticks(list(np.arange(0.4, 0.9, 0.1)))
plt.legend(loc="upper right")
plt.show()
