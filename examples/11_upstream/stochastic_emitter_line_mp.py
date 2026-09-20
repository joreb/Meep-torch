# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/stochastic_emitter_line.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/stochastic_emitter_line/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import argparse

import numpy as np
from meep_torch.mp.materials import Ag

import meep_torch.mp as mp

parser = argparse.ArgumentParser()
parser.add_argument("-res", type=int, default=50, help="resolution (pixels/um)")
parser.add_argument("-nf", type=int, default=500, help="number of frequencies")
parser.add_argument(
    "-nsrc",
    type=int,
    default=15,
    help="number of line sources with cosine Fourier series amplitude function (method 3)",
)
parser.add_argument(
    "-textured",
    action="store_true",
    default=False,
    help="flat (default) or textured surface",
)
parser.add_argument(
    "-method",
    type=int,
    choices=[2, 3],
    default=2,
    help="type of method: (2) single dipole with 1 run per dipole or (3) line source with cosine Fourier series amplitude function",
)
args = parser.parse_args()

resolution = args.res

dpml = 1.0
dair = 0.9
hrod = 0.6
wrod = 0.8
dsub = 5.4
dAg = 0.4

sx = 1.5
sy = dpml + dair + hrod + dsub + dAg

cell_size = mp.Vector3(sx, sy)

pml_layers = [mp.PML(direction=mp.Y, thickness=dpml, side=mp.High)]

fcen = 1.0
df = 0.2
nfreq = args.nf
nsrc = args.nsrc
ndipole = int(sx * resolution)
run_time = 2 * nfreq / df

geometry = [
    mp.Block(
        material=mp.Medium(index=3.45),
        center=mp.Vector3(0, 0.5 * sy - dpml - dair - hrod - 0.5 * dsub),
        size=mp.Vector3(mp.inf, dsub, mp.inf),
    ),
    mp.Block(
        material=Ag,
        center=mp.Vector3(0, -0.5 * sy + 0.5 * dAg),
        size=mp.Vector3(mp.inf, dAg, mp.inf),
    ),
]

if args.textured:
    geometry.append(
        mp.Block(
            material=mp.Medium(index=3.45),
            center=mp.Vector3(0, 0.5 * sy - dpml - dair - 0.5 * hrod),
            size=mp.Vector3(wrod, hrod, mp.inf),
        )
    )


def src_amp_func(n):
    def _src_amp_func(p):
        if n == 0:
            return 1 / np.sqrt(sx)
        else:
            return np.sqrt(2 / sx) * np.cos(n * np.pi * (p.x + 0.5 * sx) / sx)

    return _src_amp_func


def compute_flux(m, n):
    if m == 2:
        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=mp.Vector3(
                    sx * (-0.5 + n / ndipole), -0.5 * sy + dAg + 0.5 * dsub
                ),
            )
        ]
    else:
        sources = [
            mp.Source(
                mp.GaussianSource(fcen, fwidth=df),
                component=mp.Ez,
                center=mp.Vector3(0, -0.5 * sy + dAg + 0.5 * dsub),
                size=mp.Vector3(sx, 0),
                amp_func=src_amp_func(n),
            )
        ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        k_point=mp.Vector3(),
        boundary_layers=pml_layers,
        geometry=geometry,
        sources=sources,
    )

    flux_mon = sim.add_flux(
        fcen,
        df,
        nfreq,
        mp.FluxRegion(center=mp.Vector3(0, 0.5 * sy - dpml), size=mp.Vector3(sx)),
    )

    sim.run(until=run_time)

    flux = mp.get_fluxes(flux_mon)
    freqs = mp.get_flux_freqs(flux_mon)

    return freqs, flux


if args.method == 2:
    fluxes = np.zeros((nfreq, ndipole))
    for d in range(ndipole):
        freqs, fluxes[:, d] = compute_flux(2, d)
else:
    fluxes = np.zeros((nfreq, nsrc))
    for d in range(nsrc):
        freqs, fluxes[:, d] = compute_flux(3, d)


if mp.am_master():
    with open(
        f'method{args.method}_{"textured" if args.textured else "flat"}_res{resolution}_nfreq{nfreq}_{"ndipole" if args.method == 2 else "nsrc"}{ndipole if args.method == 2 else nsrc}.npz',
        "wb",
    ) as f:
        np.savez(f, freqs=freqs, fluxes=fluxes)
