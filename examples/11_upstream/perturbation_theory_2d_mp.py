# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/perturbation_theory_2d.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/perturbation_theory_2d/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import argparse

import numpy as np

import meep_torch.mp as mp


def main(args):
    if args.perpendicular:
        src_cmpt = mp.Hz
        fcen = 0.21  # pulse center frequency
    else:
        src_cmpt = mp.Ez
        fcen = 0.17  # pulse center frequency

    n = 3.4  # index of waveguide
    w = 1  # ring width
    r = 1  # inner radius of ring
    pad = 4  # padding between waveguide and edge of PML
    dpml = 2  # thickness of PML

    pml_layers = [mp.PML(dpml)]

    sxy = 2 * (r + w + pad + dpml)
    cell_size = mp.Vector3(sxy, sxy)

    symmetries = [
        mp.Mirror(mp.X, phase=+1 if args.perpendicular else -1),
        mp.Mirror(mp.Y, phase=-1 if args.perpendicular else +1),
    ]

    geometry = [
        mp.Cylinder(
            material=mp.Medium(index=n),
            radius=r + w,
            height=mp.inf,
            center=mp.Vector3(),
        ),
        mp.Cylinder(material=mp.vacuum, radius=r, height=mp.inf, center=mp.Vector3()),
    ]

    # find resonant frequency of unperturbed geometry using broadband source

    df = 0.2 * fcen  # pulse width (in frequency)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + 0.1),
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(-(r + 0.1)),
            amplitude=-1,
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        symmetries=symmetries,
    )

    h = mp.Harminv(src_cmpt, mp.Vector3(r + 0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=100)

    frq_unperturbed = h.modes[0].freq

    sim.reset_meep()

    # unperturbed geometry with narrowband source centered at resonant frequency

    fcen = frq_unperturbed
    df = 0.05 * fcen

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + 0.1),
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(-(r + 0.1)),
            amplitude=-1,
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        symmetries=symmetries,
    )

    sim.run(until_after_sources=100)

    deps = 1 - n**2
    deps_inv = 1 - 1 / n**2

    if args.perpendicular:
        para_integral = (
            deps
            * 2
            * np.pi
            * (
                r * abs(sim.get_field_point(mp.Ey, mp.Vector3(r))) ** 2
                - (r + w) * abs(sim.get_field_point(mp.Ey, mp.Vector3(r + w))) ** 2
            )
        )
        perp_integral = (
            deps_inv
            * 2
            * np.pi
            * (
                -r * abs(sim.get_field_point(mp.Dy, mp.Vector3(y=r))) ** 2
                + (r + w) * abs(sim.get_field_point(mp.Dy, mp.Vector3(y=r + w))) ** 2
            )
        )
        numerator_integral = para_integral + perp_integral
    else:
        numerator_integral = (
            deps
            * 2
            * np.pi
            * (
                r * abs(sim.get_field_point(mp.Ez, mp.Vector3(r))) ** 2
                - (r + w) * abs(sim.get_field_point(mp.Ez, mp.Vector3(r + w))) ** 2
            )
        )

    denominator_integral = sim.electric_energy_in_box(
        center=mp.Vector3(), size=mp.Vector3(sxy - 2 * dpml, sxy - 2 * dpml)
    )
    perturb_theory_dw_dR = (
        -frq_unperturbed * numerator_integral / (8 * denominator_integral)
    )

    # perturbed geometry with narrowband source

    dr = 0.04

    sim.reset_meep()

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(r + dr + 0.1),
        ),
        mp.Source(
            mp.GaussianSource(fcen, fwidth=df),
            component=src_cmpt,
            center=mp.Vector3(-(r + dr + 0.1)),
            amplitude=-1,
        ),
    ]

    geometry = [
        mp.Cylinder(
            material=mp.Medium(index=n),
            radius=r + dr + w,
            height=mp.inf,
            center=mp.Vector3(),
        ),
        mp.Cylinder(
            material=mp.vacuum, radius=r + dr, height=mp.inf, center=mp.Vector3()
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        resolution=args.res,
        sources=sources,
        symmetries=symmetries,
    )

    h = mp.Harminv(src_cmpt, mp.Vector3(r + dr + 0.1), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=100)

    frq_perturbed = h.modes[0].freq

    finite_diff_dw_dR = (frq_perturbed - frq_unperturbed) / dr

    print(
        f"dwdR:, {perturb_theory_dw_dR} (pert. theory), {finite_diff_dw_dR} (finite diff.)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-perpendicular",
        action="store_true",
        help="use perpendicular field source (default: parallel field source)",
    )
    parser.add_argument(
        "-res", type=int, default=30, help="resolution (default: 30 pixels/um)"
    )
    args = parser.parse_args()
    main(args)
