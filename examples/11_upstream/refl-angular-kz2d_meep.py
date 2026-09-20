# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/refl-angular-kz2d.py by scripts/port_upstream_example.py.
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
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/refl-angular-kz2d/"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import math

import meep as mp


def refl_planar(theta, kz_2d):
    resolution = 100

    dpml = 1.0
    sx = 10
    sx = 10 + 2 * dpml
    cell_size = mp.Vector3(sx)
    pml_layers = [mp.PML(dpml)]

    fcen = 1.0

    # plane of incidence is XZ
    k = mp.Vector3(z=math.sin(theta)).scale(fcen)

    sources = [
        mp.Source(
            mp.GaussianSource(fcen, fwidth=0.2 * fcen),
            component=mp.Ey,
            center=mp.Vector3(-0.5 * sx + dpml),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        kz_2d=kz_2d,
        resolution=resolution,
    )

    refl_fr = mp.FluxRegion(center=mp.Vector3(-0.25 * sx))
    refl = sim.add_flux(fcen, 0, 1, refl_fr)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ey, mp.Vector3(-0.5 * sx + dpml), 1e-9
        )
    )

    input_flux = mp.get_fluxes(refl)
    input_data = sim.get_flux_data(refl)
    sim.reset_meep()

    # add a block with n=3.5 for the air-dielectric interface
    geometry = [
        mp.Block(
            size=mp.Vector3(0.5 * sx, mp.inf, mp.inf),
            center=mp.Vector3(0.25 * sx),
            material=mp.Medium(index=3.5),
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k,
        kz_2d=kz_2d,
        resolution=resolution,
    )

    refl = sim.add_flux(fcen, 0, 1, refl_fr)
    sim.load_minus_flux_data(refl, input_data)

    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(
            50, mp.Ey, mp.Vector3(-0.5 * sx + dpml), 1e-9
        )
    )

    refl_flux = mp.get_fluxes(refl)
    freqs = mp.get_flux_freqs(refl)

    return -refl_flux[0] / input_flux[0]


# rotation angle of source: CCW around Y axis, 0 degrees along +X axis
theta_r = math.radians(19.4)

Rmeep_real_imag = refl_planar(theta_r, "real/imag")
Rmeep_complex = refl_planar(theta_r, "complex")
Rmeep_3d = refl_planar(theta_r, "3d")

n1 = 1
n2 = 3.5

# compute angle of refracted planewave in medium n2
# for incident planewave in medium n1 at angle theta_in
theta_out = lambda theta_in: math.asin(n1 * math.sin(theta_in) / n2)

# compute Fresnel reflectance for S-polarization in medium n2
# for incident planewave in medium n1 at angle theta_in
Rfresnel = (
    lambda theta_in: math.fabs(
        (n2 * math.cos(theta_out(theta_in)) - n1 * math.cos(theta_in))
        / (n2 * math.cos(theta_out(theta_in)) + n1 * math.cos(theta_in))
    )
    ** 2
)

print(
    f"refl:, {Rmeep_real_imag} (real/imag), {Rmeep_complex} (complex), {Rmeep_3d} (3d), {Rfresnel(theta_r)} (analytic)"
)
