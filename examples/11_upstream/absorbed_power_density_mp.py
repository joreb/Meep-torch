# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/absorbed_power_density.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/absorbed_power_density/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import matplotlib
import numpy as np

matplotlib.use("agg")
import matplotlib.pyplot as plt
from meep_torch.mp.materials import SiO2

import meep_torch.mp as mp

resolution = 100  # pixels/um

dpml = 1.0
pml_layers = [mp.PML(thickness=dpml)]

r = 1.0  # radius of cylinder
dair = 2.0  # air padding thickness

s = 2 * (dpml + dair + r)
cell_size = mp.Vector3(s, s)

wvl = 1.0
fcen = 1 / wvl

# is_integrated=True necessary for any planewave source extending into PML
sources = [
    mp.Source(
        mp.GaussianSource(fcen, fwidth=0.1 * fcen, is_integrated=True),
        center=mp.Vector3(-0.5 * s + dpml),
        size=mp.Vector3(0, s),
        component=mp.Ez,
    )
]

symmetries = [mp.Mirror(mp.Y)]

geometry = [mp.Cylinder(material=SiO2, center=mp.Vector3(), radius=r, height=mp.inf)]

sim = mp.Simulation(
    resolution=resolution,
    cell_size=cell_size,
    boundary_layers=pml_layers,
    sources=sources,
    k_point=mp.Vector3(),
    symmetries=symmetries,
    geometry=geometry,
)

dft_fields = sim.add_dft_fields(
    [mp.Dz, mp.Ez],
    fcen,
    0,
    1,
    center=mp.Vector3(),
    size=mp.Vector3(2 * r, 2 * r),
    yee_grid=True,
)

# closed box surrounding cylinder for computing total incoming flux
flux_box = sim.add_flux(
    fcen,
    0,
    1,
    mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2 * r), weight=+1),
    mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2 * r), weight=-1),
    mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2 * r, 0), weight=-1),
    mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2 * r, 0), weight=+1),
)

sim.run(until_after_sources=100)

Dz = sim.get_dft_array(dft_fields, mp.Dz, 0)
Ez = sim.get_dft_array(dft_fields, mp.Ez, 0)
absorbed_power_density = 2 * np.pi * fcen * np.imag(np.conj(Ez) * Dz)

dxy = 1 / resolution**2
absorbed_power = np.sum(absorbed_power_density) * dxy
absorbed_flux = mp.get_fluxes(flux_box)[0]
err = abs(absorbed_power - absorbed_flux) / absorbed_flux
print(
    f"flux:, {absorbed_power} (dft_fields), {absorbed_flux} (dft_flux), {err} (error)"
)


plt.figure()
sim.plot2D()
plt.savefig("power_density_cell.png", dpi=150, bbox_inches="tight")

plt.figure()
x = np.linspace(-r, r, Dz.shape[0])
y = np.linspace(-r, r, Dz.shape[1])
plt.pcolormesh(
    x,
    y,
    np.transpose(absorbed_power_density),
    cmap="inferno_r",
    shading="gouraud",
    vmin=0,
    vmax=np.amax(absorbed_power_density),
)
plt.xlabel("x (μm)")
plt.xticks(np.linspace(-r, r, 5))
plt.ylabel("y (μm)")
plt.yticks(np.linspace(-r, r, 5))
plt.gca().set_aspect("equal")
plt.title(
    "absorbed power density"
    + "\n"
    + "SiO2 Labs(λ={} μm) = {:.2f} μm".format(
        wvl, wvl / np.imag(np.sqrt(SiO2.epsilon(fcen)[0][0]))
    )
)
plt.colorbar()
plt.savefig("power_density_map.png", dpi=150, bbox_inches="tight")
