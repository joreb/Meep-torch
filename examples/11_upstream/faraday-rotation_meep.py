# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/faraday-rotation.py by scripts/port_upstream_example.py.
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
OUT = pathlib.Path(__file__).parent / "output/faraday-rotation/"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

# From the Meep tutorial: plotting Faraday rotation of a linearly polarized plane wave
import meep as mp

## Parameters for a gyrotropic Lorentzian medium
epsn = 1.5  # background permittivity
f0 = 1.0  # natural frequency
gamma = 1e-6  # damping rate
sn = 0.1  # sigma parameter
b0 = 0.15  # magnitude of bias vector

susc = [
    mp.GyrotropicLorentzianSusceptibility(
        frequency=f0, gamma=gamma, sigma=sn, bias=mp.Vector3(0, 0, b0)
    )
]
mat = mp.Medium(epsilon=epsn, mu=1, E_susceptibilities=susc)

## Set up and run the Meep simulation:
tmax = 100
L = 20.0
cell = mp.Vector3(0, 0, L)
fsrc, src_z = 0.8, -8.5
pml_layers = [mp.PML(thickness=1.0, direction=mp.Z)]

sources = [
    mp.Source(
        mp.ContinuousSource(frequency=fsrc),
        component=mp.Ex,
        center=mp.Vector3(0, 0, src_z),
    )
]

sim = mp.Simulation(
    cell_size=cell,
    geometry=[],
    sources=sources,
    boundary_layers=pml_layers,
    default_material=mat,
    resolution=50,
)
sim.run(until=tmax)

import matplotlib.pyplot as plt

## Plot results:
import numpy as np

ex_data = sim.get_efield_x().real
ey_data = sim.get_efield_y().real

z = np.linspace(-L / 2, L / 2, len(ex_data))
plt.figure(1)
plt.plot(z, ex_data, label="Ex")
plt.plot(z, ey_data, label="Ey")
plt.xlim(-L / 2, L / 2)
plt.xlabel("z")
plt.legend()

## Comparison with analytic result:
dfsq = f0**2 - 1j * fsrc * gamma - fsrc**2
eperp = epsn + sn * f0**2 * dfsq / (dfsq**2 - (fsrc * b0) ** 2)
eta = sn * f0**2 * fsrc * b0 / (dfsq**2 - (fsrc * b0) ** 2)

k_gyro = 2 * np.pi * fsrc * np.sqrt(0.5 * (eperp - np.sqrt(eperp**2 - eta**2)))
Ex_theory = 0.37 * np.cos(k_gyro * (z - src_z)).real
Ey_theory = 0.37 * np.sin(k_gyro * (z - src_z)).real

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(z, ex_data, label="Ex (MEEP)")
plt.plot(z, Ex_theory, "k--")
plt.plot(z, -Ex_theory, "k--", label="Ex envelope (theory)")
plt.xlim(-L / 2, L / 2)
plt.xlabel("z")
plt.legend(loc="lower right")

plt.subplot(2, 1, 2)
plt.plot(z, ey_data, label="Ey (MEEP)")
plt.plot(z, Ey_theory, "k--")
plt.plot(z, -Ey_theory, "k--", label="Ey envelope (theory)")
plt.xlim(-L / 2, L / 2)
plt.xlabel("z")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()
