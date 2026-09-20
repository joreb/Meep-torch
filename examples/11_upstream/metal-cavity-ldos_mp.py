# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/metal-cavity-ldos.py by scripts/port_upstream_example.py.
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
for _old, _new in (("trapz", "trapezoid"), ("complex_", "complex128"),
                   ("float_", "float64"), ("alltrue", "all"),
                   ("sometrue", "any"), ("product", "prod"),
                   ("cumproduct", "cumprod"), ("round_", "round")):
    if not hasattr(_np, _old) and hasattr(_np, _new):
        setattr(_np, _old, getattr(_np, _new))   # upstream predates numpy 2.0
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/metal-cavity-ldos/"
OUT.mkdir(parents=True, exist_ok=True)
import torch  # noqa: E402  (device probe only)
DEVICE = os.environ.get("MEEP_TORCH_DEVICE") or (
    "cuda" if torch.cuda.is_available() else "cpu")
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import math

import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp


def metal_cavity(w):
    resolution = 50
    sxy = 2
    dpml = 1
    sxy += 2 * dpml
    cell = mp.Vector3(sxy, sxy)

    pml_layers = [mp.PML(dpml)]
    a = 1
    t = 0.1
    geometry = [
        mp.Block(mp.Vector3(a + 2 * t, a + 2 * t, mp.inf), material=mp.metal),
        mp.Block(mp.Vector3(a, a, mp.inf), material=mp.air),
    ]

    geometry.append(
        mp.Block(
            center=mp.Vector3(a / 2), size=mp.Vector3(2 * t, w, mp.inf), material=mp.air
        )
    )

    fcen = math.sqrt(0.5) / a
    df = 0.2
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df), component=mp.Ez, center=mp.Vector3()
        )
    ]

    symmetries = [mp.Mirror(mp.Y)]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        boundary_layers=pml_layers,
        sources=sources,
        symmetries=symmetries,
        resolution=resolution,
    )

    h = mp.Harminv(mp.Ez, mp.Vector3(), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=500)

    m = h.modes[0]
    f = m.freq
    Q = m.Q
    Vmode = 0.25 * a * a
    ldos_1 = Q / Vmode / (2 * math.pi * f * math.pi * 0.5)

    sim.reset_meep()

    T = 2 * Q * (1 / f)
    sim.run(mp.dft_ldos(f, 0, 1), until_after_sources=T)
    ldos_2 = sim.ldos_data[0]

    return ldos_1, ldos_2


ws = np.arange(0.2, 0.5, 0.1)
ldos_1 = np.zeros(len(ws))
ldos_2 = np.zeros(len(ws))

for j in range(len(ws)):
    ldos_1[j], ldos_2[j] = metal_cavity(ws[j])
    print(f"ldos:, {ldos_1[j]}, {ldos_2[2]}")

plt.figure(dpi=150)
plt.semilogy(1 / ws, ldos_1, "bo-", label="2Q/(πωV)")
plt.semilogy(1 / ws, ldos_2, "rs-", label="LDOS")
plt.xlabel("a/w")
plt.ylabel("2Q/(πωW) or LDOS")
plt.show()
