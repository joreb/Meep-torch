# --- meep-torch example harness preamble (generated; do not hand-edit) -----
# Ported from examples_meep/examples/cavity_arrayslice.py by scripts/port_upstream_example.py.
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

matplotlib.use("Agg")                # rule 1: headless, before any pyplot import
# rule 2. parents=True because output/ itself may not exist yet -- plain
# exist_ok=True raises FileNotFoundError on the first run of a fresh clone.
OUT = pathlib.Path(__file__).parent / "output/cavity_arrayslice/"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "meep-cpu"       # real meep: CPU only, no device concept
os.chdir(OUT)         # every relative write upstream makes lands in output/<name>/
# --- end preamble ----------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np

import meep as mp

# set up the geometry
eps = 13
w = 1.2
r = 0.36
d = 1.4
N = 3
sy = 6
pad = 2
dpml = 1
sx = (2 * (pad + dpml + N)) + d - 1
fcen = 0.25
df = 0.2
nfreq = 500

cell = mp.Vector3(sx, sy, 0)

blk = mp.Block(size=mp.Vector3(mp.inf, w, mp.inf), material=mp.Medium(epsilon=eps))

geometry = [blk]

geometry.extend(mp.Cylinder(r, center=mp.Vector3(d / 2 + i)) for i in range(3))
geometry.extend(mp.Cylinder(r, center=mp.Vector3(d / -2 - i)) for i in range(3))

sim = mp.Simulation(
    cell_size=cell,
    geometry=geometry,
    sources=[],
    boundary_layers=[mp.PML(dpml)],
    resolution=20,
)

# add sources
sim.sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), mp.Hz, mp.Vector3())]

# run until sources are finished (and no later)
sim._run_sources_until(0, [])

# get 1D and 2D array slices
xMin = -0.25 * sx
xMax = +0.25 * sx
yMin = -0.15 * sy
yMax = +0.15 * sy

# 1D slice of Hz data
size_1d = mp.Vector3(xMax - xMin)
center_1d = mp.Vector3((xMin + xMax) / 2)
slice1d = sim.get_array(mp.Volume(center_1d, size=size_1d), component=mp.Hz)

# 2D slice of Hz data
size_2d = mp.Vector3(xMax - xMin, yMax - yMin)
center_2d = mp.Vector3((xMin + xMax) / 2, (yMin + yMax) / 2)
slice2d = sim.get_array(mp.Volume(center_2d, size=size_2d), component=mp.Hz)

# plot 1D slice
plt.subplot(1, 2, 1)
x1d = np.linspace(xMin, xMax, len(slice1d))
plt.plot(x1d, slice1d)

# plot 2D slice
plt.subplot(1, 2, 2)
dy = (yMax - yMin) / slice2d.shape[1]
dx = (xMax - xMin) / slice2d.shape[0]
(x2d, y2d) = np.mgrid[slice(xMin, xMax, dx), slice(yMin, yMax, dy)]
plt.contourf(x2d, y2d, slice2d)
plt.colorbar()
plt.show()
