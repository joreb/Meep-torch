# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Triangular rod lattice: TM bands at Gamma, M, K (vs mpb)
#
# TM bands of the triangular rod lattice (r = 0.2, eps = 12) computed on a
# rectangular sqrt(3) x 1 supercell with the doubly-periodic Bloch stepper.
# Supercell folding adds extra (legitimate) folded bands between the mpb
# reference points.

# %%
"""meep-torch port of meep/python/examples/mpb_tri_rods.py.

TM bands of the triangular rod lattice (r=0.2, eps=12) at Gamma/M/K via
the rectangular sqrt(3) x 1 supercell + doubly-periodic Bloch stepper +
harminv, overlaid with meep.mpb (tri_rods_mpb.json).

SUBSTITUTIONS: mpb.ModeSolver -> step_bloch2_tm; supercell folding adds
extra (legitimate) folded bands between the mpb points.

Run: uv run python examples/10_periodic_mpb/mpb_tri_rods.py
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "tests" / "examples"))
from test_mpb_track import _bloch2_bands, _tri_eps

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
ref = json.load(open(pathlib.Path(__file__).parents[3] / "tests" / "examples" / "data" / "tri_rods_mpb.json"))
# %% [markdown]
# ## Three high-symmetry k-points
#
# `_tri_eps` builds the supercell eps; each of Gamma, M, K is one Bloch
# ringdown whose harminv frequencies are plotted next to the mpb crosses.

# %%
eps, dx = _tri_eps(res=26, r=ref["r"], eps_rod=ref["eps"])
kpts = {"$\\Gamma$": (0.0, 0.0),
        "M": (math.pi / math.sqrt(3), math.pi),
        "K": (0.0, 2 * math.pi / 3)}
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for i, (name, (kx, ky)) in enumerate(kpts.items()):
    for f in _bloch2_bands(eps, kx, ky, 0.4 * dx, dx, 0.85, t_end=350.0):
        ax.plot(i, f, "b.", ms=5)
    for f in ref[name.strip("$\\Gamma") or "gamma"] if False else \
            ref[{"$\\Gamma$": "gamma", "M": "M", "K": "K"}[name]]:
        if f > 1e-3:
            ax.plot(i, f, "rx", ms=8, mew=1.5)
ax.set_xticks(range(3), list(kpts))
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("tri rods TM bands (dots: meep-torch supercell, x: mpb)")
ax.set_ylim(0, 0.9)
fig.tight_layout()
fig.savefig(OUT / "mpb_tri_rods.png", dpi=150)
plt.close(fig)
print("mpb_tri_rods.png written (TM gap visible between bands 1 and 2)")
