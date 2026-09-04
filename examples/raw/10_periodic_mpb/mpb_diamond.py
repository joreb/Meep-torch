# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# %% [markdown]
# # Diamond-lattice spheres: 3D band diagram
#
# Dielectric spheres in a diamond (fcc) lattice, solved with the
# triply-periodic 3D Bloch stepper + harminv — the 2D Bloch-track method
# lifted to 3D. The fcc primitive cell is mapped to the conventional cubic
# cell (8 atoms), and the meep.mpb oracle uses the same supercell so the
# comparison is self-consistent.

# %%
"""meep-torch port of meep/python/examples/mpb_diamond.py.

Band diagram of dielectric spheres in a diamond (fcc) lattice: the mpb
original solves the fcc primitive cell with meep.mpb. meep-torch has no
MPB solver -- instead the triply-periodic 3D Bloch stepper
(stepper_bloch3, n_abs_z=0 + kz) propagates a broadband pulse and
harminv extracts omega_n(k), exactly the 2D Bloch-track method lifted
to 3D.

Rect-supercell trick (documented in test_mpb_track.test_diamond_band_vs_mpb):
the fcc primitive cell is mapped to the conventional cubic cell of side 1
(8 atoms, r=0.25/sqrt(2), eps=11.56). The cubic BZ folds the fcc bands;
the meep.mpb oracle uses the SAME supercell so the gate is self-consistent.
Here we sweep Gamma -> X (k=(0,t*pi,t*pi)); the mpb cross is the oracle's
first band at X.

Run: uv run python examples/10_periodic_mpb/mpb_diamond.py
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[3] / "tests" / "examples"))
from test_mpb_track import diamond_bands_at, DATA

OUT = pathlib.Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## Gamma -> X sweep
#
# Six k-points along k = (0, t pi, t pi); `diamond_bands_at` (shared with the
# test gate) runs the 3D ringdown per k, and the mpb crosses mark the
# oracle's first bands at X.

# %%
# Gamma -> simple-cubic X: k = (0, t*pi, t*pi), |k| = t*pi*sqrt(2).
ts = np.linspace(0.0, 1.0, 6)
fig, ax = plt.subplots(figsize=(5.8, 4.6))
for t in ts:
    kx, ky, kz = 0.0, t * math.pi, t * math.pi
    kmag = math.sqrt(ky * ky + kz * kz) / (2 * math.pi)
    for f in diamond_bands_at(kx, ky, kz):
        if 0.05 < f <= 0.7:
            ax.plot(kmag, f, "b.", ms=4)

ref = json.load(open(DATA / "diamond_mpb.json"))
kx_x = math.sqrt(math.pi ** 2 + math.pi ** 2) / (2 * math.pi)   # |X|/(2pi)
for f in ref["bands"][0][:3]:
    ax.plot(kx_x, f, "rx", ms=7, mew=1.5)

ax.set_xlabel("$|k|$ ($2\\pi/a$)")
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("diamond lattice bands (3D Bloch + harminv)")
ax.set_xlim(0, kx_x * 1.02)
ax.set_ylim(0, 0.7)
ax.plot([0, kx_x], [0, kx_x * 2 * math.pi / (2 * math.pi)], "k--", lw=0.8,
        label="light line")
ax.plot([], [], "rx", label="meep.mpb")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "mpb_diamond.png", dpi=150)
plt.close(fig)
print(f"mpb_diamond.png written (mpb X-band {ref['bands'][0][0]:.4f} marked)")