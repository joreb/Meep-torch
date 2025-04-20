# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/mpb_honey_rods.py.

Honeycomb rod lattice (r=0.14, eps=12) TM bands at Gamma via the
rectangular supercell (4-rod basis) + doubly-periodic Bloch + harminv,
vs meep.mpb crosses. Degenerate pairs may appear once (point-source
capture); the band-1 gate lives in tests/examples/test_mpb_track.py.

Run: uv run python examples/mpb_honey_rods.py
"""
import json
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tests" / "examples"))
from test_mpb_track import _bloch2_bands, _tri_eps

OUT = pathlib.Path(__file__).parent
ref = json.load(open(OUT.parent / "tests" / "examples" / "data" / "honey_rods_mpb.json"))
eps, dx = _tri_eps(res=26, r=ref["r"], eps_rod=ref["eps"],
                   basis_shift=(math.sqrt(3) / 6, 0.0))
got = _bloch2_bands(eps, 0.0, 0.0, 0.4 * dx, dx, 0.85, t_end=350.0)
fig, ax = plt.subplots(figsize=(4.2, 4.2))
for f in got:
    ax.plot(0, f, "b.", ms=6)
for f in ref["gamma"]:
    if f > 1e-3:
        ax.plot(0, f, "rx", ms=9, mew=1.5)
ax.set_xticks([0], ["$\\Gamma$"])
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("honeycomb rods TM at Gamma (dots: ours, x: mpb)")
fig.tight_layout()
fig.savefig(OUT / "mpb_honey_rods.png", dpi=150)
plt.close(fig)
print("mpb_honey_rods.png  bands:", [round(f, 4) for f in got[:6]])
