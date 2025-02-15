# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/holey-wvg-bands.py.

Band diagram of the holey waveguide (Fan et al. 1995): Bloch-periodic
unit cell per k_x, broadband pulse, harminv -> omega_n(k). TE (meep
Hz-pol), axes swapped onto the meep-torch Bloch stepper (periodic y,
CPML x). Light line and the guided/folded bands are visible.

SUBSTITUTIONS: mp k_point sweep + mp.Harminv -> step_bloch_te + W3 FDM
harminv, 3 merged probes; resolution 10 (meep's 20); supersampled
coverage eps (reproduces meep subpixel).

Run: uv run python examples/holey-wvg-bands.py
"""
import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "tests" / "examples"))
from test_mpb_track import holey_bands_at

OUT = pathlib.Path(__file__).parent

ks = np.arange(0.05, 0.51, 0.05)
fig, ax = plt.subplots(figsize=(5.8, 4.6))
for k in ks:
    for f in holey_bands_at(round(float(k), 2)):
        if f <= 0.6:
            ax.plot(k, f, "b.", ms=4)
ax.plot([0, 0.5], [0, 0.5], "k--", lw=0.8, label="light line")
ax.set_xlabel("$k_x$ ($2\\pi/a$)")
ax.set_ylabel("frequency ($c/a$)")
ax.set_title("holey waveguide bands (Bloch + harminv)")
ax.set_xlim(0, 0.5)
ax.set_ylim(0, 0.6)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "holey-wvg-bands.png", dpi=150)
plt.close(fig)
print("holey-wvg-bands.png written (guided branch below the light line)")
