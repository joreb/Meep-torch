# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep-torch port of meep/python/examples/metal-cavity-ldos.py.

Metal cavity with an aperture: LDOS spectrum at the cavity center vs the
harminv f/Q of the same cavity, swept over aperture width w.

SUBSTITUTIONS:
  * mp.metal (PEC) -> Medium(D_conductivity=1e4) (skin depth << cell).
    Half-cell wall-position offset vs meep's PEC (~dx/a) documented in
    tests/examples/test_w6_ldos.py.
  * mp.Harminv -> modes.harminv on the center-probe ringdown (W3 FDM).
  * No Mirror symmetry (full cell).

Run: uv run python examples/metal-cavity-ldos.py
"""
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import meep_torch.mp as mp
from meep_torch.modes.harminv import harminv

OUT = pathlib.Path(__file__).parent
FCEN = math.sqrt(0.5)
METAL = mp.Medium(epsilon=1.0, D_conductivity=1e4)


def make_sim(w):
    a, t = 1.0, 0.1
    return mp.Simulation(cell_size=mp.Vector3(4, 4, 0), resolution=25,
        geometry=[
            mp.Block(size=mp.Vector3(a + 2 * t, a + 2 * t, mp.inf),
                     center=mp.Vector3(), material=METAL),
            mp.Block(size=mp.Vector3(a, a, mp.inf), center=mp.Vector3(),
                     material=mp.Medium(epsilon=1.0)),
            mp.Block(size=mp.Vector3(2 * t, w, mp.inf), center=mp.Vector3(a / 2),
                     material=mp.Medium(epsilon=1.0)),
        ],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=0.2), component=mp.Ez,
                           center=mp.Vector3())],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))


def cavity_ldos(w):
    sim = make_sim(w)
    name = sim.dft_ldos(FCEN, 0.2, 3)             # records center Ez series
    sim.run(until=400)
    acc, _, _ = sim._last_acc[name]
    tail = acc.detach().numpy()[int(round(60.0 / sim.grid.dt)):]
    modes = [m for m in harminv(tail, sim.grid.dt, 0.55, 0.8) if m.amp > 1e-6]
    m0 = max(modes, key=lambda x: x.amp)
    f1, q1 = m0.freq, m0.Q

    sim = make_sim(w)
    df_l = max(6 * f1 / q1, 0.01)
    name = sim.dft_ldos(f1, df_l, 41)
    sim.run(until=min(2 * q1 / f1 + 60.0, 2000.0))
    freqs, ld = sim.get_ldos(name)
    return f1, q1, freqs, ld


ws = [0.2, 0.3, 0.4]
fig, axes = plt.subplots(1, len(ws), figsize=(4 * len(ws), 3.6), sharey=False)
rows = []
for ax, w in zip(axes, ws):
    f1, q1, freqs, ld = cavity_ldos(w)
    ax.plot(freqs, ld / ld.max(), "b-", lw=1.0)
    ax.axvline(f1, color="r", ls="--", lw=0.8, label=f"harminv f={f1:.4f}")
    ax.set_title(f"w={w}: Q={q1:.0f}")
    ax.set_xlabel("frequency")
    ax.legend(fontsize=8)
    rows.append((w, f1, q1))
axes[0].set_ylabel("LDOS (normalized)")
fig.tight_layout()
fig.savefig(OUT / "metal-cavity-ldos.png", dpi=150)
plt.close(fig)
for w, f1, q1 in rows:
    print(f"metal-cavity-ldos  w={w}: f={f1:.4f}  Q={q1:.0f}")
