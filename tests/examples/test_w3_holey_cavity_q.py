# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W3 gate: holey-wvg-cavity defect-mode Q (FDM harminv) vs meep-oracle.

Builds the meep-torch TM (Ez) cavity (eps=13 slab w=1.2, holes r=0.36, defect
spacing d=1.4, N=3 each side, sy=6, fcen=0.25, df=0.2, res=20) matching the
meep-oracle (tests/examples/data/holey_wvg_cavity_harminv.json), probes the
cavity center, and extracts the defect-mode Q via modes.harminv. Gate: Q within
5% of the meep-oracle harminv Q.
"""
import json
import math
import os

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper import init_tm_state
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.modes.harminv import harminv

EPS, W, R, D, N, SY = 13.0, 1.2, 0.36, 1.4, 3, 6
PAD, DPML = 2, 1
SX = 2 * (PAD + DPML + N) + D - 1
RES = 20
DX = 1.0 / RES
FCEN, DF = 0.25, 0.2


def _circle(cx, cy, rad, n=48):
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return Polygon(np.stack([cx + rad * np.cos(th), cy + rad * np.sin(th)], 1))


def _cavity_probe_series(n_steps):
    nx, ny = int(round(SX / DX)), int(round(SY / DX))
    g = Grid(nx=nx, ny=ny, dx=DX)
    npml = int(round(DPML / DX))
    # slab (full x, y in [-w/2, w/2]); holes carve air
    slab = Polygon(np.array([[0, (SY - W) / 2], [SX, (SY - W) / 2],
                             [SX, (SY + W) / 2], [0, (SY + W) / 2]]))
    items = [(slab, EPS)]
    for i in range(N):
        for sgn in (1, -1):
            cx = SX / 2 + sgn * (D / 2 + i)
            items.append((_circle(cx, SY / 2, R), 1.0))
    rm = rasterize_scene(items, g, background=1.0)
    c = build_tm_coeffs(g, eps_inf=rm.eps_tm, npml=npml)
    # source at the center (Gaussian pulse), probe at the center
    icx, icy = nx // 2, ny // 2
    src = LineSource(g, icx, icy, icy + 1, profile=GaussianPulse(FCEN, DF))
    from meep_torch.core.monitors import PointProbe
    probe = PointProbe(icx, icy)
    st = init_tm_state(g)
    st, (series,) = run_tm(st, c, n_steps, source=src, monitors=[probe])
    return series.detach().numpy()


def test_holey_wvg_cavity_q_vs_oracle():
    n_steps = int(round(500.0 / (0.5 * DX)))         # ~500 time units ringdown
    series = _cavity_probe_series(n_steps)
    dt = 0.5 * DX
    # discard the early source/transient (first ~100 time units)
    cut = int(round(100.0 / dt))
    tail = series[cut:]
    modes = harminv(tail, dt, 0.2, 0.35)
    assert modes, "no cavity mode found"
    with open(os.path.join(os.path.dirname(__file__), "data",
                           "holey_wvg_cavity_harminv.json")) as f:
        ref = json.load(f)
    q0 = ref["modes"][0]["Q"]
    f0 = ref["modes"][0]["freq"]
    # The defect mode is identified by frequency proximity to the oracle (a
    # spurious low-Q slab/radiation mode at ~0.23 can dominate the probe in
    # meep-torch's discretization; meep's symmetric excitation does not show it).
    near = [m for m in modes if abs(m.freq - f0) / f0 < 2e-2]
    assert near, f"defect mode near f={f0} not found; modes={[m.freq for m in modes]}"
    m = max(near, key=lambda x: x.amp)
    assert abs(m.freq - f0) / f0 < 2e-2, f"cavity f={m.freq:.5f} vs oracle {f0:.5f}"
    assert abs(m.Q - q0) / q0 < 0.05, f"cavity Q={m.Q:.2f} vs oracle {q0:.2f} (off {abs(m.Q-q0)/q0:.3f})"
    print(f"holey-wvg-cavity: f={m.freq:.5f} Q={m.Q:.2f} (oracle f={f0:.5f} Q={q0:.2f})")