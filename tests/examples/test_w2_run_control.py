# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2 Wave 3: stop_when_fields_decayed run control."""
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.run import run_until_decayed
from meep_torch.core.sources import GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.ports.port import Port
from meep_torch.ports.smatrix import compute_smatrix


def test_run_until_decayed_stops_on_decay():
    g = Grid(nx=200, ny=20, dx=0.1)
    eps = torch.ones(g.nx, g.ny, dtype=torch.float64)
    eps[:, 8:12] = 12.0
    c = build_tm_coeffs(g, eps_inf=eps, npml=5)
    src = LineSource(g, 20, 8, 12, profile=GaussianPulse(0.15, 0.1))
    st = init_tm_state(g)
    probe = (100, 10)
    probe_fn = lambda s: s.Ez[probe] * s.Ez[probe]
    st, _ = run_until_decayed(step_tm, st, c, src, [], probe_fn,
                              decay_ratio=1e-6, n_decay=50, max_steps=20000)
    # pulse must have decayed below 1e-6 of peak by the stop
    assert float(st.Ez[probe].abs()) < 1e-3, "field not decayed at stop"


def test_compute_smatrix_decay_matches_fixed():
    g = Grid(nx=300, ny=60, dx=0.1)
    poly = Polygon(np.array([[0, 2.5], [300, 2.5], [300, 3.5], [0, 3.5]]))
    eps = rasterize_scene([(poly, 12.0)], g, background=1.0).eps_tm
    ports = [Port(axis="x", index=30, lo=20, hi=40, direction=+1),
             Port(axis="x", index=270, lo=20, hi=40, direction=-1)]
    kw = dict(eps=eps, grid=g, ports=ports, pol="tm", fcen=0.15, fwidth=0.1,
              freqs=[0.15], npml=10)
    sm_fixed = compute_smatrix(n_steps=6000, **kw)
    sm_decay = compute_smatrix(n_steps=20000, decay_probe=(270, 30), **kw)
    s21_fixed = float(sm_fixed.s[0, 1, 0].abs())
    s21_decay = float(sm_decay.s[0, 1, 0].abs())
    assert abs(s21_decay - s21_fixed) < 1e-2, f"S21 fixed {s21_fixed:.4f} vs decay {s21_decay:.4f}"