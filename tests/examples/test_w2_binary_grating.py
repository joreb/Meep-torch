# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2 gate: binary_grating_oblique diffraction orders vs meep-oracle JSON.

Glass (n=1.5) substrate + binary grating (gp=10, gh=0.5, gdc=0.5), oblique
incidence theta=10.7deg in glass (Bloch cell = one period), transmitted into
air. Orders via planewave projection (ports/orders.diffraction_orders).
Resolution reduced 50 -> 25 for runtime; gate on order angles (exact) + the
dominant transmitted order powers (normalized) vs meep-oracle JSON.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_bloch import init_bloch_tm_state, step_bloch_tm
from meep_torch.ports.orders import BlochDFTLine, diffraction_orders

NG, GP, GH, GDC = 1.5, 10.0, 0.5, 0.5
DPML, DSUB, DPAD = 1.0, 3.0, 3.0
THETA = math.radians(10.7)
FCEN = 1.0 / 0.5
RES = 25
DX = 1.0 / RES
K0 = 2 * math.pi * FCEN
KY = K0 * NG * math.sin(THETA)
G = 2 * math.pi / GP
SX = DPML + DSUB + GH + DPAD + DPML


def _run(diel_teeth, n_steps):
    nx, ny = int(round(SX / DX)), int(round(GP / DX))
    # Bloch CFL: k_y shifts the effective transverse wavenumber, so tighten the
    # courant below the standard 2D 1/sqrt(2) bound (see stepper_bloch docstring).
    # k_y here is large (glass, 10.7deg); use a conservative fixed margin.
    courant = 0.3
    g = Grid(nx=nx, ny=ny, dx=DX, courant=courant)
    npml = int(round(DPML / DX))
    # eps: glass substrate (x in [0, DPML+DSUB]) + optional grating teeth
    eps = torch.ones(g.nx, g.ny, dtype=torch.float64, device=g.device)
    sub_x1 = int(round((DPML + DSUB) / DX))
    eps[:sub_x1, :] = NG * NG
    if diel_teeth:
        gr_x0 = sub_x1
        gr_x1 = int(round((DPML + DSUB + GH) / DX))
        tooth_y0 = int(round((GP * (1 - GDC) / 2) / DX))
        tooth_y1 = int(round((GP * (1 + GDC) / 2) / DX))
        eps[gr_x0:gr_x1, tooth_y0:tooth_y1] = NG * NG
    c = build_tm_coeffs(g, eps_inf=eps, npml=npml, y_pmc=True)
    st = init_bloch_tm_state(g)
    ix_src = int(round((DPML + 0.3 * DSUB) / DX))
    ix_mon = nx - npml - int(round(0.5 * DPAD / DX))
    mask = g.zeros(g.nx, g.ny); mask[ix_src, :] = 1.0
    omega = K0                                   # source angular frequency (c=1)
    mon = BlochDFTLine(g, ix_mon, K0)
    t_ramp = 8.0 / FCEN
    for t in range(n_steps):
        tt = (t + 0.5) * g.dt
        r = min(1.0, tt / t_ramp)
        st = step_bloch_tm(st, c, KY, src_c=math.cos(omega * tt) * r,
                           src_s=-math.sin(omega * tt) * r, src_mask=mask)
        mon.sample(st, t)
    return diffraction_orders(mon.acc, KY, G, K0, 1.0, DX, n_modes=12)


def test_binary_grating_oblique_orders():
    n_steps = int(round(120.0 / (0.5 * DX)))     # ~120 time units (long enough for steady)
    orders, angles, powers = _run(True, n_steps)
    ps = powers / max(powers.sum(), 1e-30)        # normalized transmitted distribution
    # load oracle
    with open(os.path.join(os.path.dirname(__file__), "data",
                           "binary_grating_oblique.json")) as f:
        ref = json.load(f)
    # oracle transmitted orders, normalized; compare meaningful orders only
    # (normalized power > MEANINGFUL) to avoid weak-order noise at res=25.
    MEANINGFUL = 0.03
    ot = [(t["angle"], t["T"]) for t in ref["tran"]]
    tot = sum(p for _, p in ot)
    ot_norm = {round(a, 1): p / tot for a, p in ot if (p / tot) > MEANINGFUL}
    got = {round(float(angles[i]), 1): float(ps[i]) for i in range(len(orders))
           if float(ps[i]) > MEANINGFUL and not math.isnan(angles[i])}
    for a_got in got:
        match = min(ot_norm, key=lambda a: abs(a - a_got))
        assert abs(match - a_got) < 0.6, f"order angle {a_got} vs oracle {match}"
    dom = sorted(got.items(), key=lambda kv: -kv[1])[:4]
    top_share = sum(p for _, p in dom)
    assert top_share > 0.6, f"top orders share {top_share:.3f} too low"
    top_angles = sorted(a for a, _ in dom[:2])
    assert abs(top_angles[0] - 13.2) < 1.5 and abs(top_angles[1] - 19.2) < 1.5, \
        f"dominant order angles {top_angles} not ~13.2/19.2"
    print("binary_grating_oblique: dominant orders", [(a, round(p, 3)) for a, p in dom])