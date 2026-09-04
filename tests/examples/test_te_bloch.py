# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W2-deferral close: TE Bloch stepper — oblique Fresnel (p-pol) gate."""
import math

import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_bloch import (bloch_te_xflux, init_bloch_te_state,
                                           step_bloch_te)

FCEN = 1.0
THETA = math.radians(30.0)
N2 = 1.5
NX, NY, DX, NPML = 400, 40, 0.05, 30


def _run(with_interface):
    g = Grid(nx=NX, ny=NY, dx=DX)
    coeffs = build_tm_coeffs(g, eps_inf=torch.ones(NX, NY, dtype=torch.float64, device=g.device),
                             npml=NPML, y_pmc=True)
    eps = torch.ones(NX, NY, dtype=torch.float64, device=g.device)
    if with_interface:
        eps[240:, :] = N2 ** 2                    # half-space x > 12
    eps_x = eps
    eps_y = 0.5 * (eps[:-1, :] + eps[1:, :])
    st = init_bloch_te_state(g)
    ky = 2 * math.pi * FCEN * math.sin(THETA)
    mask = g.zeros(NX, NY)
    mask[NPML + 10, :] = 1.0
    omega = 2 * math.pi * FCEN
    t_ramp = 5.0 / FCEN
    n_steps = int(round(60.0 / g.dt))
    probe_ix = 160                                # upstream of the interface
    flux = []
    for t in range(n_steps):
        ramp = min(1.0, (t + 0.5) * g.dt / t_ramp)
        c = math.cos(omega * (t + 0.5) * g.dt) * ramp
        s = -math.sin(omega * (t + 0.5) * g.dt) * ramp
        st = step_bloch_te(st, coeffs, eps_x, eps_y, ky,
                           src_c=c, src_s=s, src_mask=mask)
        if t >= n_steps - int(round(5.0 / g.dt)):
            flux.append(bloch_te_xflux(st, probe_ix, DX))
    return float(np.mean(flux))


def test_te_bloch_fresnel_p():
    f_int = _run(True)
    f_vac = _run(False)
    R = 1.0 - f_int / f_vac
    cos_i = math.cos(THETA)
    cos_t = math.sqrt(1.0 - (math.sin(THETA) / N2) ** 2)
    r_p = (N2 * cos_i - cos_t) / (N2 * cos_i + cos_t)
    R_an = r_p ** 2
    assert abs(R - R_an) <= 0.02, f"R={R:.4f} vs Fresnel p {R_an:.4f}"
