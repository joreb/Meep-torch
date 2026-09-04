# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Even-mirror half-domain symmetry (build_tm_coeffs y_mirror=+1).

Gate: a straight waveguide driven by its TE0 mode gives the same |S21|
whether simulated on the full domain or on the upper half with a PMC
wall at the cut plane (fdtdx symmetry +1 semantics), within 1%.
"""
import numpy as np
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_tm import init_tm_state
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port

RES = 20
NPML = RES
FCEN = 1 / 1.55
PULSE = GaussianPulse(FCEN, 0.2)
N_CO, N_CL = 1.4880, 1.4577


def _s21(half: bool):
    nx = 300
    ny = 120 if not half else 60
    g = Grid(nx=nx, ny=ny, dx=1 / RES)
    eps = torch.full((nx, ny), N_CL ** 2, dtype=g.dtype, device=g.device)
    jw = int(0.05 * RES * 20)                      # 1.0 um half-width wg
    if half:
        eps[:, :jw] = N_CO ** 2                    # core hugs the cut plane
        c = build_tm_coeffs(g, eps_inf=eps, npml=NPML, y_mirror=+1)
        lo, hi, bc, ym = 0, int(2.8 * RES), "pmc_lo", +1
    else:
        cy = ny // 2
        eps[:, cy - jw: cy + jw] = N_CO ** 2
        c = build_tm_coeffs(g, eps_inf=eps, npml=NPML)
        lo, hi, bc, ym = cy - int(2.8 * RES), cy + int(2.8 * RES), "pec", 0
    p_in = Port(axis="x", index=2 * RES, lo=lo, hi=hi, direction=+1)
    p_out = Port(axis="x", index=nx - 2 * RES, lo=lo, hi=hi, direction=-1)
    src = ModeSourceTM(g, c, p_in, PULSE, FCEN, bc=bc)
    mons = [FluxMonitorTM(g, [FCEN], x_normal=True, idx=p.index, lo=p.lo,
                          hi=p.hi) for p in (p_in, p_out)]
    _, accs = run_tm(init_tm_state(g, y_mirror=ym), c,
                     int(round(60.0 / g.dt)), source=src, monitors=mons,
                     checkpoint_every=400)
    out = []
    for p, acc in zip((p_in, p_out), accs):
        el = eps[p.index, p.lo:p.hi]
        out.append(decompose(acc, el, 1 / RES, [FCEN], "tm", 1,
                             axis_sign=-1.0, bc=bc))
    return float((out[1][0][0, 0] / out[0][0][0, 0]).abs())


def test_half_domain_even_mirror_matches_full():
    s_full = _s21(False)
    s_half = _s21(True)
    rel = abs(s_half - s_full) / s_full
    print(f"y_mirror gate: |S21| full={s_full:.5f} half={s_half:.5f} "
          f"rel={rel:.4f}")
    assert rel < 0.01, f"half-domain |S21| off by {rel:.4f}"
