# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Sequential per-port S-matrix driver (fdtdx utils/sparams.py pattern).

Forward-only. Phase reference plane = each port's monitor plane.
"""
from typing import NamedTuple, Sequence

import torch

from ..core.coeffs import build_te_coeffs, build_tm_coeffs
from ..core.grid import Grid
from ..core.monitors import FluxMonitorTE, FluxMonitorTM
from ..core.run import run_te, run_tm
from ..core.sources import GaussianPulse
from ..core.stepper import init_tm_state
from ..core.stepper_te import init_te_state
from .mode_monitor import decompose
from .mode_source import ModeSourceTE, ModeSourceTM
from .port import Port, validate_ports


class SMatrix(NamedTuple):
    s: torch.Tensor
    freqs: torch.Tensor
    index: list


def _eps_line(eps, p: Port):
    return eps[p.index, p.lo:p.hi] if p.axis == "x" else eps[p.lo:p.hi, p.index]


def _eps_line_te(eps, p: Port):
    # eps at Ey half-y nodes (nx, ny-1): average adjacent cell-center rows.
    return 0.5 * (eps[p.index, p.lo:p.hi] + eps[p.index, p.lo + 1:p.hi + 1])


def compute_smatrix(eps, grid: Grid, ports: Sequence[Port], *, pol: str,
                    fcen: float, fwidth: float, freqs: Sequence[float],
                    n_steps: int, npml: int, checkpoint_every: int = 0) -> SMatrix:
    validate_ports(ports, grid, npml)
    if pol == "tm":
        coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
        run = run_tm
        init = init_tm_state
        src_cls = ModeSourceTM
        mon_fn = lambda p: FluxMonitorTM(grid, freqs, x_normal=(p.axis == "x"),
                                         idx=p.index, lo=p.lo, hi=p.hi)
        eps_fn = lambda p: _eps_line(eps, p)
        # S = -Ez*Hy x^ + Ez*Hx y^: +x wave stores Hy=-neff*Ez (sign -1),
        # +y wave stores Hx=+neff*Ez (sign +1).
        sign_fn = lambda p: -1.0 if p.axis == "x" else +1.0
    elif pol == "te":
        if any(p.axis != "x" for p in ports):
            raise NotImplementedError("TE y-normal not supported (FluxMonitorTE is x-normal only)")
        coeffs = build_te_coeffs(grid, eps_inf=eps, npml=npml)
        run = run_te
        init = init_te_state
        src_cls = ModeSourceTE
        mon_fn = lambda p: FluxMonitorTE(grid, freqs, ix=p.index, lo=p.lo, hi=p.hi)
        eps_fn = lambda p: _eps_line_te(eps, p)
        sign_fn = lambda p: -1.0  # unused for te (decompose te branch ignores axis_sign)
    else:
        raise ValueError(f"pol must be 'tm' or 'te', got {pol!r}")

    index = [(j, n) for j, p in enumerate(ports) for n in range(p.n_modes)]
    nf, dim = len(freqs), len(index)
    s = torch.zeros(nf, dim, dim, dtype=torch.complex128)

    # ponytail: fixed n_steps; field-decay auto-stop when someone needs it
    for col, (j, n) in enumerate(index):
        src = src_cls(grid, coeffs, ports[j], GaussianPulse(fcen, fwidth), fcen, mode=n)
        monitors = [mon_fn(p) for p in ports]
        _, accs = run(init(grid), coeffs, n_steps, source=src,
                      monitors=monitors, checkpoint_every=checkpoint_every)
        coeff = [decompose(acc, eps_fn(p), grid.dx, freqs, pol, p.n_modes,
                           axis_sign=sign_fn(p))
                 for p, acc in zip(ports, accs)]
        a_p, a_m = coeff[j]
        a_in = a_p[:, n] if ports[j].direction > 0 else a_m[:, n]
        for row, (i, m) in enumerate(index):
            b_p, b_m = coeff[i]
            b_out = b_m[:, m] if ports[i].direction > 0 else b_p[:, m]
            s[:, row, col] = b_out / a_in
    return SMatrix(s=s, freqs=torch.tensor(list(freqs), dtype=torch.float64),
                   index=index)