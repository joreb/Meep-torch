# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Sequential per-port S-matrix driver (fdtdx utils/sparams.py pattern).

Forward-only. Phase reference plane = each port's monitor plane.
"""
from typing import NamedTuple, Sequence

import torch

from ..core.coeffs import build_tm_coeffs
from ..core.grid import Grid
from ..core.monitors import FluxMonitorTM
from ..core.run import run_tm
from ..core.sources import GaussianPulse
from ..core.stepper import init_tm_state
from .mode_monitor import decompose
from .mode_source import ModeSourceTM
from .port import Port, validate_ports


class SMatrix(NamedTuple):
    s: torch.Tensor
    freqs: torch.Tensor
    index: list


def _eps_line(eps, p: Port):
    return eps[p.index, p.lo:p.hi] if p.axis == "x" else eps[p.lo:p.hi, p.index]


def _monitor(grid, freqs, p: Port):
    return FluxMonitorTM(grid, freqs, x_normal=(p.axis == "x"),
                         idx=p.index, lo=p.lo, hi=p.hi)


def compute_smatrix(eps, grid: Grid, ports: Sequence[Port], *, pol: str,
                    fcen: float, fwidth: float, freqs: Sequence[float],
                    n_steps: int, npml: int, checkpoint_every: int = 0) -> SMatrix:
    if pol != "tm":
        raise NotImplementedError("pol='te' lands in Task 7")
    validate_ports(ports, grid, npml)
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
    index = [(j, n) for j, p in enumerate(ports) for n in range(p.n_modes)]
    nf, dim = len(freqs), len(index)
    s = torch.zeros(nf, dim, dim, dtype=torch.complex128)

    # ponytail: fixed n_steps; field-decay auto-stop when someone needs it
    for col, (j, n) in enumerate(index):
        src = ModeSourceTM(grid, coeffs, ports[j], GaussianPulse(fcen, fwidth),
                           fcen, mode=n)
        monitors = [_monitor(grid, freqs, p) for p in ports]
        _, accs = run_tm(init_tm_state(grid), coeffs, n_steps, source=src,
                         monitors=monitors, checkpoint_every=checkpoint_every)
        coeff = []
        for p, acc in zip(ports, accs):
            # axis_sign=-1 for both x- and y-normal TM: step_tm stores Hy/Hx with
            # the opposite sign to the fd1d eigenmode (see mode_monitor.decompose).
            sign = -1.0
            coeff.append(decompose(acc, _eps_line(eps, p), grid.dx, freqs,
                                   "tm", p.n_modes, axis_sign=sign))
        a_p, a_m = coeff[j]
        a_in = a_p[:, n] if ports[j].direction > 0 else a_m[:, n]
        for row, (i, m) in enumerate(index):
            b_p, b_m = coeff[i]
            b_out = b_m[:, m] if ports[i].direction > 0 else b_p[:, m]
            s[:, row, col] = b_out / a_in
    return SMatrix(s=s, freqs=torch.tensor(list(freqs), dtype=torch.float64),
                   index=index)