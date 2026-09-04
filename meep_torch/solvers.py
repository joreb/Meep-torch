# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Unified S-parameter facade over the FDTD and FDFD TM engines: same
ports/eps in, dict[(out, in)] -> complex (nf,) amplitude ratio out, whichever
engine ran.

method="fdtd": the established examples_2/c_y_splitter.py s_params() pattern
(one ModeSourceTM + one FluxMonitorTM per port, single run, multi-freq DFT,
decompose, axis_sign=-1.0) -- house pulse/T_RUN convention baked in.

method="fdfd": one meep_torch.fdfd.solve_fdfd_tm per frequency, normalized
against a reference (straight-guide) solve. A plain FDFD current sheet
radiates both directions (see fdfd.make_mode_source), so there is no
power-normalized "S=1" baseline without a reference run -- ref_eps is
required for this method.
"""
from typing import Mapping, Sequence

import torch

from .core.coeffs import build_tm_coeffs
from .core.grid import Grid
from .core.monitors import FluxMonitorTM
from .core.run import run_tm
from .core.sources import GaussianPulse
from .core.stepper_tm import init_tm_state
from .fdfd import make_mode_source, mode_amplitude, solve_fdfd_tm
from .ports.mode_monitor import decompose
from .ports.mode_source import ModeSourceTM
from .ports.port import Port

__all__ = ["sparams"]


def _port_eps_line(eps: torch.Tensor, port: Port) -> torch.Tensor:
    if port.axis == "x":
        return eps[port.index, port.lo:port.hi]
    return eps[port.lo:port.hi, port.index]


def _sparams_fdtd(eps, dx, freqs, ports, src, npml):
    nx, ny = eps.shape
    grid = Grid(nx=nx, ny=ny, dx=dx, device=str(eps.device))
    coeffs = build_tm_coeffs(grid, eps_inf=eps, npml=npml)
    pulse = GaussianPulse(freqs[0], 0.2)
    source = ModeSourceTM(grid, coeffs, ports[src], pulse, freqs[0])
    names = list(ports)
    monitors = [
        FluxMonitorTM(grid, freqs, x_normal=(ports[n].axis == "x"),
                      idx=ports[n].index, lo=ports[n].lo, hi=ports[n].hi)
        for n in names
    ]
    T_RUN = 2 * (5 / 0.2) + (nx * dx) * 1.7 + 50.0
    n_steps = int(round(T_RUN / grid.dt))
    _, accs = run_tm(init_tm_state(grid), coeffs, n_steps, source=source,
                     monitors=monitors, checkpoint_every=128)
    amps = {}
    for name, acc in zip(names, accs):
        eps_line = _port_eps_line(eps, ports[name]).detach()
        a_p, _ = decompose(acc.cpu(), eps_line, dx, freqs, "tm", 1, axis_sign=-1.0)
        amps[name] = a_p[:, 0]
    a_src = amps[src]
    return {(name, src): amps[name] / a_src for name in names}


def _sparams_fdfd(eps, dx, freqs, ports, src, npml, ref_eps):
    if ref_eps is None:
        raise ValueError(
            "method='fdfd' requires ref_eps: a current-sheet source radiates "
            "both directions, so FDFD amplitudes need a straight-guide "
            "reference run to normalize against (see fdfd.make_mode_source)."
        )
    out = {name: [] for name in ports}
    for f in freqs:
        Ez = solve_fdfd_tm(eps, dx, f, make_mode_source(eps, ports[src], f, dx), npml)
        Ez_ref = solve_fdfd_tm(ref_eps, dx, f, make_mode_source(ref_eps, ports[src], f, dx), npml)
        for name, port in ports.items():
            a_dev = mode_amplitude(Ez, eps, port, f, dx)
            a_ref = mode_amplitude(Ez_ref, ref_eps, port, f, dx)
            out[name].append(a_dev / a_ref)
    return {(name, src): torch.stack(vals) for name, vals in out.items()}


def sparams(eps: torch.Tensor, dx: float, freqs: Sequence[float],
            ports: Mapping[str, Port], src: str, npml: int,
            method: str = "fdtd", ref_eps: torch.Tensor | None = None):
    freqs = list(freqs)
    if method == "fdtd":
        return _sparams_fdtd(eps, dx, freqs, ports, src, npml)
    elif method == "fdfd":
        return _sparams_fdfd(eps, dx, freqs, ports, src, npml, ref_eps)
    raise ValueError(f"method must be 'fdtd' or 'fdfd', got {method!r}")
