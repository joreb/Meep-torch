# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""sparams() facade gates (docs/plans/2026-08-17-fdfd-tm.md Phase 2):
FDTD and FDFD paths agree on the gate-3 notched structure within 5%, and
the FDTD path transmits >0.98 on a plain straight guide. Reuses the exact
parity-structure geometry from tests/examples/test_fdfd.py::test_cross_solver_parity.
"""
import torch

from meep_torch.ports.port import Port
from meep_torch.solvers import sparams

N_CORE, N_CLAD = 2.0, 1.44
FREQ = 1.0 / 1.55
DX, NPML = 0.05, 15
NX, NY = 100, 70


def _stripe_eps(core_lo, core_hi):
    eps = torch.full((NX, NY), N_CLAD ** 2, dtype=torch.float64)
    eps[:, core_lo:core_hi] = N_CORE ** 2
    return eps


def _ports():
    return {
        "src": Port(axis="x", index=30, lo=21, hi=49, direction=+1),
        "out": Port(axis="x", index=70, lo=21, hi=49, direction=+1),
    }


def test_fdtd_straight_guide_transmits():
    eps = _stripe_eps(core_lo=28, core_hi=43)
    S = sparams(eps, DX, [FREQ], _ports(), "src", NPML, method="fdtd")
    T = (S[("out", "src")][0].abs() ** 2).item()
    assert T > 0.98, f"|S21|^2={T:.4f}"


def test_fdtd_fdfd_parity():
    ports = _ports()
    eps_ref = _stripe_eps(core_lo=28, core_hi=43)
    eps_dev = eps_ref.clone()
    eps_dev[49:52, 28:43] = N_CLAD ** 2  # notch, same as test_fdfd's gate 3

    S_fdtd = sparams(eps_dev, DX, [FREQ], ports, "src", NPML, method="fdtd")
    S_fdfd = sparams(eps_dev, DX, [FREQ], ports, "src", NPML, method="fdfd", ref_eps=eps_ref)
    T_fdtd = (S_fdtd[("out", "src")][0].abs() ** 2).item()
    T_fdfd = (S_fdfd[("out", "src")][0].abs() ** 2).item()

    rel_diff = abs(T_fdfd - T_fdtd) / T_fdtd
    assert rel_diff < 0.05, f"T_fdfd={T_fdfd:.4f} T_fdtd={T_fdtd:.4f} rel_diff={rel_diff:.4f}"
