# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""2D TM driven FDFD gates (docs/plans/2026-08-17-fdfd-tm.md):

1. Straight waveguide: |S21|^2 ~ 1 within 2%, and the mode is actually
   guided (center >> edge) rather than radiating flat power everywhere.
2. Adjoint gradient vs. central finite differences on a tiny grid: rel
   err < 1e-5.
3. Cross-solver parity: FDFD transmission of a notched guide vs. the same
   structure run through the FDTD path (build_tm_coeffs/run_tm/decompose),
   agreement within 5%.
"""
import pytest
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper import init_tm_state
from meep_torch.fdfd import make_mode_source, mode_amplitude, solve_fdfd_tm
from meep_torch.ports.mode_monitor import decompose
from meep_torch.ports.mode_source import ModeSourceTM
from meep_torch.ports.port import Port, validate_ports

N_CORE, N_CLAD = 2.0, 1.44
FREQ = 1.0 / 1.55


def _stripe_eps(nx, ny, core_lo, core_hi):
    eps = torch.full((nx, ny), N_CLAD ** 2, dtype=torch.float64)
    eps[:, core_lo:core_hi] = N_CORE ** 2
    return eps


def test_straight_transmission():
    dx, npml = 0.05, 20
    nx, ny = 200, 120
    eps = _stripe_eps(nx, ny, core_lo=53, core_hi=68)  # 15 cells, mid=60
    port_src = Port(axis="x", index=40, lo=44, hi=76, direction=+1)
    port_mid = Port(axis="x", index=100, lo=44, hi=76, direction=+1)
    port_out = Port(axis="x", index=160, lo=44, hi=76, direction=+1)

    jz = make_mode_source(eps, port_src, FREQ, dx)
    Ez = solve_fdfd_tm(eps, dx, FREQ, jz, npml)

    a_mid = mode_amplitude(Ez, eps, port_mid, FREQ, dx)
    a_out = mode_amplitude(Ez, eps, port_out, FREQ, dx)
    ratio = (a_out.abs() / a_mid.abs()).item()
    assert 0.98 < ratio < 1.02, f"|a_out/a_mid|={ratio:.4f}"

    center = abs(complex(Ez[160, 60]))
    edge = abs(complex(Ez[160, 44]))
    assert center > 10 * edge, f"center={center:.3e} edge={edge:.3e}"


def _tiny_ports():
    # lo/hi span the full non-PML interior in y (npml=10, ny=40): the core
    # (rows 13:28) only leaves 2-3 cladding cells on each side at this grid
    # size, so the mode line needs every available cell to clear the
    # neff > cladding guard.
    return (
        Port(axis="x", index=20, lo=10, hi=30, direction=+1),  # source
        Port(axis="x", index=40, lo=10, hi=30, direction=+1),  # monitor
    )


def _objective(eps, dx, npml, port_src, port_out, backend="scipy"):
    jz = make_mode_source(eps, port_src, FREQ, dx)
    Ez = solve_fdfd_tm(eps, dx, FREQ, jz, npml, backend=backend)
    a = mode_amplitude(Ez, eps, port_out, FREQ, dx)
    return (a.conj() * a).real


def test_gradient_check():
    dx, npml = 0.05, 10
    nx, ny = 60, 40
    port_src, port_out = _tiny_ports()
    eps0 = _stripe_eps(nx, ny, core_lo=13, core_hi=28)  # 15 cells, mid=20

    eps = eps0.clone().requires_grad_(True)
    L = _objective(eps, dx, npml, port_src, port_out)
    L.backward()
    grad = eps.grad

    h = 1e-4
    pixels = [(25, 20), (30, 20), (35, 20)]  # between src(20) and out(40), in the core
    for ix, iy in pixels:
        ep = eps0.clone()
        ep[ix, iy] += h
        Lp = float(_objective(ep, dx, npml, port_src, port_out))
        em = eps0.clone()
        em[ix, iy] -= h
        Lm = float(_objective(em, dx, npml, port_src, port_out))
        fd = (Lp - Lm) / (2 * h)
        ad = float(grad[ix, iy])
        rel_err = abs(fd - ad) / abs(fd)
        assert rel_err < 1e-5, f"pixel ({ix},{iy}): fd={fd:.6e} ad={ad:.6e} rel_err={rel_err:.2e}"


def test_cross_solver_parity():
    # NOT gate 2's 60x40/npml=10 grid: the neff>cladding guard there forces
    # a port span that reaches to within 2-3 cells of the guide walls, so
    # the truncated-domain mode profile used for decompose's absolute power
    # normalization is a poor match to the true mode -- FDTD S21 on a plain
    # straight guide (no notch at all) came out at 0.887, not ~1. FDFD's
    # a_dev/a_ref ratio is immune (same profile cancels top and bottom), so
    # this bias only shows up as an FDFD-vs-FDTD gap. A bigger grid with
    # real (non-PML-adjacent) cladding margin on the port lines fixes the
    # FDTD baseline (straight-guide check: 0.993) without touching the
    # notch or loosening the 5% gate.
    dx, npml = 0.05, 15
    nx, ny = 100, 70
    port_src = Port(axis="x", index=30, lo=21, hi=49, direction=+1)
    port_out = Port(axis="x", index=70, lo=21, hi=49, direction=+1)
    eps_ref = _stripe_eps(nx, ny, core_lo=28, core_hi=43)  # 15 cells, mid=35
    eps_dev = eps_ref.clone()
    eps_dev[49:52, 28:43] = N_CLAD ** 2  # 3-cell notch across the core, mid-x

    jz = make_mode_source(eps_ref, port_src, FREQ, dx)
    Ez_ref = solve_fdfd_tm(eps_ref, dx, FREQ, jz, npml)
    Ez_dev = solve_fdfd_tm(eps_dev, dx, FREQ, jz, npml)
    a_ref = mode_amplitude(Ez_ref, eps_ref, port_out, FREQ, dx)
    a_dev = mode_amplitude(Ez_dev, eps_dev, port_out, FREQ, dx)
    T_fdfd = float((a_dev.abs() / a_ref.abs()) ** 2)

    grid = Grid(nx=nx, ny=ny, dx=dx, device="cpu")
    validate_ports([port_src, port_out], grid, npml)
    coeffs = build_tm_coeffs(grid, eps_inf=eps_dev, npml=npml)
    pulse = GaussianPulse(FREQ, 0.2)
    src = ModeSourceTM(grid, coeffs, port_src, pulse, FREQ)
    mon_in = FluxMonitorTM(grid, [FREQ], x_normal=True, idx=port_src.index,
                           lo=port_src.lo, hi=port_src.hi)
    mon_out = FluxMonitorTM(grid, [FREQ], x_normal=True, idx=port_out.index,
                            lo=port_out.lo, hi=port_out.hi)
    T_RUN = 2 * (5 / 0.2) + (nx * dx) * 1.7 + 50.0
    n_steps = int(round(T_RUN / grid.dt))
    _, accs = run_tm(init_tm_state(grid), coeffs, n_steps, source=src,
                     monitors=[mon_in, mon_out], checkpoint_every=128)
    eps_in_line = eps_dev.detach()[port_src.index, port_src.lo:port_src.hi]
    eps_out_line = eps_dev.detach()[port_out.index, port_out.lo:port_out.hi]
    a_in_p, _ = decompose(accs[0].cpu(), eps_in_line, dx, [FREQ], "tm", 1, axis_sign=-1.0)
    a_out_p, _ = decompose(accs[1].cpu(), eps_out_line, dx, [FREQ], "tm", 1, axis_sign=-1.0)
    T_fdtd = float((a_out_p[0, 0].abs() / a_in_p[0, 0].abs()) ** 2)

    rel_diff = abs(T_fdfd - T_fdtd) / T_fdtd
    assert rel_diff < 0.05, f"T_fdfd={T_fdfd:.4f} T_fdtd={T_fdtd:.4f} rel_diff={rel_diff:.4f}"


def test_backend_torch_sla_matches_scipy():
    """Phase 2 adoption gate: torch-sla must match scipy on the actual
    (indefinite) FDFD Helmholtz matrix, not just a diagonally-dominant
    probe matrix -- see docs/plans/2026-08-17-fdfd-tm.md Phase 2."""
    pytest.importorskip("torch_sla")
    dx, npml = 0.05, 15
    nx, ny = 100, 70
    port_src = Port(axis="x", index=30, lo=21, hi=49, direction=+1)
    eps_dev = _stripe_eps(nx, ny, core_lo=28, core_hi=43)
    eps_dev[49:52, 28:43] = N_CLAD ** 2  # same notched parity structure as gate 3
    jz = make_mode_source(eps_dev, port_src, FREQ, dx)
    Ez_scipy = solve_fdfd_tm(eps_dev, dx, FREQ, jz, npml, backend="scipy")
    Ez_tsla = solve_fdfd_tm(eps_dev, dx, FREQ, jz, npml, backend="torch-sla")
    rel = ((Ez_tsla - Ez_scipy).norm() / Ez_scipy.norm()).item()
    assert rel < 1e-8, f"torch-sla vs scipy rel err = {rel:.2e}"


def test_backend_torch_sla_gradient():
    pytest.importorskip("torch_sla")
    dx, npml = 0.05, 10
    nx, ny = 60, 40
    port_src, port_out = _tiny_ports()
    eps0 = _stripe_eps(nx, ny, core_lo=13, core_hi=28)

    eps = eps0.clone().requires_grad_(True)
    L = _objective(eps, dx, npml, port_src, port_out, backend="torch-sla")
    L.backward()
    grad = eps.grad

    h = 1e-4
    for ix, iy in [(25, 20), (30, 20), (35, 20)]:
        ep = eps0.clone(); ep[ix, iy] += h
        em = eps0.clone(); em[ix, iy] -= h
        Lp = float(_objective(ep, dx, npml, port_src, port_out, backend="torch-sla"))
        Lm = float(_objective(em, dx, npml, port_src, port_out, backend="torch-sla"))
        fd = (Lp - Lm) / (2 * h)
        ad = float(grad[ix, iy])
        rel_err = abs(fd - ad) / abs(fd)
        assert rel_err < 1e-5, f"pixel ({ix},{iy}): fd={fd:.6e} ad={ad:.6e} rel_err={rel_err:.2e}"
