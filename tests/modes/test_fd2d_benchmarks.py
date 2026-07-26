# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Flexcompute
# Copyright (c) 2019 Tyler Hughes
# Derived from the ceviche package (https://github.com/fancompute/ceviche).
# Adaptations (c) 2026 meep-torch contributors, MIT.
import numpy as np
import pytest
import torch

from meep_torch.modes.fd2d import solve_2d

# Hadley 2002 rectangular-waveguide benchmarks. Geometry transcribed from
# femwell/docs/benchmarks/mode_solver_rectangle.py (parameters are Hadley's
# published setup, public science; cited to Hadley 2002):
#   - domain 1 x 1, core box(0, 0, 0.5, 0.5) in the corner, wavelength = 1.5 um
#   - metallic (PEC) on left + right (x walls); natural (PMC) on top + bottom
#     (y walls) in femwell's FEM. Our solver imposes PMC at ymin via dmin_pmc and
#     PMC at ymax via dmax_pmc (the high-side mirror _apply_max_pmc doubles the
#     forward-derivative max-face diagonal, since tidy3d's derivative layout only
#     encodes PMC at the low-index side); see task-7-report for the approximation.
#
# Per-case tolerance: the strong-contrast case (eps=8) converges to the Hadley
# reference under grid refinement (n=200 |err| ~1.5e-3), so abs=2e-3. The
# weak-contrast case (eps=2.25) does NOT: the mode is weakly confined, its
# evanescent tail reaches the metallic y walls, and the 2nd-order FD boundary
# scheme (tidy3d dxb[0,0]=2 PMC / dxf-dropped +1 PEC) leaves a refinement-stable
# residual ~6.6e-3 (Richardson limit ~1.2696 vs Hadley 1.27627). dmax_pmc (ymax PMC
# mirror) changes only ~5.6e-5, so the gap is the FD-BC discretization, not the
# ymax condition. abs=1e-2 here is a documented limitation -- do NOT tighten it
# without a higher-order boundary stencil or sub-cell interface treatment.
HADLEY_CASES = [
    (2.25, 1.0, 1.27627404, 1e-2),
    (8.0, 1.0, 2.65679692, 2e-3),
]


def _hadley_eps(eps_core, eps_clad, n=200):
    dx = 1.0 / n
    eps = torch.full((n, n), float(eps_clad), dtype=torch.float64)
    half = n // 2
    eps[:half, :half] = float(eps_core)
    return eps, dx


@pytest.mark.parametrize("eps_core,eps_clad,ref,tol", HADLEY_CASES)
def test_hadley_rectangle(eps_core, eps_clad, ref, tol):
    eps, dx = _hadley_eps(eps_core, eps_clad, n=200)
    lam = 1.5
    f = 1.0 / lam  # spatial frequency (1/um); k0 = 2*pi*f, same vacuum wavenumber as Hz-convention C_0/lam
    m = solve_2d(eps, dx=dx, f=f, k=1, dmin_pmc=(False, True), dmax_pmc=(False, True))
    assert m.neff[0].real.item() == pytest.approx(ref, abs=tol)


def test_slab_limit_matches_fd1d():
    from meep_torch.modes.fd1d import solve_1d

    dx = 0.02
    Nx, Ny = 400, 200
    eps = torch.full((Nx, Ny), 1.444 ** 2, dtype=torch.float64)
    eps[:, 87:113] = 3.4777 ** 2  # 0.52um slab, infinite-ish in x
    m2 = solve_2d(eps, dx=dx, f=1 / 1.55, k=1, dmin_pmc=(True, False))
    line = eps[200, :]
    m1 = solve_1d(line, dx=dx, f=1 / 1.55, pol="tm", k=1)
    assert m2.neff[0].real.item() == pytest.approx(m1.neff[0].item(), abs=1e-3)


def test_si_strip_fundamental_neff():
    dx = 0.01
    Nx, Ny = 300, 200
    eps = torch.full((Nx, Ny), 1.444 ** 2, dtype=torch.float64)
    eps[125:175, 89:111] = 3.4777 ** 2  # 0.5 x 0.22 um core
    m = solve_2d(eps, dx=dx, f=1 / 1.55, k=2)
    assert 2.2 < m.neff[0].real.item() < 2.6
    assert m.neff[0].real.item() > m.neff[1].real.item()


def test_pml_realpart_stable_for_confined_mode():
    # PML sanity: a WELL-CONFINED mode's Re(neff) must be invariant under PML
    # thickness (PML should only add absorption for leakage, not perturb a bound
    # mode). The earlier draft used a barely-guided 2.0-vs-1.444 strip whose
    # evanescent tail (~8 um decay) reached the PEC box walls 1.2 um away; there the
    # PEC box perturbs Re(neff) and the PML *removes* that perturbation, so Re(neff)
    # legitimately shifts (1.449 -> 1.468). That geometry tests the box, not the
    # PML. Here a high-contrast Si strip (3.4777 vs 1.444) is well confined, so the
    # box does not touch the mode and Re(neff) is PML-invariant. We also assert
    # Im(neff) ~ 0 (bound mode does not leak).
    dx = 0.03
    eps = torch.full((100, 60), 1.444 ** 2, dtype=torch.float64)
    eps[40:60, 25:35] = 3.4777 ** 2
    neffs = [solve_2d(eps, dx=dx, f=1 / 1.55, k=1, npml=npml).neff[0] for npml in [(0, 0), (8, 8), (15, 15)]]
    for a, b in [(neffs[0], neffs[1]), (neffs[0], neffs[2]), (neffs[1], neffs[2])]:
        assert a.real.item() == pytest.approx(b.real.item(), abs=5e-3)
    assert abs(neffs[0].imag.item()) < 1e-6