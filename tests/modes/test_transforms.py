# SPDX-License-Identifier: MIT
# Copyright (c) 2021 Flexcompute
# Copyright (c) 2019 Tyler Hughes
# Derived from the ceviche package (https://github.com/fancompute/ceviche).
# Adaptations (c) 2026 meep-torch contributors, MIT.
import numpy as np
import pytest
import torch

from meep_torch.modes.fd2d import solve_2d
from meep_torch.modes.transforms import angled_eps_tensor, apply_bend


def strip_eps(Nx=300, Ny=200):
    eps = torch.full((Nx, Ny), 1.444**2, dtype=torch.float64)
    eps[125:175, 89:111] = 3.4777**2
    return eps


def test_bend_converges_to_straight():
    eps = strip_eps()
    straight = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1).neff[0].real.item()
    prev_err = None
    for R in (5.0, 20.0, 80.0):
        nb = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1, bend_radius=R).neff[0].real.item()
        err = abs(nb - straight)
        if prev_err is not None:
            assert err < prev_err          # monotone approach
        prev_err = err
    assert prev_err < 1e-3                  # R=80um essentially straight


def test_bend_raises_neff():
    # outer-shifted mode in a bend has higher effective path -> neff rises at small R
    eps = strip_eps()
    straight = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1).neff[0].real.item()
    bent = solve_2d(eps, dx=0.01, f=1 / 1.55, k=1, bend_radius=5.0).neff[0].real.item()
    assert bent != pytest.approx(straight, abs=1e-6)


def test_angled_tensor_offdiagonal_structure():
    eps = strip_eps(60, 40)
    t = angled_eps_tensor(eps, theta=np.deg2rad(10), phi=0.0)
    assert t.shape == (3, 3, 60, 40)
    assert torch.any(t[0, 2] != 0)          # eps_xz appears
    assert torch.allclose(t[0, 1], torch.zeros_like(t[0, 1]))  # no xy coupling (phi=0)
    t0 = angled_eps_tensor(eps, theta=0.0, phi=0.0)
    for i in range(3):
        assert torch.allclose(t0[i, i], eps)          # theta=0 -> diagonal, unchanged
        for j in range(3):
            if i != j:
                assert torch.allclose(t0[i, j], torch.zeros_like(eps))


# ---------------------------------------------------------------------------
# The typed Jacobians, against the derivation in symbolic_test/transforms.py.
# The existing tests above are property and convergence checks -- a bend that
# approaches straight, an off-diagonal that appears. None of them touches the
# contraction ALGEBRA, which is where the literals live.
# ---------------------------------------------------------------------------

def test_the_typed_bend_dwdz_matches_the_derived_scale_factor():
    """`bend_dwdz` types `R/u`; the derivation gets it from a column norm."""
    import sympy as sp

    from symbolic_test.transforms import R as R_sym
    from symbolic_test.transforms import bend_jacobian
    from symbolic_test.transforms import x as x_sym
    from meep_torch.modes.transforms import bend_dwdz

    _, dwdz_sym = bend_jacobian()
    Nx, Ny, dx, radius, center = 8, 3, 0.25, 4.0, 1.0
    got = bend_dwdz(Nx, Ny, dx, radius, center)
    for i in range(Nx):
        x_cc = (i + 0.5) * dx - center            # offset from the bend radius
        want = float(dwdz_sym.subs({R_sym: radius, x_sym: x_cc}))
        assert got[i, 0] == pytest.approx(want, rel=1e-14), i


def test_the_bend_contraction_in_fd2d_is_the_derived_one():
    """The six hand-expanded lines of `fd2d.py`, against `J eps J^T / det(J)`.

    Built here the way `fd2d` builds it -- scale a diagonal tensor by `dwdz` --
    and compared against the contraction sympy performs with no knowledge of
    that expansion. This is the identity that had no executable form.
    """
    import sympy as sp

    from symbolic_test.transforms import R as R_sym
    from symbolic_test.transforms import bend_jacobian, contract
    from symbolic_test.transforms import x as x_sym
    from meep_torch.modes.transforms import bend_dwdz

    J_sym, _ = bend_jacobian()
    radius, dx, center = 4.0, 0.25, 1.0
    dwdz = bend_dwdz(6, 1, dx, radius, center)[:, 0]
    eps = np.array([2.25, 3.1, 11.9])              # xx, yy, zz, anisotropic

    for i, d in enumerate(dwdz):
        x_cc = (i + 0.5) * dx - center
        # what fd2d.py:429-434 does, transcribed
        typed = np.diag([eps[0] / d, eps[1] / d, eps[2] * d])
        derived = contract(J_sym.subs({R_sym: radius, x_sym: float(x_cc)}),
                           sp.diag(*[sp.Float(v) for v in eps]))
        for a in range(3):
            for b in range(3):
                assert typed[a, b] == pytest.approx(float(derived[a, b]),
                                                    rel=1e-13), (i, a, b)


def test_the_angled_tensor_matches_its_closed_form():
    """Two einsums over a (3, 3, N) array, against four expressions."""
    import sympy as sp

    from symbolic_test.transforms import angled_closed_form, phi as phi_sym
    from symbolic_test.transforms import theta as th_sym
    from meep_torch.modes.transforms import angled_eps_tensor

    closed = angled_closed_form()
    e = sp.Symbol("e", positive=True)
    eps = torch.full((4, 3), 2.25, dtype=torch.float64)
    th, ph = np.deg2rad(12.0), np.deg2rad(35.0)
    got = angled_eps_tensor(eps, theta=th, phi=ph)

    for a in range(3):
        for b in range(3):
            want = float(closed[a, b].subs({th_sym: th, phi_sym: ph, e: 2.25}))
            assert got[a, b].flatten()[0].item() == pytest.approx(want, rel=1e-13), (a, b)


def test_a_bend_radius_inside_the_cross_section_is_refused():
    """`u <= 0` has no arc length, so it has no bend mode.

    The derived scale factor is `R/|R + x|`; the engine types `R/u`, which goes
    NEGATIVE once the cross-section reaches the bend axis and hands the solver a
    medium with `eps_xx < 0` — modes of which it will happily return. Taking the
    derived `Abs` into the engine would be worse: it accepts the degenerate
    geometry silently instead of flipping a sign. Refuse instead.
    """
    from meep_torch.modes.transforms import bend_dwdz

    bend_dwdz(8, 2, 0.25, 4.0, 1.0)                  # u > 0 everywhere: fine
    with pytest.raises(ValueError, match="reaches the bend axis"):
        bend_dwdz(8, 2, 0.25, 0.5, 1.0)              # radius inside the section
