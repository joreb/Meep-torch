# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import torch

from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper import init_tm_state, step_tm


def gaussian_blob(g, x0, y0, w):
    x = torch.arange(g.nx, dtype=g.dtype)[:, None]
    y = torch.arange(g.ny, dtype=g.dtype)[None, :]
    return torch.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * w**2))


def total_energy(s, m, dx):
    """Naive field energy of a single state.

    Ez (time n) and Hx/Hy (time n - 1/2) sit half a step apart, so this metric
    carries an intrinsic O(omega*dt) staggering oscillation (~3% for the setups
    here). Use it for energy RATIOS and gross-blowup diagnostics (e.g. PML
    absorption), not for exact conservation checks -- use yee_invariant there.
    """
    e = 0.5 * (m.eps_inf * s.Ez**2).sum()
    h = 0.5 * (s.Hx**2).sum() + 0.5 * (s.Hy**2).sum()
    return ((e + h) * dx * dx).item()


def yee_invariant(s_prev, s_curr, m, dx):
    """Exactly-conserved discrete energy of the staggered Yee leapfrog.

    U(n) = 0.5*sum(eps*Ez(n)^2) + 0.5*sum(Hx(n-1/2)*Hx(n+1/2))
                                + 0.5*sum(Hy(n-1/2)*Hy(n+1/2))

    where s_prev is the state before step_tm (Ez at n, H at n-1/2) and s_curr
    the state after (H at n+1/2). In vacuum with closed (PEC) boundaries this
    is constant to machine precision.
    """
    e = 0.5 * (m.eps_inf * s_prev.Ez**2).sum()
    h = 0.5 * (s_prev.Hx * s_curr.Hx).sum() + 0.5 * (s_prev.Hy * s_curr.Hy).sum()
    return ((e + h) * dx * dx).item()


def test_energy_conserved_in_pec_box():
    g = Grid(nx=120, ny=120, dx=0.02)
    m = build_tm_coeffs(g, npml=0)
    s = init_tm_state(g)
    s = s._replace(Ez=gaussian_blob(g, 60.0, 60.0, 6.0), Dz=gaussian_blob(g, 60.0, 60.0, 6.0))
    invariants, energies = [], []
    for i in range(2000):
        s_prev = s
        s = step_tm(s, m)
        if i % 50 == 0:
            invariants.append(yee_invariant(s_prev, s, m, g.dx))
            energies.append(total_energy(s, m, g.dx))
    u = torch.tensor(invariants, dtype=torch.float64)
    e = torch.tensor(energies, dtype=torch.float64)
    # exact discrete invariant: conserved to float64 machine level (measured ~5e-16)
    assert (u.max() - u.min()) / u.mean() < 1e-12
    # naive staggered metric: intrinsic ~3% oscillation, no gross gain/loss
    assert (e.max() - e.min()) / e.mean() < 5e-2
    assert torch.isfinite(u).all() and torch.isfinite(e).all()
