# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import torch

from meep_torch.core.coeffs import build_te_coeffs, build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.core.stepper_te import init_te_state, step_te


def test_te_state_shapes():
    g = Grid(nx=8, ny=6, dx=0.02)
    s = init_te_state(g, n_poles=1)
    assert s.Hz.shape == (7, 5)
    assert s.Ex.shape == (7, 6)
    assert s.Ey.shape == (8, 5)
    assert s.Px_cur.shape == (1, 7, 6)
    assert s.Py_cur.shape == (1, 8, 5)


def test_te_tm_vacuum_duality():
    # In vacuum the discrete TE Hz update graph is isomorphic to the TM Ez graph.
    g = Grid(nx=160, ny=160, dx=0.02)
    x = torch.arange(g.nx, dtype=g.dtype, device=g.device)[:, None]
    y = torch.arange(g.ny, dtype=g.dtype, device=g.device)[None, :]
    G = torch.exp(-((x - 80.0) ** 2 + (y - 80.0) ** 2) / (2 * 5.0**2))

    mtm = build_tm_coeffs(g, npml=0)
    stm = init_tm_state(g)._replace(Ez=G.clone(), Dz=G.clone())

    mte = build_te_coeffs(g, npml=0)
    ste = init_te_state(g)._replace(Hz=G[:-1, :-1].clone())

    for _ in range(40):   # 40 steps * courant 0.5 = 20 cells < 60-cell margin to walls
        stm = step_tm(stm, mtm)
        ste = step_te(ste, mte)

    win = slice(40, 120)
    assert torch.allclose(ste.Hz[win, win], stm.Ez[win, win], atol=1e-12, rtol=0)


def test_te_energy_bounded():
    # The naive same-state metric pairs Hz(n+1) with E(n+1/2) and so carries an
    # intrinsic O(omega*dt) staggering oscillation (~2% here) -- same artifact
    # documented for TM in test_energy.py. Assert the exactly-conserved
    # discrete leapfrog invariant to machine level, and boundedness (no gross
    # gain/loss) of the naive metric.
    g = Grid(nx=100, ny=100, dx=0.02)
    m = build_te_coeffs(g, npml=0)
    x = torch.arange(g.nx - 1, dtype=g.dtype, device=g.device)[:, None]
    y = torch.arange(g.ny - 1, dtype=g.dtype, device=g.device)[None, :]
    G = torch.exp(-((x - 50.0) ** 2 + (y - 50.0) ** 2) / (2 * 5.0**2))
    s = init_te_state(g)._replace(Hz=G)

    def naive_energy(s):
        return (0.5 * (s.Hz**2).sum() + 0.5 * (m.eps_x * s.Ex**2).sum()
                + 0.5 * (m.eps_y * s.Ey**2).sum()).item()

    def invariant(s_prev, s_curr):
        # U(n) = 0.5*sum(Hz(n)*Hz(n+1)) + 0.5*sum(eps*E(n+1/2)^2), the TE dual
        # of yee_invariant in test_energy.py; constant to float64 machine level
        # in a vacuum PEC box.
        h = 0.5 * (s_prev.Hz * s_curr.Hz).sum()
        e = (0.5 * (m.eps_x * s_curr.Ex**2).sum()
             + 0.5 * (m.eps_y * s_curr.Ey**2).sum())
        return (h + e).item()

    invariants, energies = [], []
    for i in range(1500):
        s_prev = s
        s = step_te(s, m)
        if i % 50 == 0:
            invariants.append(invariant(s_prev, s))
            energies.append(naive_energy(s))
    u = torch.tensor(invariants, dtype=torch.float64, device=g.device)
    e = torch.tensor(energies, dtype=torch.float64, device=g.device)
    assert (u.max() - u.min()) / u.mean() < 1e-12
    assert (e.max() - e.min()) / e.mean() < 5e-2
    assert torch.isfinite(u).all() and torch.isfinite(e).all()
