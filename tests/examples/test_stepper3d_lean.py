# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Parity gate: memory-lean forward 3D stepper vs the reference step_3d.

Small CPML grid, pulsed soft plane source, 400 steps: the lean in-place
shell-psi stepper must reproduce the functional full-grid stepper's Ez
to float64 round-off.
"""
import torch

from meep_torch.core.monitors_3d import PlaneSource, run_3d
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state
from meep_torch.core.stepper_3d_lean import build_lean, run_3d_lean


def test_lean_matches_reference():
    nx, ny, nz, npml, dx = 60, 52, 44, 10, 0.05
    eps = torch.ones(nx, ny, nz, dtype=torch.float64)
    eps[24:36, 20:32, 16:28] = 4.0                    # dielectric block
    pulse = GaussianPulse(1.0, 0.5)

    cref = build_3d_coeffs(nx, ny, nz, dx, eps=eps, npml=npml)
    src = PlaneSource(nx, ny, nz, cref.dt, "Ey", 0, 14, pulse)
    st, _ = run_3d(init_3d_state(nx, ny, nz, npml=npml), cref, 400,
                   source=src, monitors=[])

    # eps lands on the default device; the coefficients follow it.
    clean = build_lean(nx, ny, nz, dx, eps, npml, dtype=torch.float64,
                       device=eps.device)
    src2 = PlaneSource(nx, ny, nz, clean.dt, "Ey", 0, 14, pulse)
    view, _ = run_3d_lean(clean, 400, source=src2, monitors=[])

    ref = st.Ez
    rel = float((view.Ez - ref).abs().max() / ref.abs().max())
    print(f"lean-vs-ref parity: rel {rel:.3e}")
    assert rel < 1e-10, f"lean stepper diverges from reference: rel {rel:.3e}"


def test_lean_psi_actually_lives_in_the_shells():
    """The memory claim needs its own gate; the parity gate cannot see it.

    `test_lean_matches_reference` catches shells that are too THIN -- the psi
    recursion then misses cells that needed it and the field diverges. It is
    blind to shells that are too FAT, because over-allocating is numerically
    perfect. That asymmetry is not hypothetical: the shell width was derived
    with `b != 0`, and this project's CPML profile has `b == 1` outside the
    PML, so every cell tested active and the shells were the whole grid --
    m_lo = 47 on a 96 grid with npml = 8. The stepper was "memory-lean" in
    name only and the parity gate stayed green throughout.

    A cell is inert iff a == 0 AND k == 1: psi' = b psi + a d leaves psi at
    zero forever once a == 0, whatever b is.
    """
    n, npml = 48, 6
    eps = torch.ones(n, n, n, dtype=torch.float64)
    c = build_lean(n, n, n, 0.05, eps, npml, dtype=torch.float64,
                   device=eps.device)

    for ax in range(3):
        for kind, axes in (("H", c.axH), ("E", c.axE)):
            m_lo, m_hi = axes[ax].m_lo, axes[ax].m_hi
            assert m_lo <= npml and m_hi <= npml, (
                f"{kind} axis {ax}: shells ({m_lo}, {m_hi}) exceed npml={npml}")
            assert m_lo >= npml - 1 and m_hi >= npml - 1, (
                f"{kind} axis {ax}: shells ({m_lo}, {m_hi}) thinner than the PML")

    # and the totals: two shells of npml cells, not two halves of the grid
    shell = sum(p.numel() for pair in c.psi.values() for p in pair)
    full = sum(lo.numel() * n // lo.shape[_shell_axis(lo, hi, npml)]
               for lo, hi in c.psi.values())
    assert shell * 3 < full, (
        f"psi is {shell} elements against a full-volume {full} -- the shells "
        "save nothing, which is exactly what the b != 0 bug looked like")


def _shell_axis(lo, hi, npml):
    """The axis a psi pair is shelled along: the short one."""
    return min(range(3), key=lambda i: min(lo.shape[i], hi.shape[i]))
