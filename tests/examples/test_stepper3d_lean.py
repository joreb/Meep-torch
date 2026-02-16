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

    clean = build_lean(nx, ny, nz, dx, eps, npml, dtype=torch.float64,
                       device="cpu")
    src2 = PlaneSource(nx, ny, nz, clean.dt, "Ey", 0, 14, pulse)
    view, _ = run_3d_lean(clean, 400, source=src2, monitors=[])

    ref = st.Ez
    rel = float((view.Ez - ref).abs().max() / ref.abs().max())
    print(f"lean-vs-ref parity: rel {rel:.3e}")
    assert rel < 1e-10, f"lean stepper diverges from reference: rel {rel:.3e}"
