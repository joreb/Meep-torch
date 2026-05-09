# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""FD gate for the checkpointed 3D adjoint (run_3d_adjoint).

Single-application objective (even-power chains are blind to sign
errors -- see the fdfd source-gradient incident); central finite
differences on eps voxels at float64."""
import math

import torch

from meep_torch.core.adjoint3d import run_3d_adjoint
from meep_torch.core.monitors_3d import DFTFieldsMonitor3D
from meep_torch.core.sources import GaussianPulse
from meep_torch.core.stepper_3d import build_3d_coeffs, init_3d_state


class _PointSrc:
    def __init__(self, profile, dt, ijk):
        self.profile, self.dt = profile, dt
        self.src = ("Ez", *ijk)

    def amplitude(self, t):
        return self.profile.amplitude((t + 0.5) * self.dt)


def _loss_of(eps, n_steps, chunk):
    nx, ny, nz = eps.shape
    dx = 0.05
    c = build_3d_coeffs(nx, ny, nz, dx, eps=eps, npml=6)
    src = _PointSrc(GaussianPulse(0.6, 0.25), c.dt, (nx // 2, ny // 2, nz // 2 - 3))
    mon = DFTFieldsMonitor3D(c.dt, [0.6], "Ez", plane=(0, nx - 10))
    st0 = init_3d_state(nx, ny, nz, npml=6, dtype=eps.dtype, device=str(eps.device))

    def loss_fn(accs):
        E2 = accs[0][0, 0] ** 2 + accs[0][0, 1] ** 2
        return E2.sum()

    L, _ = run_3d_adjoint(c, n_steps, chunk, src, [mon], loss_fn, st0)
    return L


def test_adjoint3d_fd_gate():
    torch.manual_seed(0)
    nx, ny, nz = 30, 24, 20
    n_steps, chunk = 90, 16                       # non-divisible: tests tail
    eps = torch.full((nx, ny, nz), 1.5, dtype=torch.float64)
    eps[12:18, 9:15, 8:12] = 4.0
    eps = eps.requires_grad_(True)

    L = _loss_of(eps, n_steps, chunk)
    assert eps.grad is not None and torch.isfinite(eps.grad).all()

    h = 1e-4
    checked = 0
    for vox in ((14, 11, 9), (16, 12, 10), (13, 10, 11)):
        g_adj = float(eps.grad[vox])
        ep = eps.detach().clone(); ep[vox] += h
        em = eps.detach().clone(); em[vox] -= h
        g_fd = (_loss_of(ep, n_steps, chunk) - _loss_of(em, n_steps, chunk)) / (2 * h)
        rel = abs(g_adj - g_fd) / max(abs(g_fd), 1e-30)
        assert rel < 1e-4, f"voxel {vox}: adjoint {g_adj} vs FD {g_fd}, rel {rel}"
        checked += 1
    assert checked == 3
