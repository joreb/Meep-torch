# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
# Diagonal-projected Kottke averaging, implemented from the published scheme
# (Kottke, Farjadpour & Johnson, Phys. Rev. E 77, 036611 (2008)).
import torch


def blend_tm(eps_in, eps_out, cov: torch.Tensor) -> torch.Tensor:
    return cov * eps_in + (1.0 - cov) * eps_out


def blend_te(eps_in, eps_out, cov: torch.Tensor, nrm: torch.Tensor):
    a = cov * eps_in + (1.0 - cov) * eps_out
    h = 1.0 / (cov / eps_in + (1.0 - cov) / eps_out)
    nx2 = nrm[..., 0] ** 2
    ny2 = nrm[..., 1] ** 2
    boundary = (nx2 + ny2) > 0.5           # unit normal where set, zero elsewhere
    eps_xx = torch.where(boundary, nx2 * h + (1 - nx2) * a, a)
    eps_yy = torch.where(boundary, ny2 * h + (1 - ny2) * a, a)
    return eps_xx, eps_yy
