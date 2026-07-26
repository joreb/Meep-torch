# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Per-frequency effective-index tables and autograd group index for 2.5D
slab-mode collapses.

Original code (not a port).

A 1D layer stack ``(thickness, material)`` is discretized at ``dx`` and the
fundamental slab mode is solved per frequency with ``fd1d.solve_1d`` (TM or
TE). The first/last layers are extended by 2 um of cladding padding so the
guided mode is well resolved away from the Dirichlet walls.

``group_index`` computes ``ng = neff + f * dneff/df`` exactly by autograd:
``f`` is a torch scalar with ``requires_grad``; the per-layer eps is built
differentiably from the materials' Lorentz-pole real part (torch, no numpy
detach) and the TM slab eigenproblem is solved with the differentiable
``fd1d.solve_1d_tm_torch``. The total derivative ``dneff/df`` therefore
accounts for both the explicit ``f`` in ``k0 = 2*pi*f`` and the ``f``-dependence
of the eps line.

Lossy materials (substantial imaginary eps) are rejected: the slab-mode
formulation here is lossless (real eps, real eigenproblem). The rejection
threshold is set to separate genuinely lossy media (imag eps ~ O(1)) from
realistic weakly-absorptive library materials whose Lorentz damping gives a
small radiative imaginary part (cSi imag ~ 0.03, SiO2 imag ~ 6e-4 at
telecom). The brief's nominal 1e-6 bound is incompatible with those library
materials and would reject them; 0.1 keeps the contract (reject Fake with
n = 2 + 0.5j, imag eps ~ 2.0) while passing the standard lossless stack -- see
Concerns.

Taking the real part of a slightly-complex eps is the lossless approximation.
Its relative neff error is second order in the imaginary part:
``~ 0.5 * (eps_i / (2*eps_r))**2`` (the first-order perturbation from dropping
``eps_i`` is imaginary and does not shift the real neff; the real shift is the
second-order term). For cSi at 1.55 um (eps_r ~ 12.2, eps_i ~ 0.027) this is
``~ 6e-7`` relative, well below solver discretization error; for SiO2 (eps_i ~
6e-4) it is ``~ 3e-10``. The lossless stack therefore pays no measurable price
for the real-part projection.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import torch

from ..core.materials import Material, analytic_eps
from .fd1d import solve_1d, solve_1d_tm_torch

LOSSY_EPS_IMAG = 0.1
CLADDING_PAD = 2.0


def _material_eps_complex(mat, f: float) -> complex:
    """Complex eps of ``mat`` at scalar frequency ``f``.

    ``Material`` -> ``analytic_eps``. Real number -> constant. Anything else
    is duck-typed via a ``.n`` method and eps is taken as ``n(f)**2`` (so a
    custom lossy test stub that overrides ``n`` is detected).
    """
    if isinstance(mat, Material):
        return analytic_eps(mat.eps_inf, mat.poles, f)
    if isinstance(mat, (int, float)):
        return complex(mat)
    n = mat.n(f)
    return n * n


def _check_lossy(eps_c: complex) -> float:
    """Return ``eps_c.real`` or raise if the material is lossy."""
    if abs(eps_c.imag) > LOSSY_EPS_IMAG:
        raise ValueError(
            f"lossy material not supported by lossless slab solver "
            f"(imag eps = {eps_c.imag:.3e} > {LOSSY_EPS_IMAG}); use a real "
            f"permittivity or a complex mode solver"
        )
    return float(eps_c.real)


def _padded(stack: Sequence[tuple[float, object]]) -> list[tuple[float, object]]:
    """Extend first/last layers by ``CLADDING_PAD`` um."""
    if len(stack) < 2:
        raise ValueError(
            f"need at least 2 layers in the stack to pad (cladding + core), "
            f"got {len(stack)}"
        )
    out = [(stack[0][0] + CLADDING_PAD, stack[0][1])]
    for thickness, mat in stack[1:-1]:
        out.append((thickness, mat))
    out.append((stack[-1][0] + CLADDING_PAD, stack[-1][1]))
    return out


def _eps_line_numpy(stack, f: float, dx: float) -> np.ndarray:
    """Real eps line (numpy) for ``solve_1d`` at scalar frequency ``f``."""
    parts = []
    for thickness, mat in _padded(stack):
        n_cells = max(1, int(round(thickness / dx)))
        eps_real = _check_lossy(_material_eps_complex(mat, f))
        parts.append(np.full(n_cells, eps_real, dtype=np.float64))
    return np.concatenate(parts)


def neff_table(
    stack: Sequence[tuple[float, object]],
    freqs: Sequence[float],
    pol: str,
    dx: float = 0.005,
) -> torch.Tensor:
    """Fundamental slab-mode neff per frequency, as a ``(nf,)`` float64 tensor.

    ``pol`` is ``"tm"`` or ``"te"``. Lossy materials (|imag eps| > threshold)
    raise ``ValueError``.
    """
    out = []
    for f in freqs:
        eps_line = _eps_line_numpy(stack, f, dx)
        modes = solve_1d(eps_line, dx, f, pol, k=1, bc="pec")
        out.append(float(modes.neff[0].real))
    return torch.tensor(out, dtype=torch.float64)


def _analytic_eps_real_torch(mat, f: torch.Tensor) -> torch.Tensor:
    """Real part of ``analytic_eps(mat, f)`` as a torch scalar differentiable in
    ``f`` (the lossless Lorentzian real part; the small gamma term is kept so
    this equals ``analytic_eps(...).real`` exactly, matching ``neff_table``)."""
    if isinstance(mat, Material):
        eps = torch.full((), float(mat.eps_inf), dtype=f.dtype, device=f.device)
        for pole in mat.poles:
            fn2 = float(pole.fn) ** 2
            gamma = float(pole.gamma)
            sigma = float(pole.sigma)
            dr = fn2 - f * f
            eps = eps + sigma * fn2 * dr / (dr * dr + (f * gamma) ** 2)
        return eps
    if isinstance(mat, (int, float)):
        return torch.full((), float(mat), dtype=f.dtype, device=f.device)
    # Duck-typed material with .n: only reached for stubs; not differentiable
    # through f, but group_index is only called with real Materials.
    n = mat.n(float(f.detach().item()))
    return torch.full((), float(n.real) ** 2, dtype=f.dtype, device=f.device)


def _eps_line_torch(stack, f: torch.Tensor, dx: float) -> torch.Tensor:
    """Real eps line (torch, differentiable in ``f``) for ``solve_1d_tm_torch``."""
    parts = []
    for thickness, mat in _padded(stack):
        n_cells = max(1, int(round(thickness / dx)))
        eps_l = _analytic_eps_real_torch(mat, f)
        parts.append(eps_l * torch.ones(n_cells, dtype=f.dtype, device=f.device))
    return torch.cat(parts)


def group_index(
    stack: Sequence[tuple[float, object]],
    f0: float,
    pol: str,
    dx: float = 0.005,
) -> float:
    """Group index ``ng = neff + f * dneff/df`` at ``f0`` via exact autograd.

    Only the TM path is differentiable (``solve_1d_tm_torch``); TE raises
    ``NotImplementedError``.
    """
    if pol != "tm":
        raise NotImplementedError("group_index autograd supports only TM (pol='tm')")
    # Validate lossless up front with the same check as neff_table (the torch
    # real-eps path below does not run _check_lossy itself, so a lossy stack
    # would otherwise be silently projected to its real part).
    for _thickness, mat in stack:
        _check_lossy(_material_eps_complex(mat, float(f0)))
    f = torch.tensor(float(f0), dtype=torch.float64, requires_grad=True)
    eps_line = _eps_line_torch(stack, f, dx)
    neff = solve_1d_tm_torch(eps_line, dx, f, k=1)[0]
    (dneff_df,) = torch.autograd.grad(neff, f, create_graph=False)
    ng = neff + f * dneff_df
    return float(ng.item())


class Medium2D(NamedTuple):
    """2.5D effective-index medium: neff table + linear interpolation on neff^2."""

    table_freqs: torch.Tensor
    table_neff: torch.Tensor

    def eps_at(self, f: float) -> float:
        """Effective eps (= neff^2) at frequency ``f`` by linear interpolation
        of the ``(freqs, neff^2)`` table."""
        freqs = self.table_freqs.detach().cpu().numpy().astype(np.float64)
        neff2 = self.table_neff.detach().cpu().numpy().astype(np.float64) ** 2
        return float(np.interp(f, freqs, neff2))
