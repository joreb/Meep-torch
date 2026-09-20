# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
import math

import torch


def side_counts(n_or_pair) -> tuple[int, int]:
    """Normalize a per-axis layer count to (n_low, n_high) cells: an int
    means the same count on both ends, a (lo, hi) pair sets each end."""
    if isinstance(n_or_pair, (tuple, list)):
        lo, hi = n_or_pair
        return int(lo), int(hi)
    return int(n_or_pair), int(n_or_pair)


def cpml_profiles(
    n: int,
    npml,
    offset: float,
    dt: float,
    dx: float,
    m: int = 3,
    R0: float = 1e-6,
    alpha_max: float = 0.05,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
):
    """CFS-PML (CPML) 1D coefficient profiles at positions i + offset.

    ``npml`` is an int (same layer on both ends) or a (lo, hi) pair of cell
    counts (0 = no layer on that end -> plain PEC wall there).
    Returns (b, a, kappa), each shape (n,). Recursion: psi' = b*psi + a*d,
    effective derivative = d/kappa + psi'.

    ``alpha_max`` is the CFS frequency shift in the stretch

        s(w) = kappa + sigma / (alpha + i w eps0)

    which absorbs evanescent and low-frequency content but leaves the layer
    impedance-matched only in the limit alpha -> 0, so a propagating wave at
    normal incidence sees a residual reflection that grows with alpha.

    Measured on tests/examples/test_w9_wave2.py::test_3d_pml_residual_vs_
    absorber, which pits this layer against a matched-sigma absorber of the
    same depth: at alpha_max=0.05 the CPML decayed 5.693x SLOWER than the
    absorber -- i.e. the PML was the worse absorber -- and at alpha_max=0 and
    0.01 it passes. The 2D guided-reflection gate is unaffected either way,
    and an earlier 2D sweep recorded in wiki/bugs/cpml-high-edge-anchor.md
    found alpha_max=0 inert there (0.230 against the default's 0.227).

    It also matches the oracle: meep carries no CFS term at all. Its PML is
    sigma plus kappa (`mean_stretch`) -- see `use_pml` in
    reference/meep/src/structure.cpp -- and there is no alpha anywhere in it.

    THE DEFAULT IS LEFT AT 0.05 PENDING A DECISION, and the reason is worth
    stating rather than burying. Changing it to 0 moves every CPML number in
    the package, which invalidates two sets of bitwise golden fixtures --
    ``tests/analytic/test_kernels.py::_TM_PRE_KERNEL`` and
    ``tests/analytic/test_allocator.py::_PSI3D_BITWISE``. Those are not physics
    gates; they were captured from the engine BEFORE a refactor to prove the
    refactor moved no bits, so regenerating them at a new alpha would destroy
    exactly the historical comparison they exist for. They now pin
    ``pml_alpha=0.05`` explicitly, so the default can move without taking their
    meaning with it -- a fixture that inherits a default was always depending
    on something it did not state.

    Callers that need the alpha-free layer ask for it: the 3D ringdown gate
    passes ``pml_alpha=0.0`` because it compares this layer against a
    matched-sigma absorber, and the absorber has no CFS term by construction,
    so any alpha at all makes that an unfair comparison rather than a
    measurement.
    """
    n_lo, n_hi = side_counts(npml)
    x = torch.arange(n, dtype=dtype, device=device) + offset
    u = torch.zeros(n, dtype=dtype, device=device)
    sigma = torch.zeros_like(u)
    if n_lo > 0:
        u_lo = torch.clamp((n_lo - x) / n_lo, min=0.0)
        u = u + u_lo
        sigma = sigma + (-(m + 1) * math.log(R0) / (2.0 * n_lo * dx)) * u_lo**m
    if n_hi > 0:
        # the high wall sits at the last node plus the offset: a half-offset
        # array is one node shorter, so anchoring the ramp at n-1 would start
        # it a full cell inside the E ramp and mismatch the E/H stretches
        # (0.18 amplitude reflection of a returning wave at dpml=1, res=10)
        wall = n - 1 + 2.0 * offset
        u_hi = torch.clamp((x - (wall - n_hi)) / n_hi, min=0.0)
        u = u + u_hi
        sigma = sigma + (-(m + 1) * math.log(R0) / (2.0 * n_hi * dx)) * u_hi**m
    u = torch.clamp(u, max=1.0)
    alpha = torch.where(u > 0, alpha_max * (1.0 - u), torch.zeros_like(u))
    kappa = torch.ones_like(u)
    b = torch.exp(-(sigma / kappa + alpha) * dt)
    denom = kappa * (sigma + kappa * alpha)
    a = torch.where(sigma > 0, sigma * (b - 1.0) / denom.clamp(min=1e-300), torch.zeros_like(b))
    return b, a, kappa


def absorber_sigma(n: int, nabs, dx: float, *, R0: float = 1e-15,
                   m: int = 2, dtype=torch.float64, device="cpu") -> torch.Tensor:
    """meep mp.Absorber profile: quadratic sigma ramp on the edges of an
    axis (length n), peak set like the PML profile for asymptotic
    reflection R0. ``nabs`` is an int (both edges) or a (lo, hi) pair of
    cell counts. Matched sigma_d = sigma_b uses this same curve."""
    n_lo, n_hi = side_counts(nabs)
    p = torch.zeros(n, dtype=dtype, device=device)

    def ramp(nc):
        smax = -(m + 1) * math.log(R0) / (2.0 * nc * dx)
        i = torch.arange(nc, dtype=dtype, device=device)
        return smax * ((nc - i - 0.5) / nc) ** m

    if n_lo > 0:
        p[:n_lo] = ramp(n_lo)
    if n_hi > 0:
        p[n - n_hi:] = torch.flip(ramp(n_hi), [0])
    return p
