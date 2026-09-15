# SPDX-License-Identifier: MIT
# Two FDTD-stencil tests on the symbolic branch: devito and pylops each run
# the straight-waveguide scenario and are compared to the meep_torch.core
# TMz reference (same dx, courant, PML thickness -- only the PML recursion
# differs, so the gate is interior field correlation + peak, not bitwise).
# Both backends are optional: tests skip if the import is missing.
import numpy as np
import pytest

from . import scenario as S


def _interior(mask_pml: int) -> slice:
    return slice(mask_pml, -mask_pml)


def _interior_corr(ref: np.ndarray, cand: np.ndarray, npml: int) -> float:
    s = _interior(npml)
    a = ref[s, s].ravel() - ref[s, s].mean()
    b = cand[s, s].ravel() - cand[s, s].mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def _interior_peak_ratio(ref: np.ndarray, cand: np.ndarray, npml: int) -> float:
    s = _interior(npml)
    return float(np.max(np.abs(cand[s, s])) / max(np.max(np.abs(ref[s, s])), 1e-30))


def test_devito_matches_reference():
    pytest.importorskip("devito")
    from .fdtd_devito import run_devito
    ref = S.reference_ez()
    dv = run_devito(gpu=False)
    assert ref.shape == dv.shape, f"shape {dv.shape} != ref {ref.shape}"
    corr = _interior_corr(ref, dv, S.NPML)
    peak = _interior_peak_ratio(ref, dv, S.NPML)
    assert corr > 0.90, f"devito interior corr {corr:.4f} < 0.90"
    assert 0.5 < peak < 2.0, f"devito peak ratio {peak:.4f} out of [0.5, 2.0]"


def test_sympy_matches_reference():
    """The sympy stencil is the oracle the other two are debugged against, so
    it gets the same gate they do. It found two real devito bugs on its first
    run: the leapfrog read the old H, and the source was not divided by the
    eps of the cell it sits in."""
    from .fdtd_sympy import run_sympy
    ref = S.reference_ez()
    sy = run_sympy()
    assert ref.shape == sy.shape, f"shape {sy.shape} != ref {ref.shape}"
    corr = _interior_corr(ref, sy, S.NPML)
    peak = _interior_peak_ratio(ref, sy, S.NPML)
    assert corr > 0.90, f"sympy interior corr {corr:.4f} < 0.90"
    assert 0.5 < peak < 2.0, f"sympy peak ratio {peak:.4f} out of [0.5, 2.0]"


def test_sympy_is_differentiable():
    """The payoff of the torch backend: grad flows through the sympy-compiled
    stencil with no autograd.Function and no hand-written adjoint anywhere --
    unlike fdtd_pylops_2d (sparse transpose) and fdtd_devito_2d (hand-written
    adjoint Operator), both of which exist only because their round-trip
    severs the tape. Short run; this gates the graph, not the physics."""
    import torch
    from .fdtd_sympy import fields, step_sympy, _stencils
    import math

    eps = S.eps_field("cpu").clone().requires_grad_(True)
    g = fields("cpu", eps=eps)
    fns = _stencils()[:3]
    ez, hx, hy = g["ez"], g["hx"], g["hy"]
    for t in range(20):
        amp = math.cos(2.0 * math.pi * S.FREQ * (t * g["dt"]))
        ez, hx, hy = step_sympy(ez, hx, hy, eps, g["damp"],
                                amp * g["mask"], g["dt"], g["dx"], fns)
    (ez ** 2).sum().backward()
    assert eps.grad is not None, "no gradient reached eps"
    assert torch.isfinite(eps.grad).all(), "non-finite entries in d(sum Ez^2)/d(eps)"
    assert float(eps.grad.abs().max()) > 0.0, "gradient is identically zero"


def test_pylops_matches_reference():
    pytest.importorskip("pylops")
    from .fdtd_pylops import run_pylops
    ref = S.reference_ez()
    pl = run_pylops(gpu=False)
    assert ref.shape == pl.shape, f"shape {pl.shape} != ref {ref.shape}"
    corr = _interior_corr(ref, pl, S.NPML)
    peak = _interior_peak_ratio(ref, pl, S.NPML)
    assert corr > 0.90, f"pylops interior corr {corr:.4f} < 0.90"
    assert 0.5 < peak < 2.0, f"pylops peak ratio {peak:.4f} out of [0.5, 2.0]"