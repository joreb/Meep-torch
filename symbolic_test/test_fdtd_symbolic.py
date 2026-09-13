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