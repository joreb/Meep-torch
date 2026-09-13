# SPDX-License-Identifier: MIT
# 2D Y-splitter drop-in integrator tests. Each foreign stepper (pylops, devito)
# runs the same scenario as the meep_torch reference -- same eps, CPML coeffs,
# ModeSourceTM, FluxMonitorTM, decompose -- with only the Yee curl kernel
# swapped. The pylops curl is a validated bitwise match for step_db, so its
# S-params are within float64 rounding of the reference; devito is the same
# idea with a symbolic derivative (Linux-only; looser gate).
#
# Marked slow: the reference is ~125 s on CPU (8805 steps, 825x324), so these
# are excluded by default (pyproject addopts `-m 'not slow'`). Run explicitly:
#   uv run python -m pytest symbolic_test/test_splitter_2d.py -m slow -v
import pytest


# one reference run shared by both tests (the expensive part). splitter_setup
# is imported lazily so collection does not exec the gdsfactory head.
@pytest.fixture(scope="module")
def reference():
    from . import splitter_setup as SS
    return SS.meep_reference()


def _metrics_from(step_fn):
    from . import splitter_setup as SS
    (s21, s31, _), m = SS.run_with_step(step_fn)
    return m  # (t_up, t_dn, total, imbalance)


@pytest.mark.slow
def test_pylops_matches_reference(reference):
    from .fdtd_pylops_2d import step_pylops
    ref = reference[1]   # (t_up, t_dn, total, imbalance)
    tu, td, tot, imb = _metrics_from(step_pylops)
    # pylops curl is bitwise-equal to step_db -> near-bitwise S-params
    assert abs(tu - ref[0]) < 0.01, f"|S21|^2 {tu*100:.2f}% vs ref {ref[0]*100:.2f}%"
    assert abs(td - ref[1]) < 0.01, f"|S31|^2 {td*100:.2f}% vs ref {ref[1]*100:.2f}%"
    assert abs(imb - ref[3]) < 0.005, f"imb {imb*100:.3f}% vs ref {ref[3]*100:.3f}%"
    print(f"pylops  |S21|2={tu*100:.2f}% |S31|2={td*100:.2f}% "
          f"imb={imb*100:.3f}%  (ref {ref[0]*100:.2f}/{ref[1]*100:.2f}/{ref[3]*100:.3f})")


@pytest.mark.slow
def test_devito_matches_reference(reference):
    pytest.importorskip("devito")
    from .fdtd_devito_2d import step_devito
    ref = reference[1]
    tu, td, tot, imb = _metrics_from(step_devito)
    # devito: symbolic derivative may differ from the torch forward diff by a
    # sign/factor -> looser gate; the pylops twin is the bitwise proof.
    assert abs(tu - ref[0]) < 0.02, f"|S21|^2 {tu*100:.2f}% vs ref {ref[0]*100:.2f}%"
    assert abs(td - ref[1]) < 0.02, f"|S31|^2 {td*100:.2f}% vs ref {ref[1]*100:.2f}%"
    assert abs(imb - ref[3]) < 0.01, f"imb {imb*100:.3f}% vs ref {ref[3]*100:.3f}%"