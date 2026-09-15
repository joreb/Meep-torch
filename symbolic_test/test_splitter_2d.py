# SPDX-License-Identifier: MIT
# 2D Y-splitter drop-in integrator tests. Each foreign stepper (pylops, devito)
# runs the same scenario as the meep_torch reference -- same eps, CPML coeffs,
# ModeSourceTM, FluxMonitorTM, decompose -- with only the Yee curl kernel
# swapped. Both curls are validated bitwise matches for step_db (a 3-step state
# diff against step_tm is exactly 0.0), so both sets of S-params sit within
# float64 rounding of the reference. devito is Linux-only, hence the importorskip.
#
# Marked slow: the reference is ~125 s on CPU (8805 steps, 825x324), so these
# are excluded by default (pyproject addopts `-m 'not slow'`). Run explicitly:
#   uv run python -m pytest symbolic_test/test_splitter_2d.py -m slow -v
import numpy as np
import pytest
import torch


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
    # same gate as pylops: the devito forward difference is bitwise the torch
    # one once the grid extent and dtype are right (measured, not assumed).
    assert abs(tu - ref[0]) < 0.01, f"|S21|^2 {tu*100:.2f}% vs ref {ref[0]*100:.2f}%"
    assert abs(td - ref[1]) < 0.01, f"|S31|^2 {td*100:.2f}% vs ref {ref[1]*100:.2f}%"
    assert abs(imb - ref[3]) < 0.005, f"imb {imb*100:.3f}% vs ref {ref[3]*100:.3f}%"

# ---------------------------------------------------------------- adjoints --
# The S-param tests above only exercise the FORWARD kernel; a wrong backward
# would sail through them. These gate the adjoint that makes the drop-ins
# usable inside optimize_splitter --stepper. Fast (no FDTD run for the first,
# 30 steps for the second), so they are NOT marked slow.

def test_polarisation_blocks_decouple():
    """The claim the whole symbolic stepper rests on: with d/dz = 0 the 6x6
    Maxwell system is block-diagonal, and the two 3x3 blocks ARE the
    polarisations. Nothing here declares TM or TE -- assert_decoupled proves
    the off-diagonal blocks vanish, and the membership is read off after."""
    from .maxwell_sympy import assert_decoupled, polarisation_blocks
    assert_decoupled(("z",))                  # raises if they couple
    blocks = polarisation_blocks(("z",))
    assert blocks["tm"]["components"] == ("Ez", "Hx", "Hy"), blocks["tm"]
    assert blocks["te"]["components"] == ("Hz", "Ex", "Ey"), blocks["te"]


def test_sympy_stepper_matches_step_tm():
    """Bitwise drop-in check: 3 steps of step_tm_sympy against step_tm from the
    same state must differ by exactly 0.0 on every field.

    This is the same bar the pylops and devito curls meet. It is reachable
    because the PML coefficients (b, a, k) arrive through the coeffs object --
    only the recursion FORM is derived here, and it lands on meep's."""
    from meep_torch.core.stepper_tm import init_tm_state, step_tm

    from . import splitter_setup as SS
    from .stepper_sympy import step_tm_sympy

    ref = init_tm_state(SS.grid)
    got = init_tm_state(SS.grid)
    for t in range(3):
        amp = SS.src.amplitude(t)
        ref = step_tm(ref, SS.coeffs, amp, SS.src.mask)
        got = step_tm_sympy(got, SS.coeffs, amp, SS.src.mask)
    for name, a, b in zip(ref._fields, ref, got):
        if a is None or a.numel() == 0:
            continue          # P_cur/P_prev are (0, nx, ny) with no poles
        err = float((a - b).abs().max())
        assert err == 0.0, f"{name}: max|diff| {err:.3e} != 0 (not bitwise)"


def test_sparse_operators_are_self_adjoint():
    """<A x, y> == <x, A^T y> for the four pylops difference operators: the
    definition of the adjoint the backward pass applies."""
    from . import splitter_setup as SS
    from .fdtd_pylops_2d import mats
    rng = np.random.default_rng(0)
    grid = (SS.grid.nx, SS.grid.ny, SS.grid.dx)
    for key, (A, _At, _op) in mats(*grid).items():
        x = rng.standard_normal(A.shape[1])
        y = rng.standard_normal(A.shape[0])
        lhs, rhs = float((A @ x) @ y), float(x @ (A.T @ y))
        assert abs(lhs - rhs) <= 1e-10 * abs(lhs), f"{key}: {lhs} vs {rhs}"


def test_cupy_path_matches_numpy_path():
    """The GPU branch of _Diff (cupy CSR + DLPack, no host round-trip) must
    produce the same numbers as the validated numpy branch -- forward AND
    backward. Tolerance is float64 rounding, not a correlation: it is the same
    matrix, so only the summation order differs between cuSPARSE and scipy."""
    pytest.importorskip("cupy")
    if not torch.cuda.is_available():
        pytest.skip("no cuda device")
    from . import fdtd_pylops_2d as PL
    from . import splitter_setup as SS
    nx, ny = SS.grid.nx, SS.grid.ny
    grid = (nx, ny, SS.grid.dx)
    rng = np.random.default_rng(2)

    for key, in_shape, out_shape in (("dyz", (nx, ny), (nx, ny - 1)),
                                     ("dxz", (nx, ny), (nx - 1, ny)),
                                     ("dyh", (nx, ny - 1), (nx, ny - 2)),
                                     ("dxh", (nx - 1, ny), (nx - 2, ny))):
        x = torch.from_numpy(rng.standard_normal(in_shape))
        gy = torch.from_numpy(rng.standard_normal(out_shape))

        def run(t_in, t_g):
            t_in = t_in.clone().requires_grad_(True)
            out = PL._apply(key, t_in, out_shape, grid)
            out.backward(t_g)
            return out.detach(), t_in.grad

        f_cpu, b_cpu = run(x, gy)
        f_gpu, b_gpu = run(x.cuda(), gy.cuda())
        for got, ref, what in ((f_gpu, f_cpu, "forward"), (b_gpu, b_cpu, "backward")):
            scale = max(float(ref.abs().max()), 1e-30)
            err = float((got.cpu() - ref).abs().max())
            assert err <= 1e-12 * scale, f"{key} {what}: max|diff| {err} vs {scale}"


def test_devito_adjoint_matches_sparse_transpose():
    """The devito adjoint Operator is hand-written (devito derives no adjoints),
    so it is checked elementwise against the sparse transpose -- the independent
    twin. Covers both phases and both boundary rows."""
    pytest.importorskip("devito")
    from . import fdtd_devito_2d as DV
    from . import splitter_setup as SS
    from .fdtd_pylops_2d import mats
    nx, ny = SS.grid.nx, SS.grid.ny
    _MAT = {k: v[0] for k, v in mats(nx, ny, SS.grid.dx).items()}
    rng = np.random.default_rng(1)

    def rnd(shape):
        return torch.from_numpy(rng.standard_normal(shape))

    for gy_shape, gx_shape, out_y, out_x, ky, kx in (
        ((nx, ny - 1), (nx - 1, ny), (nx, ny), (nx, ny), "dyz", "dxz"),
        ((nx, ny - 2), (nx - 2, ny), (nx, ny - 1), (nx - 1, ny), "dyh", "dxh"),
    ):
        gy, gx = rnd(gy_shape), rnd(gx_shape)
        ay, ax = DV._adjoint(gy, gx, out_y, out_x, (nx, ny, SS.grid.dx))
        ref_y = (_MAT[ky].T @ gy.numpy().reshape(-1)).reshape(out_y)
        ref_x = (_MAT[kx].T @ gx.numpy().reshape(-1)).reshape(out_x)
        for got, ref, key in ((ay, ref_y, ky), (ax, ref_x, kx)):
            scale = max(float(np.abs(ref).max()), 1e-30)
            err = float((got - torch.from_numpy(ref)).abs().max())
            assert err <= 1e-10 * scale, f"{key}: max|diff| {err} vs {scale}"


@pytest.mark.parametrize("name", ["pylops", "devito"])
def test_gradient_matches_reference(name):
    """End-to-end: d(sum Ez^2)/d(eps) after 30 steps, drop-in vs step_tm. This
    is the gate that matters for optimize_splitter -- forward agreement says
    nothing about the backward."""
    from meep_torch.core.coeffs import build_tm_coeffs
    from meep_torch.core.stepper_tm import init_tm_state, step_tm

    from . import splitter_setup as SS
    if name == "pylops":
        from .fdtd_pylops_2d import step_pylops as step_fn
    else:
        pytest.importorskip("devito")
        from .fdtd_devito_2d import step_devito as step_fn

    def grad_of(fn):
        eps = SS._eps_fin.clone().requires_grad_(True)
        m = build_tm_coeffs(SS.grid, eps_inf=eps, npml=SS.NPML)
        st = init_tm_state(SS.grid)
        for t in range(30):
            st = fn(st, m, SS.src.amplitude(t), SS.src.mask)
        (st.Ez ** 2).sum().backward()
        return eps.grad.clone()

    ref = grad_of(step_tm)
    got = grad_of(step_fn)
    rel = float((got - ref).abs().max() / ref.abs().max())
    assert rel < 1e-10, f"{name}: max|dgrad|/max|grad| = {rel:.2e}"
