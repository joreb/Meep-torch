# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Gates for the 3D planewave band solver.

The order here is the order the checks have to happen in. An operator bug
and an eigensolver bug move the spectrum the same way, so the matvec test
comes first and compares ``Theta v`` against an independently assembled
``A @ v``; only once that holds does an eigenvalue comparison mean
anything. The empty lattice comes next because ``eps = 1`` removes the
epsilon model entirely and leaves the basis, the duals and the FFT
indexing as the only things that can be wrong -- it is exact to 1e-12
where the oracle gates are 1%.
"""
import json
import math
import os

import numpy as np
import pytest

import meep_torch.mp as mp
from meep_torch.mpb.planewave3d import (assemble_dense_3d, kvectors_3d,
                                        planewave_bands_3d,
                                        reciprocal_duals_3d,
                                        theta_matvec_residual,
                                        transverse_frame)
from meep_torch.mpb.raster3d import eps_inv_from_geometry

DATA = os.path.join(os.path.dirname(__file__), "..", "examples", "data")
FCC = ((0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0))
CUBIC = ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def _random_eta(grid, seed=1):
    rng = np.random.default_rng(seed)
    return 1.0 / (1.0 + 10.0 * rng.random(grid))


def test_transverse_frame_is_orthonormal_and_perpendicular():
    B, _ = reciprocal_duals_3d(*FCC)
    K = kvectors_3d(6, 6, 6, B, 2 * np.pi * (np.array([0.1, 0.2, 0.3]) @ B))
    t1, t2 = transverse_frame(K)
    assert np.abs(np.linalg.norm(t1, axis=1) - 1).max() < 1e-12
    assert np.abs(np.linalg.norm(t2, axis=1) - 1).max() < 1e-12
    assert np.abs(np.sum(t1 * t2, axis=1)).max() < 1e-12
    assert np.abs(np.sum(t1 * K, axis=1)).max() < 1e-11
    assert np.abs(np.sum(t2 * K, axis=1)).max() < 1e-11


def test_frame_survives_k_along_z():
    """A K parallel to the frame's reference axis must not collapse.

    ``K x zhat`` is zero for K along z, and every band diagram walks
    through such a point. The swap to xhat is what keeps the normalisation
    from dividing by zero exactly where it is most likely to be used.
    """
    K = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, -2.0], [0.0, 0.0, 0.0]])
    t1, t2 = transverse_frame(K)
    assert np.isfinite(t1).all() and np.isfinite(t2).all()
    assert np.abs(np.linalg.norm(t1, axis=1) - 1).max() < 1e-12
    assert np.abs(np.sum(t1[:2] * K[:2], axis=1)).max() < 1e-12


@pytest.mark.parametrize("lattice", [FCC, CUBIC])
def test_matvec_matches_independent_dense_assembly(lattice):
    """The gate that separates an operator bug from an eigensolver bug."""
    g = (8, 8, 8)
    B, _ = reciprocal_duals_3d(*lattice)
    K = kvectors_3d(*g, B, 2 * np.pi * (np.array([0.1, 0.2, 0.3]) @ B))
    assert theta_matvec_residual(_random_eta(g), K, g) < 1e-12


def test_operator_is_hermitian():
    g = (6, 6, 6)
    B, _ = reciprocal_duals_3d(*FCC)
    K = kvectors_3d(*g, B, 2 * np.pi * (np.array([0.0, 0.5, 0.5]) @ B))
    A = assemble_dense_3d(_random_eta(g), K, g)
    assert np.abs(A - A.conj().T).max() < 1e-12


@pytest.mark.parametrize("kfrac,n_zero", [((0.0, 0.5, 0.5), 0),
                                          ((0.0, 0.0, 0.0), 2)])
def test_empty_lattice_bands_are_analytic(kfrac, n_zero):
    """eps = 1: the eigenvalues are |K_G|^2, each twice.

    Twice because H has two transverse components per G. The zero COUNT at
    Gamma is the sharper half of this test: exactly the K_G = 0 row goes
    null, and a frame that collapsed somewhere else would add zeros here
    without moving any nonzero value enough to notice.
    """
    g = (8, 8, 8)
    B, _ = reciprocal_duals_3d(*FCC)
    K = kvectors_3d(*g, B, 2 * np.pi * (np.array(kfrac) @ B))
    w = np.sort(np.linalg.eigvalsh(assemble_dense_3d(np.ones(g), K, g)))
    want = np.sort(np.repeat(np.sum(K ** 2, axis=1), 2))
    assert np.abs(w - want).max() < 1e-9
    assert int((w < 1e-9).sum()) == n_zero


def test_lobpcg_matches_dense_eigenvalues():
    """The eigensolver, judged against the reference operator's own eigh."""
    g = (8, 8, 8)
    eta = _random_eta(g)
    B, _ = reciprocal_duals_3d(*FCC)
    kc = 2 * np.pi * (np.array([0.0, 0.5, 0.5]) @ B)
    K = kvectors_3d(*g, B, kc)
    ref = np.sort(np.linalg.eigvalsh(assemble_dense_3d(eta, K, g)))[:5]
    got = planewave_bands_3d(eta, *FCC, kc, 5)
    want = np.sqrt(np.clip(ref, 0, None)) / (2 * np.pi)
    assert np.abs(got - want).max() / max(want.max(), 1e-30) < 1e-8


def test_non_positive_epsilon_raises_rather_than_reporting_zero():
    """A negative eigenvalue is an indefinite operator, not round-off."""
    g = (6, 6, 6)
    eta = _random_eta(g)
    eta[0, 0, 0] = -4.0
    B, _ = reciprocal_duals_3d(*FCC)
    kc = 2 * np.pi * (np.array([0.0, 0.5, 0.5]) @ B)
    with pytest.raises((ValueError, RuntimeError)):
        planewave_bands_3d(eta, *FCC, kc, 4)


def _diamond_conventional(n, mesh_size=5):
    r = 0.25 / math.sqrt(2)
    off = (0.25, 0.25, 0.25)
    geometry = []
    for c in ((0, 0, 0),) + FCC:
        for d in ((0, 0, 0), off):
            geometry.append(mp.Sphere(
                r, center=mp.Vector3(c[0] + d[0], c[1] + d[1], c[2] + d[2]),
                material=mp.Medium(epsilon=11.56)))
    basis = [mp.Vector3(1, 0, 0), mp.Vector3(0, 1, 0), mp.Vector3(0, 0, 1)]
    return eps_inv_from_geometry(*CUBIC, n, n, n, geometry, mp.Vector3(0, 0, 0),
                                 mp.Medium(epsilon=1.0), mesh_size, basis)


def test_diamond_conventional_cell_vs_mpb_oracle():
    """8-atom conventional cubic cell at the SC X point, against meep.mpb.

    1% is the gate: MPB's own oracle is itself a resolution-16 calculation,
    so the two are compared at the same discretization rather than against
    a converged truth.
    """
    ref = np.array(json.load(open(os.path.join(DATA, "diamond_mpb.json")))
                   ["bands"][0])
    eta, _ = _diamond_conventional(16)
    B, _ = reciprocal_duals_3d(*CUBIC)
    got = planewave_bands_3d(eta, *CUBIC, 2 * np.pi * (np.array([0, 0.5, 0.5]) @ B),
                             len(ref))
    rel = np.abs(got - ref) / ref
    assert rel.max() < 0.01, f"worst {rel.max():.2%}, got {got}, mpb {ref}"


def test_diamond_epsilon_fill_matches_analytic_sphere_volume():
    """The rasterizer, checked where the answer is known in closed form."""
    _, eps = _diamond_conventional(16)
    r = 0.25 / math.sqrt(2)
    want = 1.0 + (11.56 - 1.0) * 8 * (4 / 3) * math.pi * r ** 3
    assert abs(float(eps.mean()) - want) / want < 0.02


# --- mirror parity --------------------------------------------------------

def _z_symmetric_eta(grid, seed=0):
    """eta symmetric under the FFT grid's own mirror ``k -> (-k) % n``."""
    rng = np.random.default_rng(seed)
    base = 1.0 / (1.0 + 10.0 * rng.random(grid))
    n = grid[2]
    return 0.5 * (base + np.take(base, (-np.arange(n)) % n, axis=2))


def _op(grid, eta, kfrac=(0.2, 0.3, 0.0)):
    from meep_torch.mpb.planewave3d import ThetaOperator
    B, _ = reciprocal_duals_3d(*CUBIC)
    K = kvectors_3d(*grid, B, 2 * np.pi * (np.array(kfrac) @ B))
    return ThetaOperator(eta, K, grid)


@pytest.mark.parametrize("grid", [(8, 8, 8), (8, 8, 7)])
def test_mirror_squares_to_identity_and_projector_is_idempotent(grid):
    """An index-map error breaks these while still looking like a mirror."""
    from meep_torch.mpb.parity3d import zparity_operator
    op = _op(grid, _z_symmetric_eta(grid))
    P = zparity_operator(op, even=True)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((op.n, 3)) + 1j * rng.standard_normal((op.n, 3))
    Xk = X * P._keep[:, None]
    scale = np.abs(Xk).max()
    assert np.abs(P.mirror(P.mirror(Xk)) - Xk).max() / scale < 1e-12
    assert np.abs(P(P(X)) - P(X)).max() / scale < 1e-12


def test_mirror_commutes_with_theta_on_an_odd_grid():
    """Exact only without a Nyquist plane -- which is why parity forces odd.

    On an even grid the mirrored axis has a frequency ``-n/2`` with no
    partner ``+n/2``; the projector drops it, Theta does not, and the two
    stop commuting (measured 0.35 at 8x8x8 against 5e-16 at 8x8x7). The
    sector is then only a Rayleigh-Ritz compression rather than an
    invariant subspace, so the solver makes the axis odd instead.
    """
    from meep_torch.mpb.parity3d import zparity_operator
    grid = (8, 8, 7)
    op = _op(grid, _z_symmetric_eta(grid))
    M = zparity_operator(op, even=True).mirror
    rng = np.random.default_rng(4)
    X = rng.standard_normal((op.n, 2)) + 1j * rng.standard_normal((op.n, 2))
    lhs, rhs = M(op.matmat(X)), op.matmat(M(X))
    assert np.abs(lhs - rhs).max() / np.abs(rhs).max() < 1e-12


def test_parity_requires_zero_k_along_the_mirrored_axis():
    from meep_torch.mpb.parity3d import zparity_operator
    grid = (6, 6, 7)
    op = _op(grid, _z_symmetric_eta(grid), kfrac=(0.2, 0.3, 0.25))
    with pytest.raises(ValueError, match="k_z = 0"):
        zparity_operator(op, even=True)


def test_even_and_odd_sectors_partition_the_kept_space():
    from meep_torch.mpb.parity3d import zparity_operator
    grid = (8, 8, 7)
    op = _op(grid, _z_symmetric_eta(grid))
    Pe = zparity_operator(op, even=True)
    Po = zparity_operator(op, even=False)
    rng = np.random.default_rng(5)
    X = rng.standard_normal((op.n, 2)) + 1j * rng.standard_normal((op.n, 2))
    assert np.abs(Pe(X) + Po(X) - X).max() / np.abs(X).max() < 1e-12


def test_parity_expectation_is_plus_or_minus_one_on_a_pure_sector():
    """<h|M|h> is what MPB prints; a projected vector must be exactly +-1."""
    from meep_torch.mpb.parity3d import zparity_operator
    grid = (8, 8, 7)
    op = _op(grid, _z_symmetric_eta(grid))
    rng = np.random.default_rng(6)
    X = rng.standard_normal((op.n, 2)) + 1j * rng.standard_normal((op.n, 2))
    for even, want in ((True, 1.0), (False, -1.0)):
        P = zparity_operator(op, even=even)
        assert np.abs(P.expectation(P(X)) - want).max() < 1e-10
