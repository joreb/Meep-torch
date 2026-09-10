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
from meep_torch import mpb
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


# --- the oblique primitive cell -------------------------------------------

def _diamond_primitive(resolution=16, mesh_size=5, k=(0, 0.5, 0.5),
                       num_bands=5):
    """MPB's own mpb_diamond.py cell: fcc primitive, 2 spheres, oblique.

    This is the gate a rectangular supercell cannot stand in for: a
    supercell folds the bands, and a wrong oblique operator can fold the
    same way and still agree.
    """
    from meep_torch import mpb
    s = math.sqrt(0.5)
    lattice = mp.Lattice(basis_size=mp.Vector3(s, s, s),
                         basis1=mp.Vector3(0, 1, 1),
                         basis2=mp.Vector3(1, 0, 1),
                         basis3=mp.Vector3(1, 1, 0))
    geometry = [mp.Sphere(0.25, center=mp.Vector3(c, c, c),
                          material=mp.Medium(epsilon=11.56))
                for c in (0.125, -0.125)]
    ms = mpb.ModeSolver(geometry_lattice=lattice, geometry=geometry,
                        k_points=[mp.Vector3(*k)], resolution=resolution,
                        num_bands=num_bands, mesh_size=mesh_size)
    ms.run()
    return ms


def test_oblique_primitive_cell_is_solved_directly():
    """The fcc basis is 60 degrees; the FDTD path refuses it outright."""
    ms = _diamond_primitive()
    assert ms._use_3d_path()
    assert len(ms.all_freqs[0]) == 5


@pytest.mark.parametrize("kname,ki", [("X", 0), ("L", 2), ("Gamma", 3)])
def test_diamond_primitive_vs_mpb_oracle(kname, ki):
    """1.5%: measured worst is 1.259% at resolution 16.

    Not 1%, and the reason is worth keeping rather than tightening away.
    Both sides are resolution-16 calculations, and what is left after the
    analytic interface normal is the residual difference between two
    subpixel-smoothing implementations. It converges -- 2.914 / 1.980 /
    1.685% at resolution 16 / 24 / 32 with the moment normal -- so a
    tighter gate here would be a statement about resolution, not about
    correctness.
    """
    path = os.path.join(DATA, "diamond_primitive_mpb.json")
    d = json.load(open(path))
    ref = np.array(d["bands"][ki])
    ms = _diamond_primitive(k=tuple(d["k"][ki]))
    got = np.array(ms.all_freqs[0])
    nz = ref > 1e-9
    rel = np.abs(got[nz] - ref[nz]) / ref[nz]
    assert rel.max() < 0.015, f"{kname}: worst {rel.max():.3%}, got {got}, mpb {ref}"
    # the omega = 0 bands at Gamma must come back exactly zero, not small
    assert np.all(got[~nz] == 0.0)


def test_analytic_normal_beats_the_moment_estimate_on_degeneracy():
    """The cubic symmetry is the discriminator, not the band values.

    A moment-estimated normal is anisotropic on a cubic block and splits
    the triple degeneracy at Gamma; the analytic one keeps it. Comparing
    the SPLITTING rather than the frequency is what makes this test able to
    fail for the right reason.
    """
    ms = _diamond_primitive(k=(0, 0, 0))
    f = np.array(ms.all_freqs[0])
    upper = f[f > 1e-9]
    assert len(upper) == 3
    assert (upper.max() - upper.min()) / upper.mean() < 1e-3


# --- z-parity, cross-section, fields, velocities ---------------------------

def _hole_slab():
    from meep_torch import mpb
    return mpb.ModeSolver(
        num_bands=3, resolution=mp.Vector3(16, 16, 16),
        geometry_lattice=mp.Lattice(
            size=mp.Vector3(1, 1, 4),
            basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
            basis2=mp.Vector3(math.sqrt(3) / 2, -0.5)),
        geometry=[mp.Block(material=mp.Medium(epsilon=12),
                           size=mp.Vector3(mp.inf, mp.inf, 0.5)),
                  mp.Cylinder(0.3, material=mp.air, height=mp.inf)],
        k_points=[mp.Vector3(-1 / 3, 1 / 3)], mesh_size=5)


def test_parity_run_forces_an_odd_grid_on_the_mirrored_axis():
    """The extra grid line: 64 -> 65 along z under zeven."""
    ms = _hole_slab()
    ms._parity = 0
    assert ms._grid_3d()[2] == 64
    from meep_torch.mpb.parity import EVEN_Z
    ms._parity = EVEN_Z
    assert ms._grid_3d()[2] == 65


def test_hole_slab_zeven_at_K_vs_mpb_oracle():
    """run_zeven in 3D: solved INSIDE the sector, not filtered after."""
    ref = np.array(json.load(open(os.path.join(DATA, "hole_slab_mpb.json")))
                   ["K_zeven"])
    ms = _hole_slab()
    ms.run_zeven()
    got = np.array(ms.all_freqs[0])
    rel = np.abs(got - ref) / ref
    assert rel.max() < 0.02, f"worst {rel.max():.3%}, got {got}, mpb {ref}"
    # every returned band must actually BE z-even, not merely nearby
    assert np.abs(np.array(ms.compute_zparities()) - 1.0).max() < 1e-6


def _strip():
    from meep_torch import mpb
    r = json.load(open(os.path.join(DATA, "strip_mpb.json")))
    h, w = r["h"], r["w"]
    geometry = [
        mp.Block(material=mp.Medium(index=r["n_ox"]),
                 size=mp.Vector3(mp.inf, mp.inf, 1 - h / 2),
                 center=mp.Vector3(0, 0, (-1 - h / 2) / 2)),
        mp.Block(material=mp.Medium(index=r["n_si"]),
                 size=mp.Vector3(mp.inf, w, h))]
    ms = mpb.ModeSolver(geometry_lattice=mp.Lattice(size=mp.Vector3(0, 2, 2)),
                        geometry=geometry,
                        k_points=[mp.Vector3(b, 0, 0) for b in r["beta"]],
                        resolution=32, num_bands=2, mesh_size=5)
    return ms, r


def test_strip_cross_section_routes_to_the_3d_path():
    """size=(0,y,z) has two nonzero axes and is NOT a 2D crystal."""
    ms, _ = _strip()
    assert ms._is_cross_section() and ms._use_3d_path()
    assert ms._grid_3d()[0] == 1


def test_strip_dispersion_vs_mpb_oracle():
    ms, r = _strip()
    ms.run()
    got, ref = np.array(ms.all_freqs), np.array(r["bands"])
    rel = np.abs(got - ref) / ref
    assert rel.max() < 0.015, f"worst {rel.max():.3%}, got {got}, mpb {ref}"


def _small_diamond_run(res=12, k=(0, 0.3, 0.2), num_bands=4):
    return _diamond_primitive(resolution=res, k=k, num_bands=num_bands)


def test_field_energy_identities():
    """int |H|^2 = 1 AND int Re(E* . D) = 1.

    The second is the one with teeth: it closes only if the operator, the
    reconstruction and the normalisation all agree, so it catches a wrong
    factor anywhere in the chain that |H|^2 alone would not.
    """
    ms = _small_diamond_run()
    _, volume = reciprocal_duals_3d(*ms._lattice_3d())
    cell = volume / np.prod(ms._grid_3d())
    for band in (1, 3):
        E, D, H = (ms.get_efield(band), ms.get_dfield(band),
                   ms.get_hfield(band))
        assert H.shape == ms._grid_3d() + (3,)
        assert abs(float(np.sum(np.abs(H) ** 2) * cell) - 1.0) < 1e-9
        assert abs(float(np.sum(np.real(np.conj(E) * D)) * cell) - 1.0) < 1e-6


def test_bloch_and_envelope_differ_by_a_unit_phase():
    ms = _small_diamond_run()
    Hb = ms.get_hfield(3)
    He = ms.get_hfield(3, bloch_phase=False)
    sel = np.abs(He) > 1e-12
    assert np.abs(np.abs(Hb[sel] / He[sel]) - 1.0).max() < 1e-12


def test_group_velocity_matches_central_differences():
    """Hellmann-Feynman against df/dk, which needs no second solve to be
    right but is worth one to prove."""
    k0, d = (0.0, 0.30, 0.20), 1e-3
    ms = _small_diamond_run(k=k0)
    got = np.array([[getattr(v, c) for c in "xyz"]
                    for v in ms.compute_group_velocities()])
    fd = []
    for axis in range(3):
        kp, km = list(k0), list(k0)
        kp[axis] += d
        km[axis] -= d
        fp = np.array(_small_diamond_run(k=tuple(kp)).all_freqs[0])
        fm = np.array(_small_diamond_run(k=tuple(km)).all_freqs[0])
        fd.append((fp - fm) / (2 * d))
    fd = np.array(fd).T
    assert np.abs(got - fd).max() < 1e-3, f"got {got}, fd {fd}"


# --- output, poynting, find_k ---------------------------------------------

def test_3d_getters_load_curfield_for_the_output_writers(tmp_path):
    """The writers and the energy functions read curfield, not the return.

    Returning the array without loading curfield is what produced both
    "no field loaded to output" (output_hfield_z) and "needs a D, H or B
    field loaded first" (output_dpwr): the 2D path gets this from _finish
    and the 3D getters bypassed it.
    """
    ms = _hole_slab()
    ms.filename_prefix = str(tmp_path / "hs")
    K = mp.Vector3(-1 / 3, 1 / 3)
    ms.run_zeven(mpb.output_at_kpoint(K, mpb.output_hfield_z))
    assert ms.get_parity_string() == "zeven"      # 3D says zeven, not te
    written = sorted(p.name for p in tmp_path.glob("*.h5"))
    assert written and all(".z.zeven.h5" in n and n.startswith("hs-h.k")
                           for n in written)
    import h5py
    with h5py.File(tmp_path / written[0]) as f:
        assert {"z.r", "z.i", "lattice vectors", "Bloch wavevector"} <= set(f)
        assert f["z.r"].shape == ms._grid_3d()


def test_3d_output_dpwr_writes_without_a_parity_suffix(tmp_path):
    ms = _diamond_primitive(resolution=8, num_bands=2)
    ms.filename_prefix = str(tmp_path / "dm")
    ms.run(mpb.output_dpwr)
    written = sorted(p.name for p in tmp_path.glob("*.h5"))
    assert written == ["dm-dpwr.k01.b01.h5", "dm-dpwr.k01.b02.h5"]


def test_3d_poynting_is_real_and_transverse_dominant():
    """S = Re(conj(E) x H): a guided slab mode carries flux in-plane."""
    ms = _hole_slab()
    ms.run_zeven()
    S = np.asarray(ms.get_poynting(1))
    assert S.shape == ms._grid_3d() + (3,)
    assert np.abs(np.imag(S)).max() < 1e-12
    inplane = np.abs(np.real(S[..., :2])).sum()
    out = np.abs(np.real(S[..., 2])).sum()
    assert inplane > out


@pytest.mark.parametrize("fn,df,name", [
    (lambda k: (k * k - 2, 2 * k), None, "increasing"),
    (lambda k: (2 - k * k, -2 * k), None, "decreasing"),
    (lambda k: (k * k - 2, 0.0), None, "no-derivative"),
])
def test_root_finder_is_orientation_free(fn, df, name):
    """A band with NEGATIVE group velocity must not read backwards.

    The first version tracked the bracket as "below x"/"above x", which
    silently assumes f rises with k. On a decreasing function it returned
    the search bound (0.1) instead of the root (1.414). Frequency does
    usually rise with |k|, which is why that would have survived every
    ordinary case.
    """
    from meep_torch.mpb.findk import _newton_bisect
    got = _newton_bisect(fn, 0.1, 10.0, 5.0 if name != "decreasing" else 0.5,
                         1e-10)
    assert abs(got - math.sqrt(2)) < 1e-6, f"{name}: got {got}"


def test_find_k_round_trips_against_the_solver():
    """find_k answers k(omega); solving at that k must give back omega.

    That round trip is the whole check -- it needs no oracle and it fails
    for every way the root find, the direction normalisation or the group
    velocity could be wrong.
    """
    r = json.load(open(os.path.join(DATA, "strip_mpb.json")))
    ms, _ = _strip()
    ms.resolution = mp.Vector3(24, 24, 24)
    omega = 1 / 1.55
    ks = ms.find_k(0, omega, 1, 2, mp.Vector3(1), 1e-3,
                   omega * r["n_si"], omega * 0.1, omega * 4)
    assert len(ks) == 2 and ks[0] > ks[1] > 0
    ms.num_bands = 2
    ms.k_points = [mp.Vector3(ks[0], 0, 0)]
    ms.run()
    assert abs(float(ms.all_freqs[0][0]) - omega) < 1e-3


def test_weak_convergence_warns_instead_of_raising():
    """MPB does not raise on a marginal miss and neither do we.

    Raising turned mpb_data_analysis into a traceback over a residual of
    1.259e-05 against a 1e-05 threshold. The stopping rule is now the
    relative EIGENVALUE change against ms.tolerance, MPB's own; the
    residual survives as a warning because the eigenvalue history alone
    cannot see a stalled symmetry sector.
    """
    from meep_torch.mpb.planewave3d import planewave_bands_3d
    eta = _random_eta((8, 8, 8))
    B, _ = reciprocal_duals_3d(*FCC)
    kc = 2 * np.pi * (np.array([0.0, 0.5, 0.5]) @ B)
    with pytest.warns(RuntimeWarning, match="weakly converged"):
        planewave_bands_3d(eta, *FCC, kc, 5, maxiter=1, residual_warn=1e-12)
    stats = planewave_bands_3d.last_stats
    assert set(stats) == {"iterations", "eig_change", "residual", "converged"}


# --- device path ----------------------------------------------------------

def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _has_cuda(), reason="no CUDA device")


@needs_cuda
def test_cuda_matvec_matches_numpy():
    """The device apply is a SECOND implementation, not a port of the first.

    The CPU path stays numpy deliberately: every committed gate was
    measured against numpy's FFT, and torch-on-CPU differs in the last
    bits. So the two have to be pinned to each other rather than assumed
    equal by construction.
    """
    from meep_torch.mpb.planewave3d import ThetaOperator
    g = (8, 8, 8)
    eta = _random_eta(g)
    B, _ = reciprocal_duals_3d(*FCC)
    K = kvectors_3d(*g, B, 2 * np.pi * (np.array([0.1, 0.2, 0.3]) @ B))
    cpu = ThetaOperator(eta, K, g)
    gpu = ThetaOperator(eta, K, g, device="cuda")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((cpu.n, 4)) + 1j * rng.standard_normal((cpu.n, 4))
    a, b = cpu.matmat(X), gpu.matmat(X)
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-12


@needs_cuda
def test_cuda_bands_match_cpu_on_the_fcc_case():
    g = (8, 8, 8)
    eta = _random_eta(g)
    B, _ = reciprocal_duals_3d(*FCC)
    kc = 2 * np.pi * (np.array([0.0, 0.5, 0.5]) @ B)
    cpu = planewave_bands_3d(eta, *FCC, kc, 5)
    gpu = planewave_bands_3d(eta, *FCC, kc, 5, device="cuda")
    assert np.abs(cpu - gpu).max() < 1e-12


@needs_cuda
def test_cuda_parity_mirror_matches_cpu():
    """The projector runs twice per operator apply, so it lives on the
    device too -- built on the host while the solve ran on the GPU, it
    cancelled the whole speedup (153 s against 156 s)."""
    from meep_torch.mpb.planewave3d import ThetaOperator
    from meep_torch.mpb.parity3d import zparity_operator
    grid = (8, 8, 7)
    eta = _z_symmetric_eta(grid)
    B, _ = reciprocal_duals_3d(*CUBIC)
    K = kvectors_3d(*grid, B, 2 * np.pi * (np.array([0.2, 0.3, 0.0]) @ B))
    rng = np.random.default_rng(3)
    ops = [ThetaOperator(eta, K, grid, device=d) for d in ("cpu", "cuda")]
    X = rng.standard_normal((ops[0].n, 3)) + 1j * rng.standard_normal((ops[0].n, 3))
    a, b = [zparity_operator(o, even=True).mirror(X) for o in ops]
    assert np.abs(a - b).max() / np.abs(a).max() < 1e-12
