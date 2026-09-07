# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""Planewave band solver parity: meep_torch.mpb.planewave vs meep.mpb.

Gated on the committed meep.mpb oracle jsons in tests/examples/data. Every
gate below records the error actually measured, not just the threshold --
if a change moves a number, the docstring is the thing that has to move
with it.

    square rods TM, res 32, Gamma-X-M-Gamma   1.4e-3   gate 1%
    square rods TE, res 32, Kottke diagonal   1.4e-2   gate 2%
    square rods TE, res 32, scalar eps        2.2e-2   (recorded, not gated)
    triangular rods TM, res 32, Gamma/M/K     2.7e-4   gate 1%
    honeycomb rods TM, res 32, Gamma          1.3e-3   gate 1%

Non-oracle numbers worth keeping: the commensurate supercell folds to
2e-13, the sqrt(3) x 1 rectangular supercell agrees with the primitive
cell to 1.9e-3 (two discretizations, not two solves), autograd matches a
central difference to 4e-8, the field energy identity closes to 2e-13, and
the matrix-free TM path reproduces the dense one to 9e-14 while turning a
220 s solve of a 6400-planewave supercell into ~0.01 s.

TE misses the 1% the other three clear, and it is not the solver: the
planewave basis has no operator error left to remove, so what is left is
the epsilon model. ``blend_te_supersampled`` is the DIAGONAL-projected
Kottke variant and drops the eps_xy an interface oblique to the grid axes
carries, which is most of a circular rod's boundary -- the same ceiling
:func:`~meep_torch.geometry.smoothing.blend_te_supersampled` documents for
the FDFD path (which measures 1.22% on this case, i.e. no better). A full
2x2 Kottke tensor would fix it and would need eps_xy support in the
operator; see the report note on planewave.py.
"""
import json
import math
import pathlib

import numpy as np
import pytest
import torch

from meep_torch.geometry.smoothing import blend_te_supersampled
from meep_torch.mpb import planewave as pw
from meep_torch.mpb.planewave import (lattice_grid, planewave_bands_2d,
                                      planewave_fields_2d, reciprocal_duals,
                                      reciprocal_k_to_cartesian)

DATA = pathlib.Path(__file__).parent.parent / "examples" / "data"
SQRT3 = math.sqrt(3)
# MPB's triangular lattice, straight from generate_mpb_oracles2.py
TRI_A1, TRI_A2 = (SQRT3 / 2, 0.5), (SQRT3 / 2, -0.5)


def _rod_blocks(nu, nv, a1, a2, centers, r, eps_rod, ss=16):
    """Supersampled rod rasterization, (nu, ss, nv, ss) -- coarse cells with
    their subsamples still separate, which is what blend_te_supersampled
    wants and what ``.mean(axis=(1, 3))`` turns into the cell average.

    ``centers`` are in LATTICE (fractional) coordinates, MPB's own
    convention for geometry centers on a non-identity basis; ``r`` is a
    cartesian radius. Distances are min-image in fractional coordinates,
    then mapped to the world point ``du*a1 + dv*a2``, so the rod stays
    round on an oblique lattice instead of being sheared.

    ss = 16, not 4: the coverage error of a boundary cell goes like 1/ss^2
    and it is multiplied by (eps_rod - 1) = 11, so it lands on top of the
    1e-3 the gates are trying to measure.
    """
    a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
    uf = (np.arange(nu * ss) + 0.5) / (nu * ss) - 0.5
    vf = (np.arange(nv * ss) + 0.5) / (nv * ss) - 0.5
    U, V = np.meshgrid(uf, vf, indexing="ij")
    fine = np.ones((nu * ss, nv * ss))
    for cu, cv in centers:
        du = (U - cu) - np.round(U - cu)
        dv = (V - cv) - np.round(V - cv)
        x, y = du * a1[0] + dv * a2[0], du * a1[1] + dv * a2[1]
        fine[x * x + y * y < r * r] = float(eps_rod)
    return fine.reshape(nu, ss, nv, ss)


def _cell(*args, **kw):
    return _rod_blocks(*args, **kw).mean(axis=(1, 3))


def _max_rel(got, want):
    return max(abs(g - w) / w for g, w in zip(got, want) if w > 1e-6)


# --- 1. square lattice rods, the mpb tutorial ----------------------------

def _square_blocks(ref):
    n = ref["resolution"]
    return _rod_blocks(n, n, (1, 0), (0, 1), [(0.0, 0.0)], ref["r"], ref["eps"])


def test_square_rods_tm_vs_mpb():
    """r=0.2 eps=12 rods, res 32, TM along Gamma-X-M-Gamma.

    Measured max relative error vs meep.mpb: 1.4e-3 (gate 1%). The FDFD
    sibling measures 3.4e-3 on the same case.
    """
    ref = json.load(open(DATA / "sq_tut_gamma_mpb.json"))
    eps = _square_blocks(ref).mean(axis=(1, 3))
    worst = 0.0
    for kp, want in zip(ref["k"], ref["bands"]):
        got = planewave_bands_2d(eps, (1, 0), (0, 1), 2 * np.pi * np.array(kp[:2]),
                                 len(want), "tm")
        worst = max(worst, _max_rel(got, want))
    assert worst <= 0.01, f"square TM max rel err {worst:.2e}"


def test_square_rods_te_vs_mpb():
    """Same lattice, TE (Hz), with the Kottke diagonal from
    blend_te_supersampled.

    Measured max relative error vs meep.mpb: 1.4e-2 with the Kottke
    diagonal, 2.2e-2 with scalar eps. Gated at 2%, NOT the 1% the TM gates
    use -- see the module docstring: the residual is the dropped eps_xy in
    the diagonal-projected smoothing, not the eigensolver.
    """
    ref = json.load(open(DATA / "sq_tut_te_mpb.json"))
    blocks = _square_blocks(ref)
    eps = blocks.mean(axis=(1, 3))
    eps_te = tuple(np.asarray(e) for e in blend_te_supersampled(blocks))
    worst = {None: 0.0, "kottke": 0.0}
    for kp, want in zip(ref["k"], ref["bands"]):
        kc = 2 * np.pi * np.array(kp[:2])
        for tag, te in ((None, None), ("kottke", eps_te)):
            got = planewave_bands_2d(eps, (1, 0), (0, 1), kc, len(want), "te",
                                     eps_te=te)
            worst[tag] = max(worst[tag], _max_rel(got, want))
    assert worst["kottke"] < worst[None], "Kottke smoothing must beat scalar eps"
    assert worst["kottke"] <= 0.02, f"square TE max rel err {worst['kottke']:.2e}"


# --- 2. oblique lattices, solved directly -------------------------------

def test_tri_rods_tm_vs_mpb():
    """Triangular lattice a1=(sqrt3/2, 1/2), a2=(sqrt3/2, -1/2) solved on
    its own primitive cell -- no rectangular supercell, so the bands come
    out unfolded and band n is band n.

    Measured max relative error vs meep.mpb at Gamma/M/K: 2.7e-4 (gate 1%).
    """
    ref = json.load(open(DATA / "tri_rods_mpb.json"))
    eps = _cell(32, 32, TRI_A1, TRI_A2, [(0.0, 0.0)], ref["r"], ref["eps"])
    # MPB's k_points from the generator, in the reciprocal basis
    kpts = {"gamma": (0.0, 0.0), "M": (0.0, 0.5), "K": (-1 / 3, 1 / 3)}
    for name, kfrac in kpts.items():
        kc = reciprocal_k_to_cartesian(kfrac, TRI_A1, TRI_A2)
        got = planewave_bands_2d(eps, TRI_A1, TRI_A2, kc, len(ref[name]), "tm")
        err = _max_rel(got, ref[name])
        assert err <= 0.01, f"{name}: max rel err {err:.2e}, got {got}"


def test_honey_rods_gamma_vs_mpb():
    """Honeycomb: the same triangular lattice with a 2-rod basis at
    +-(1/6, 1/6) in LATTICE coordinates (= +- sqrt3/6 xhat cartesian).

    Measured max relative error at Gamma: 1.3e-3 (gate 1%). The largest
    contribution is the split of mpb's degenerate 0.7256 pair, which the
    rasterized rods break at the 1e-3 level.
    """
    ref = json.load(open(DATA / "honey_rods_mpb.json"))
    eps = _cell(32, 32, TRI_A1, TRI_A2, [(1 / 6, 1 / 6), (-1 / 6, -1 / 6)],
                ref["r"], ref["eps"])
    got = planewave_bands_2d(eps, TRI_A1, TRI_A2, (0.0, 0.0),
                             len(ref["gamma"]), "tm")
    err = _max_rel(got, ref["gamma"])
    assert err <= 0.01, f"honeycomb gamma: max rel err {err:.2e}, got {got}"


# --- 3. supercell folding ------------------------------------------------

def test_commensurate_supercell_folds_exactly():
    """A supercell that is an exact tiling of the primitive cell must
    reproduce the primitive bands to eigensolver precision.

    Doubling along a2 (lattice vectors a1, 2*a2; grid nu x 2nv from the
    tiled epsilon) leaves the DISCRETE structure identical, so any residual
    is round-off. That is what makes this the test of the G-indexing: get
    the FFT frequency ordering or the reciprocal duals wrong and the folded
    set stops containing the primitive one, with nothing to blame on the
    grid. Measured: 2e-13.
    """
    prim = _cell(24, 24, TRI_A1, TRI_A2, [(0.0, 0.0)], 0.2, 12.0)
    kc = reciprocal_k_to_cartesian((-1 / 3, 1 / 3), TRI_A1, TRI_A2)
    want = planewave_bands_2d(prim, TRI_A1, TRI_A2, kc, 4, "tm")
    doubled = np.concatenate([prim, prim], axis=1)
    got = planewave_bands_2d(doubled, TRI_A1, 2 * np.asarray(TRI_A2), kc, 8, "tm")
    for w in want:
        assert min(abs(got - w)) <= 1e-10, f"band {w} missing from {got}"


def test_rect_supercell_contains_primitive_bands():
    """The sqrt(3) x 1 rectangular supercell of the triangular lattice at
    the K-equivalent k contains the primitive-cell bands at K.

    Unlike the commensurate test above, this compares two DIFFERENT
    discretizations -- 55 x 32 rectangular pixels against 32 x 32
    parallelogram ones -- so the agreement is bounded by the epsilon
    averaging, not by the eigensolver. Measured max relative deviation:
    1.9e-3 (gate 3e-3). It is here as the check that the oblique path and
    the rectangular path describe the same crystal; the exact folding
    check is the one that catches an indexing bug.
    """
    res = 32
    nu = int(round(SQRT3 * res))
    rect_a1, rect_a2 = (nu / res, 0.0), (0.0, 1.0)
    kc = reciprocal_k_to_cartesian((-1 / 3, 1 / 3), TRI_A1, TRI_A2)
    prim = _cell(res, res, TRI_A1, TRI_A2, [(0.0, 0.0)], 0.2, 12.0)
    rect = _cell(nu, res, rect_a1, rect_a2, [(0.0, 0.0), (0.5, 0.5)], 0.2, 12.0)
    want = planewave_bands_2d(prim, TRI_A1, TRI_A2, kc, 4, "tm")
    got = np.asarray(planewave_bands_2d(rect, rect_a1, rect_a2, kc, 8, "tm"))
    for w in want:
        rel = min(abs(got - w)) / w
        assert rel <= 3e-3, f"primitive band {w} not in supercell set {got}"


# --- 4. differentiable path ---------------------------------------------

def test_grad_eps_rod_matches_finite_difference():
    """d f_2 / d eps_rod by autograd against a central difference.

    Measured relative agreement: 4e-8 (gate 1e-6).

    k is generic, (0.3, 0.1): eigvalsh's backward divides by (lam_i -
    lam_j), so a high-symmetry point with a degenerate pair would hand back
    NaN for reasons that have nothing to do with the operator.
    """
    frac = _rod_blocks(16, 16, (1, 0), (0, 1), [(0.0, 0.0)], 0.2, 2.0,
                       ss=8).mean(axis=(1, 3)) - 1.0      # rod fill fraction
    kc = 2 * np.pi * np.array([0.3, 0.1])

    def bands(eps_rod):
        return planewave_bands_2d(1.0 + frac * (eps_rod - 1.0), (1, 0), (0, 1),
                                  kc, 3, "tm")

    rod = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
    eps_t = 1.0 + torch.as_tensor(frac) * (rod - 1.0)
    planewave_bands_2d(eps_t, (1, 0), (0, 1), kc, 3, "tm",
                       differentiable=True)[1].backward()
    h = 1e-5
    fd = (bands(12.0 + h)[1] - bands(12.0 - h)[1]) / (2 * h)
    assert abs(float(rod.grad) - fd) <= 1e-6 * abs(fd), \
        f"autograd {float(rod.grad):.8e} vs fd {fd:.8e}"


def test_differentiable_path_matches_numpy_path():
    """Same operator either way -- the torch path is a backend, not a model.

    TE is checked WITH eps_te as well as without: the torch branch converts
    the Kottke pair itself and indexes eta-hat with torch index tensors, so
    a scalar-only comparison would leave the two-eta assembly untested on
    that side.
    """
    blocks = _rod_blocks(16, 16, (1, 0), (0, 1), [(0.0, 0.0)], 0.2, 12.0, ss=8)
    eps = blocks.mean(axis=(1, 3))
    eps_te = tuple(np.asarray(e) for e in blend_te_supersampled(blocks))
    kc = 2 * np.pi * np.array([0.3, 0.1])
    for pol, te in (("tm", None), ("te", None), ("te", eps_te)):
        a = np.asarray(planewave_bands_2d(eps, (1, 0), (0, 1), kc, 4, pol,
                                          eps_te=te))
        b = planewave_bands_2d(eps, (1, 0), (0, 1), kc, 4, pol, eps_te=te,
                               differentiable=True).detach().numpy()
        assert np.allclose(a, b, atol=1e-12), f"{pol} eps_te={te is not None}"


# --- 5. fields -----------------------------------------------------------

def _generic_case(pol, n=24):
    eps = _cell(n, n, (1, 0), (0, 1), [(0.0, 0.0)], 0.2, 12.0, ss=8)
    kc = 2 * np.pi * np.array([0.3, 0.17])
    freqs, coeffs = planewave_bands_2d(eps, (1, 0), (0, 1), kc, 3, pol,
                                       return_vectors=True)
    return eps, kc, freqs, coeffs


@pytest.mark.parametrize("pol", ["tm", "te"])
def test_field_energy_identity(pol):
    """integral |H|^2 == integral eps|E|^2 == 1, to 1e-8.

    This is the one check that couples the three pieces: the operator, the
    field reconstruction from its eigenvector, and the normalisation. Get
    a K factor, an omega, or a 1/eps wrong in any one of them and the two
    integrals stop agreeing even though each still normalises to something.
    Measured: both sides within 2e-13 of 1.
    """
    eps, kc, freqs, coeffs = _generic_case(pol)
    dA = reciprocal_duals((1, 0), (0, 1))[1] / eps.size
    for b in range(3):
        E, H = planewave_fields_2d(coeffs[:, b], eps, (1, 0), (0, 1), kc,
                                   freqs[b], pol)
        h2 = float((np.abs(H) ** 2).sum() * dA)
        e2 = float((eps[:, :, None] * np.abs(E) ** 2).sum() * dA)
        assert abs(h2 - 1.0) <= 1e-8, f"{pol} band {b}: int|H|^2 = {h2}"
        assert abs(e2 - 1.0) <= 1e-8, f"{pol} band {b}: int eps|E|^2 = {e2}"


@pytest.mark.parametrize("pol", ["tm", "te"])
def test_bloch_phase_factor(pol):
    """Bloch field and periodic envelope differ by exactly exp(i k.r)."""
    eps, kc, freqs, coeffs = _generic_case(pol)
    _, _, r = lattice_grid(*eps.shape, (1, 0), (0, 1))
    phase = np.exp(1j * (r @ np.asarray(kc, dtype=float)))[..., None]
    for b in range(3):
        args = (coeffs[:, b], eps, (1, 0), (0, 1), kc, freqs[b], pol)
        Eb, Hb = planewave_fields_2d(*args)
        Ee, He = planewave_fields_2d(*args, envelope=True)
        assert np.allclose(Hb, He * phase, atol=1e-14, rtol=0)
        assert np.allclose(Eb, Ee * phase, atol=1e-14, rtol=0)


def test_zero_frequency_band_carries_no_field():
    """Gamma TM band 1 (omega = 0) comes back with E and H both identically
    zero, and that is the transverse-H parametrization being honest rather
    than a bug: the null space of A = Kn Eta Kn is exactly the G with
    |K_G| = 0, and the polarization vector (zhat x K_G)/|K_G| that would
    turn that coefficient into an H field does not exist there. The mode it
    stands for is a static uniform Ez, which carries no H and which the
    H-field formulation cannot represent. MPB prints the same 0.0 band.
    """
    eps = _cell(16, 16, (1, 0), (0, 1), [(0.0, 0.0)], 0.2, 12.0, ss=8)
    freqs, coeffs = planewave_bands_2d(eps, (1, 0), (0, 1), (0.0, 0.0), 2,
                                       "tm", return_vectors=True)
    assert freqs[0] == 0.0
    E, H = planewave_fields_2d(coeffs[:, 0], eps, (1, 0), (0, 1), (0.0, 0.0),
                               freqs[0], "tm")
    assert not np.any(E) and not np.any(H)
    # band 2 is a real mode and does carry one
    _, H2 = planewave_fields_2d(coeffs[:, 1], eps, (1, 0), (0, 1), (0.0, 0.0),
                                freqs[1], "tm")
    assert np.abs(H2).max() > 0.1


# --- 6. Gamma zero and the targeted solver -------------------------------

def test_gamma_tm_band1_is_exactly_zero():
    """Not 1e-7, zero: MPB prints 0.0 and so does this."""
    eps = _cell(16, 16, (1, 0), (0, 1), [(0.0, 0.0)], 0.2, 12.0, ss=8)
    for pol in ("tm", "te"):
        f = planewave_bands_2d(eps, (1, 0), (0, 1), (0.0, 0.0), 4, pol)
        assert f[0] == 0.0
        assert f[1] > 0.1


def test_target_freq_picks_the_nearest_band():
    """The targeted solver returns the band nearest the target, not the
    lowest. At X the square-lattice TM bands are well separated, so band 3
    is unambiguous -- ask for one band near it and get exactly it."""
    eps = _cell(16, 16, (1, 0), (0, 1), [(0.0, 0.0)], 0.2, 12.0, ss=8)
    kc = (np.pi, 0.0)
    ladder = planewave_bands_2d(eps, (1, 0), (0, 1), kc, 4, "tm")
    assert ladder[2] - ladder[1] > 0.05 and ladder[3] - ladder[2] > 0.05
    one = planewave_bands_2d(eps, (1, 0), (0, 1), kc, 1, "tm",
                             target_freq=float(ladder[2]) * 1.01)
    assert abs(one[0] - ladder[2]) <= 1e-12
    two = planewave_bands_2d(eps, (1, 0), (0, 1), kc, 2, "tm",
                             target_freq=float(ladder[2]))
    assert min(abs(np.asarray(two) - ladder[2])) <= 1e-12
    assert list(two) == sorted(two)


# --- matrix-free path ----------------------------------------------------

def test_matrix_free_tm_matches_dense(monkeypatch):
    """A = Kn Eta Kn is exactly invertible, so the matrix-free TM solve is
    not an approximation -- it must agree with the dense eigh to round-off,
    frequencies AND fields, at Gamma (where the operator is singular and
    the omega = 0 band has to be put back by hand) as well as at a generic
    k. Measured max deviation: 1e-14."""
    eps = _cell(24, 24, TRI_A1, TRI_A2, [(0.0, 0.0)], 0.2, 12.0, ss=8)
    dA = reciprocal_duals(TRI_A1, TRI_A2)[1] / eps.size
    for kc in [(0.0, 0.0),
               reciprocal_k_to_cartesian((-1 / 3, 1 / 3), TRI_A1, TRI_A2),
               (0.7, 0.3)]:
        want = np.asarray(planewave_bands_2d(eps, TRI_A1, TRI_A2, kc, 5, "tm"))
        monkeypatch.setattr(pw, "DENSE_MAX", 100)
        got, vecs = planewave_bands_2d(eps, TRI_A1, TRI_A2, kc, 5, "tm",
                                       return_vectors=True)
        monkeypatch.undo()
        assert np.allclose(got, want, atol=1e-12, rtol=0), f"{kc}: {got} {want}"
        for b in range(5):
            if got[b] == 0.0:                     # see the omega = 0 test
                continue
            _, H = planewave_fields_2d(vecs[:, b], eps, TRI_A1, TRI_A2, kc,
                                       got[b], "tm")
            assert abs(float((np.abs(H) ** 2).sum() * dA) - 1.0) <= 1e-8


def test_oversized_cell_refuses_rather_than_allocating(monkeypatch):
    """TE and the targeted solver have no matrix-free path; they say so
    instead of silently asking for a gigabyte."""
    monkeypatch.setattr(pw, "DENSE_MAX", 100)
    eps = np.ones((16, 16))
    with pytest.raises(ValueError, match="DENSE_MAX"):
        planewave_bands_2d(eps, (1, 0), (0, 1), (0.3, 0.1), 2, "te")
    with pytest.raises(ValueError, match="DENSE_MAX"):
        planewave_bands_2d(eps, (1, 0), (0, 1), (0.3, 0.1), 2, "tm",
                           target_freq=0.5)


# --- surface -------------------------------------------------------------

def test_reciprocal_duals_and_k_conversion():
    """b_i . a_j = delta_ij, and MPB's K point lands where it should."""
    B, area = reciprocal_duals(TRI_A1, TRI_A2)
    A = np.column_stack([TRI_A1, TRI_A2])
    assert np.allclose(B @ A, np.eye(2))
    assert abs(area - SQRT3 / 2) < 1e-12
    # |K| = 4pi/3 for the triangular lattice with |a| = 1
    kc = reciprocal_k_to_cartesian((-1 / 3, 1 / 3), TRI_A1, TRI_A2)
    assert abs(np.linalg.norm(kc) - 4 * np.pi / 3) < 1e-12


def test_bad_polarization_raises():
    eps = np.ones((4, 4))
    with pytest.raises(ValueError, match="polarization"):
        planewave_bands_2d(eps, (1, 0), (0, 1), (0, 0), 1, "tem")
    with pytest.raises(ValueError, match="num_bands"):
        planewave_bands_2d(eps, (1, 0), (0, 1), (0, 0), 99, "tm")
