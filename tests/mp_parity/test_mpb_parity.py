# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""MPB-side classes parity: meep_torch.mpb ModeSolver / MPBArray / MPBData.

Band gate reruns test_mpb_track's square-lattice-rods TM case (and the 1D
sine-index Bragg case) through the meep-shaped ModeSolver API against the
committed meep.mpb oracle jsons, to the same 2% tolerance. MPBData is gated
on exact invariants (period tiling, constant rectify) plus bit-parity with
real meep.mpb.MPBData when the meep-oracle env is present.
"""
import json
import math
import os
import pathlib
import subprocess

import numpy as np
import pytest

from meep_torch.mpb import (EVEN_Z, NO_PARITY, ODD_Z, ModeSolver, MPBArray,
                            MPBData)
from meep_torch.mp.geometry import Cylinder, Lattice
from meep_torch.mp.material import Medium
from meep_torch.mp.vec import Vector3

DATA = pathlib.Path(__file__).parent.parent / "examples" / "data"
ORACLE = os.path.expanduser("~/miniforge3/envs/meep-oracle/bin/python")


# --- MPBArray -------------------------------------------------------------

def test_mpbarray_attrs_survive_views():
    lat = np.eye(3)
    arr = MPBArray(np.arange(12.0).reshape(3, 4), lat, kpoint=Vector3(0.1),
                   bloch_phase=True)
    assert arr.lattice is lat
    assert arr.kpoint == Vector3(0.1)
    assert arr.bloch_phase
    sl = arr[1:, :2]
    assert sl.lattice is lat and sl.kpoint == Vector3(0.1) and sl.bloch_phase
    v = np.asarray(np.zeros(3)).view(MPBArray)
    assert v.lattice is None and v.kpoint is None and not v.bloch_phase


# --- ModeSolver surface ---------------------------------------------------

def test_modesolver_constructor_surface():
    ms = ModeSolver()
    assert ms.num_bands == 1
    assert ms.resolution == [10, 10, 10]
    assert ms.k_points == [] and ms.geometry == []
    assert ms.freqs == [] and ms.band_range_data == []
    assert ms.tolerance == 1.0e-7 and ms.mesh_size == 3
    assert ms.target_freq == 0.0 and not ms.is_negative_epsilon_ok
    ms.resolution = Vector3(8, 16, 1)
    assert ms.resolution == [8.0, 16.0, 1.0]
    with pytest.raises(TypeError):
        ms.resolution = "nope"


def test_unbacked_paths_raise():
    ms = ModeSolver(resolution=8, num_bands=1, k_points=[Vector3(0.1)],
                    geometry_lattice=Lattice(size=Vector3(1, 1)),
                    geometry=[Cylinder(0.2, material=Medium(epsilon=4.0))])
    with pytest.raises(NotImplementedError):
        ms.run()                          # 2D no-parity: union of TM + TE
    # 2D DOES export fields now; the field getters raise on the 1D/3D
    # harminv backend instead, which recovers frequencies from a time
    # series and has no eigenvectors to hand out.
    ms1 = ModeSolver(resolution=8, num_bands=1, k_points=[Vector3(0.1)],
                     geometry_lattice=Lattice(size=Vector3(1, 0, 0)),
                     geometry=[])
    ms1.run_tm()
    with pytest.raises(NotImplementedError, match="eigenvectors"):
        ms1.get_efield(1)
    for call in (lambda: ms.get_charge_density(1), lambda: ms.get_poynting(1),
                 lambda: ms.get_eigenvectors(1, 1),
                 lambda: ms.compute_group_velocities(),
                 lambda: ms.randomize_fields()):
        with pytest.raises(NotImplementedError):
            call()


def test_te_bands_vs_mpb_oracle():
    """run_te (zeven) through the FDFD Bloch eigensolve, over the full
    Gamma-X-M-Gamma path of the mpb_tutorial square-lattice case.

    TE carries E in-plane across the rod boundaries, where the normal
    component of E is discontinuous, so the interface treatment dominates
    the error. With diagonal-projected Kottke subpixel smoothing:
    2.45% at resolution 16, 1.22% at 32, 0.67% at 64 (plain volume
    averaging was 5.43 / 2.60 / 1.16%).
    """
    ref = json.load(open(DATA / "sq_tut_te_mpb.json"))
    ms = ModeSolver(num_bands=4, resolution=ref["resolution"],
                    k_points=[Vector3(*k) for k in ref["k"]],
                    geometry_lattice=Lattice(size=Vector3(1, 1)),
                    geometry=[Cylinder(ref["r"],
                                       material=Medium(epsilon=ref["eps"]))])
    ms.run_te()
    assert ms.parity == "zeven"
    got, want = np.array(ms.all_freqs), np.array(ref["bands"])[:, :4]
    assert got[0, 0] == 0.0                       # omega = 0 band at Gamma
    err = np.abs(got - want) / np.where(want > 1e-9, want, 1.0)
    assert err.max() <= 0.015, f"worst TE band error {err.max():.3%}"


def test_tm_bands_vs_mpb_oracle_full_path():
    """TM over the same path, including the four small-|k| points where the
    harminv backend silently misnumbered bands ((0.1,0), (0.2,0), (0.3,0),
    (0.1,0.1) -- band 1 was missed and every band shifted down a slot).
    An eigensolve returns the N lowest eigenvalues complete and ordered, so
    the band index cannot drift."""
    ref = json.load(open(DATA / "sq_tut_gamma_mpb.json"))
    ms = ModeSolver(num_bands=4, resolution=ref["resolution"],
                    k_points=[Vector3(*k) for k in ref["k"]],
                    geometry_lattice=Lattice(size=Vector3(1, 1)),
                    geometry=[Cylinder(ref["r"],
                                       material=Medium(epsilon=ref["eps"]))])
    ms.run_tm()
    got, want = np.array(ms.all_freqs), np.array(ref["bands"])[:, :4]
    err = np.abs(got - want) / np.where(want > 1e-9, want, 1.0)
    assert err.max() <= 0.01, f"worst TM band error {err.max():.3%}"
    # the TM gap the tutorial is about: mpb reports 39.8% gap-midgap
    gap = min(got[:, 1]) - max(got[:, 0])
    assert 0.35 <= 2 * gap / (min(got[:, 1]) + max(got[:, 0])) <= 0.42


def test_differentiable_bands_match_finite_difference():
    """differentiable=True must put real gradient on epsilon, and must not
    change the frequencies: a detached path that silently returned the same
    numbers without gradient would pass any value-only check."""
    import torch
    from meep_torch.mpb.eigensolver import bloch_bands_2d

    n = 16
    ii = (np.arange(n) + 0.5) / n - 0.5
    X, Y = np.meshgrid(ii, ii, indexing="ij")
    rod = torch.as_tensor((X ** 2 + Y ** 2) < 0.2 ** 2, dtype=torch.float64)

    def band2(scale):
        eps = 1.0 + rod * (scale - 1.0)
        return bloch_bands_2d(eps, 1 / n, 1 / n, 2 * np.pi * 0.3, 0.0, 3,
                              "tm", differentiable=True)[1]

    s = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
    band2(s).backward()
    h = 1e-5
    with torch.no_grad():
        fd = (band2(torch.tensor(12.0 + h, dtype=torch.float64)).item()
              - band2(torch.tensor(12.0 - h, dtype=torch.float64)).item()) / (2 * h)
    assert s.grad is not None and abs(s.grad.item()) > 1e-6
    assert abs(s.grad.item() - fd) / abs(fd) < 1e-5
    # both backends must agree on the values themselves
    eps_np = 1.0 + rod.cpu().numpy() * 11.0
    sci = bloch_bands_2d(eps_np, 1 / n, 1 / n, 2 * np.pi * 0.3, 0.0, 3, "tm")
    with torch.no_grad():
        tor = bloch_bands_2d(1.0 + rod * 11.0, 1 / n, 1 / n, 2 * np.pi * 0.3,
                             0.0, 3, "tm", differentiable=True)
    assert np.allclose(sci, [float(v) for v in tor], atol=1e-12)


def test_oblique_lattice_raises():
    """An oblique basis is rasterized on a cartesian grid and would solve a
    different structure (triangular K band 1 came back 27.75% off), so it
    must raise rather than return plausible wrong bands."""
    tri = Lattice(size=Vector3(1, 1),
                  basis1=Vector3(math.sqrt(3) / 2, 0.5),
                  basis2=Vector3(math.sqrt(3) / 2, -0.5))
    ms = ModeSolver(num_bands=2, resolution=16, geometry_lattice=tri,
                    k_points=[Vector3(0.2)],
                    geometry=[Cylinder(0.2, material=Medium(epsilon=12))])
    with pytest.raises(NotImplementedError, match="oblique lattice basis"):
        ms.run_tm()


def test_gamma_tm_band1_is_zero():
    """At k = 0 the omega = 0 mode (Ez = const) never oscillates, so harminv
    cannot see it; without reinserting it every found band lands one index
    too low. meep.mpb gives exactly 0.0 for TM band 1 at Gamma."""
    ref = json.load(open(DATA / "sq_tut_gamma_mpb.json"))
    ms = ModeSolver(num_bands=3, resolution=32, k_points=[Vector3()],
                    geometry_lattice=Lattice(size=Vector3(1, 1)),
                    geometry=[Cylinder(ref["r"],
                                       material=Medium(epsilon=ref["eps"]))])
    ms.run_tm()
    assert ms.all_freqs[0][0] == 0.0
    # band 2 at Gamma follows the oracle once band 1 occupies its own slot
    assert abs(ms.all_freqs[0][1] - ref["bands"][0][1]) / ref["bands"][0][1] <= 0.02


def test_gamma_3d_no_parity_raises():
    """3D no-parity has more than one omega = 0 mode at Gamma and the
    multiplicity is not established, so band numbering must refuse."""
    ms = ModeSolver(num_bands=1, resolution=8, k_points=[Vector3()],
                    geometry_lattice=Lattice(size=Vector3(1, 1, 1)))
    ms.fmax, ms.t_end = 0.4, 150.0
    with pytest.raises(NotImplementedError, match="Gamma"):
        ms.run()


# --- band gates vs committed meep.mpb oracles -----------------------------

def test_sq_lattice_tm_bands_vs_mpb_oracle():
    """Gate: square lattice rods (r=0.2, eps=8.9) TM bands through the
    ModeSolver API vs the committed meep.mpb oracle -- same case, tolerance
    and k subset as tests/examples/test_mpb_track.py."""
    ref = json.load(open(DATA / "sq_lattice_mpb.json"))
    ms = ModeSolver(
        num_bands=4, resolution=16,
        k_points=[Vector3(0.2), Vector3(0.4)],
        geometry_lattice=Lattice(size=Vector3(1, 1)),
        geometry=[Cylinder(ref["r"], material=Medium(epsilon=ref["eps"]))])
    ms.fmax = 0.8
    ms.run_tm()
    assert ms.all_freqs.shape == (2, 4)
    assert ms.parity == "zodd"
    for i, kxf in enumerate((0.2, 0.4)):
        want = ref["bands"][ref["kx"].index(kxf)][:2]
        got = [f for f in ms.all_freqs[i] if np.isfinite(f)]
        for wf in want:
            best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
            assert abs(best - wf) / wf <= 0.02, \
                f"kx={kxf}: band {wf:.4f} -> nearest {best:.4f} (got {got})"
    # gap bookkeeping: two k-points -> band_range_data + gap_list populated
    assert len(ms.band_range_data) >= 2
    assert ms.retrieve_gap(1) > 0         # TM gap between bands 1 and 2
    assert ms.gap_list and ms.gap_list[0][0] > 0


def test_bragg_1d_run_and_solve_kpoint_vs_mpb_oracle():
    """Gate: 1D sine-index Bragg through run() (polarizations degenerate in
    1D -> TM backend) with a callable default_material, plus solve_kpoint."""
    ref = json.load(open(DATA / "bragg_sine_mpb.json"))

    def eps_func(p):
        return Medium(index=ref["index_min"] + 0.5
                      * (ref["index_max"] - ref["index_min"])
                      * (1 + math.cos(2 * math.pi * p.x)))

    ms = ModeSolver(num_bands=2, resolution=32,
                    k_points=[Vector3(0.3)],
                    geometry_lattice=Lattice(size=Vector3(1)),
                    default_material=eps_func)
    ms.fmax = 0.8
    ms.t_end = 400.0
    ms.run()
    want = ref["bands"][ref["kx"].index(0.3)][:2]
    got = [f for f in ms.freqs if np.isfinite(f)]
    for wf in want:
        best = min(got, key=lambda g: abs(g - wf)) if got else 1e9
        assert abs(best - wf) / wf <= 0.02, \
            f"band {wf:.4f} -> nearest {best:.4f} (got {got})"
    assert ms.gap_list == []              # single k-point -> no gap scan
    # solve_kpoint alone updates freqs the same way
    before = list(ms.freqs)
    ms.solve_kpoint(Vector3(0.3))
    assert np.allclose(
        [f for f in ms.freqs if np.isfinite(f)],
        [f for f in before if np.isfinite(f)], rtol=1e-12)


def test_empty_lattice_3d_band():
    """3D no-parity smoke: vacuum cubic cell, k=(0.25,0,0) -> lowest folded
    empty-lattice band at f=0.25."""
    ms = ModeSolver(num_bands=1, resolution=8,
                    k_points=[Vector3(0.25)],
                    geometry_lattice=Lattice(size=Vector3(1, 1, 1)))
    ms.fmax = 0.4
    ms.t_end = 150.0
    ms.run()
    got = [f for f in ms.freqs if np.isfinite(f)]
    best = min(got, key=lambda g: abs(g - 0.25)) if got else 1e9
    assert abs(best - 0.25) <= 0.02 * 0.25, f"got {got}"


# --- MPBData --------------------------------------------------------------

def test_mpbdata_periods_tiles_input():
    rng = np.random.default_rng(7)
    eps = rng.uniform(1.0, 12.0, (8, 8))
    arr = MPBArray(eps, np.eye(3))
    out = MPBData(periods=2).convert(arr)
    assert out.shape == (16, 16)
    # output grid is grid-aligned with the input -> exact (rolled) tiling
    assert np.allclose(out[:8, :], out[8:, :])
    assert np.allclose(out[:, :8], out[:, 8:])
    assert np.allclose(np.sort(out.ravel()),
                       np.sort(np.tile(eps, (2, 2)).ravel()))


def test_mpbdata_rectify_constant_triangular():
    lat = np.array([[math.sqrt(3) / 2, 0.5, 0.0],
                    [math.sqrt(3) / 2, -0.5, 0.0],
                    [0.0, 0.0, 1.0]])
    arr = MPBArray(np.full((12, 12), 6.5), lat)
    md = MPBData(rectify=True, periods=2, resolution=24)
    out = md.convert(arr)
    assert out.ndim == 2 and min(out.shape) >= 24
    assert np.allclose(out, 6.5)


def test_mpbdata_requires_lattice():
    with pytest.raises(ValueError):
        MPBData(periods=2).convert(np.ones((4, 4)))


COMPUTE = r'''
import json
import math

import numpy as np


def compute(MPBData, Vector3):
    rng = np.random.default_rng(3)
    lat = [[math.sqrt(3) / 2, 0.5, 0.0],
           [math.sqrt(3) / 2, -0.5, 0.0],
           [0.0, 0.0, 1.0]]
    a_re = rng.standard_normal((12, 12))
    a_im = rng.standard_normal((12, 12))
    arr = a_re + 1j * a_im
    out = {}
    md = MPBData(lattice=np.array(lat), kpoint=Vector3(0.3, 0.4, 0.0),
                 rectify=True, resolution=24)
    c = md.convert(arr)
    out["rectify_c"] = [[[float(v.real), float(v.imag)] for v in row]
                        for row in c]
    md2 = MPBData(lattice=np.array(lat), periods=3)
    r = md2.convert(a_re)
    out["tile_r"] = [[float(v) for v in row] for row in r]
    return out
'''


@pytest.mark.skipif(not os.path.exists(ORACLE),
                    reason="meep-oracle env not available")
def test_mpbdata_parity_vs_meep_mpb():
    """Same convert() calls through real meep.mpb.MPBData (oracle env) and
    ours: rectified complex rank-2 with Bloch kvector + real 3-period tile."""
    script = COMPUTE + (
        "\nimport meep as mp\nfrom meep import mpb\n"
        "print('ORACLE_JSON=' + json.dumps(compute(mpb.MPBData, mp.Vector3)))\n")
    r = subprocess.run([ORACLE, "-c", script], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    line = [ln for ln in r.stdout.splitlines()
            if ln.startswith("ORACLE_JSON=")][-1]
    oracle = json.loads(line[len("ORACLE_JSON="):])

    env = {}
    exec(COMPUTE, env)
    ours = env["compute"](MPBData, Vector3)

    o = np.array(oracle["rectify_c"])
    u = np.array(ours["rectify_c"])
    assert o.shape == u.shape
    assert np.allclose(o, u, atol=1e-8)
    assert np.allclose(np.array(oracle["tile_r"]), np.array(ours["tile_r"]),
                       atol=1e-10)


def test_kottke_supersampled_matches_analytic_means():
    """A cell half-covered by an interface with normal x-hat must come back
    harmonic in xx and arithmetic in yy -- the whole point of the scheme.

    Also pins the coverage convention: the fraction must reference one
    phase across the grid, not each cell's own maximum. Per-cell referencing
    makes uniform inside and uniform outside cells both read 1.0, so the
    gradient (and with it every interface normal) vanishes.
    """
    from meep_torch.geometry.smoothing import blend_te_supersampled

    nx, ny, ss = 8, 8, 4
    fine = np.ones((nx * ss, ny * ss))
    fine[:14, :] = 12.0                       # cuts coarse column 3 in half
    exx, eyy = blend_te_supersampled(fine.reshape(nx, ss, ny, ss))
    assert exx[3, 0].item() == pytest.approx(1 / (0.5 / 12 + 0.5 / 1))
    assert eyy[3, 0].item() == pytest.approx(0.5 * 12 + 0.5 * 1)
    # uniform cells stay untouched on both components
    assert exx[0, 0].item() == pytest.approx(12.0)
    assert eyy[6, 0].item() == pytest.approx(1.0)


# --- per-band mode fields (mpb/fields.py) ---------------------------------

FIELD_RES = 16
FIELD_K = Vector3(0.3)


def _field_ms(parity="tm", **kw):
    """Square lattice of eps-12 rods, one k-point, solved."""
    ms = ModeSolver(num_bands=kw.pop("num_bands", 3), resolution=FIELD_RES,
                    k_points=[kw.pop("k", FIELD_K)],
                    geometry_lattice=Lattice(size=Vector3(1, 1)),
                    geometry=[Cylinder(0.2, material=Medium(epsilon=12))],
                    **kw)
    (ms.run_tm if parity == "tm" else ms.run_te)()
    return ms


def _dA(ms):
    _, d = ms._eps
    return float(d) ** 2


def test_tm_efield_shape_normalisation_and_faraday():
    """The gate that actually proves the eigenvector is right.

    Normalisation alone proves nothing -- any field scaled to 1 passes it,
    including one recovered wrongly from the B = D^-1 A D^-1 transform.
    The Faraday identity does: because H is built with the SAME difference
    operators the eigensolve used, the eigen-equation forces
    integral eps|E|^2 == integral |H|^2 exactly, so a mis-scaled or
    mis-derived field breaks it immediately.
    """
    for differentiable in (False, True):
        ms = ModeSolver(num_bands=3, resolution=FIELD_RES,
                        k_points=[FIELD_K],
                        geometry_lattice=Lattice(size=Vector3(1, 1)),
                        geometry=[Cylinder(0.2, material=Medium(epsilon=12))])
        ms.differentiable = differentiable       # class attribute, not a kwarg
        ms.run_tm()
        E = ms.get_efield(2)
        assert E.shape == (FIELD_RES, FIELD_RES, 3)
        assert E.dtype == np.complex128
        assert np.abs(E[..., 0]).max() == 0.0    # TM carries Ez only
        assert np.abs(E[..., 1]).max() == 0.0
        eps = ms.get_epsilon()
        e_energy = float((eps * np.abs(E[..., 2]) ** 2).sum()) * _dA(ms)
        assert e_energy == pytest.approx(1.0, abs=1e-10)

        H = ms.get_hfield(2)
        assert np.abs(H[..., 2]).max() == 0.0    # ... and Hx, Hy only
        h_energy = float((np.abs(H[..., 0]) ** 2
                          + np.abs(H[..., 1]) ** 2).sum()) * _dA(ms)
        assert h_energy == pytest.approx(e_energy, rel=1e-8), (
            f"Faraday residual: eps|E|^2 {e_energy} vs |H|^2 {h_energy} "
            f"(differentiable={differentiable})")


def test_te_hfield_is_the_mirror_of_tm():
    ms = _field_ms("te")
    H = ms.get_hfield(2)
    assert H.shape == (FIELD_RES, FIELD_RES, 3)
    assert np.abs(H[..., 0]).max() == 0.0 and np.abs(H[..., 1]).max() == 0.0
    assert float((np.abs(H[..., 2]) ** 2).sum()) * _dA(ms) == pytest.approx(
        1.0, abs=1e-10)
    E = ms.get_efield(2)
    assert np.abs(E[..., 2]).max() == 0.0        # TE carries Ex, Ey only
    assert np.abs(E[..., 0]).max() > 0.0


def test_bloch_phase_flag_is_exactly_the_phase_factor():
    ms = _field_ms("tm")
    full = np.asarray(ms.get_efield(2, True))
    env = np.asarray(ms.get_efield(2, False))
    _, d = ms._eps
    d = float(d)
    x = (np.arange(FIELD_RES) + 0.5) * d - FIELD_RES * d / 2.0
    ph = np.exp(1j * 2 * math.pi * FIELD_K.x * x)[:, None]
    assert np.abs(full[..., 2] - env[..., 2] * ph).max() < 1e-12


def test_compute_field_energy_totals_and_fractions():
    ms = _field_ms("tm")
    ms.get_dfield(2)
    out = ms.compute_field_energy()
    assert len(out) == 7
    assert out[0] == pytest.approx(1.0, abs=1e-8)
    assert sum(out[1:]) == pytest.approx(1.0, abs=1e-8)
    # TM D is z-only, so all of it sits in the z pair
    assert out[5] + out[6] == pytest.approx(1.0, abs=1e-8)
    ms.get_bfield(2)
    b = ms.compute_field_energy()
    assert b[0] == pytest.approx(1.0, abs=1e-8)
    assert b[5] + b[6] == pytest.approx(0.0, abs=1e-12)   # ... B is in-plane


def test_compute_energy_in_objects():
    ms = _field_ms("tm")
    ms.get_dfield(1)
    ms.compute_field_energy()
    # band 1 is the dielectric band: most of its energy sits in the rod
    inside = ms.compute_energy_in_objects([Cylinder(0.2, material=Medium(epsilon=12))])
    assert 0.5 < inside < 1.0
    # an object covering the whole cell collects all of it
    whole = ms.compute_energy_in_objects([Cylinder(5.0, material=Medium(epsilon=12))])
    assert whole == pytest.approx(1.0, abs=1e-8)
    assert ms.compute_energy_in_objects([]) == 0.0


def test_targeted_solver_finds_the_defect_band():
    """MPB's targeted eigensolver: one band near a frequency, instead of
    every band below it. The 5x5 supercell from the tutorial.

    Gated against tests/examples/data/defect_mpb.json, generated in the
    meep-oracle env by generate_defect_mpb.py (2026-09-07):
    defect band 25 = 0.3773004164996916, energy in a radius-1 cylinder =
    0.6227672211202755, targeted = 0.37730041137599807. This engine runs
    ~0.6% below on both, which is the 2nd-order FDFD grid against MPB's
    planewave basis, so the oracle gate is 1% and the EXACT gates are the
    internal ones (targeted == band 25 of the full run, mode inside the
    gap this same engine computes).

    The fixture also pins the geometry-precedence rule: the defect is an
    air rod APPENDED over a lattice duplicate, so it only exists if later
    objects win. With the old reversed order this test would read
    0.2792 -- 26% off, not 0.6%.
    """
    from meep_torch.mp.functions import geometric_objects_lattice_duplicates

    air = Medium(epsilon=1.0)

    def _defect(num_bands, target_freq):
        lat = Lattice(size=Vector3(5, 5))
        geom = geometric_objects_lattice_duplicates(
            lat, [Cylinder(0.2, material=Medium(epsilon=12))])
        geom.append(Cylinder(0.2, material=air))
        ms = ModeSolver(num_bands=num_bands, resolution=FIELD_RES,
                        k_points=[Vector3(0.5, 0.5)], geometry_lattice=lat,
                        geometry=geom, target_freq=target_freq,
                        tolerance=1e-8)
        ms.run_tm()
        return ms

    full = _defect(50, 0.0)
    targeted = _defect(1, (0.2812 + 0.4174) / 2)
    f_defect = targeted.get_freqs()[0]
    # the perfect lattice's TM gap, from the same engine
    perfect = ModeSolver(num_bands=2, resolution=FIELD_RES,
                         k_points=[Vector3(), Vector3(0.5), Vector3(0.5, 0.5)],
                         geometry_lattice=Lattice(size=Vector3(1, 1)),
                         geometry=[Cylinder(0.2, material=Medium(epsilon=12))])
    perfect.run_tm()
    lo = max(perfect.all_freqs[:, 0])
    hi = min(perfect.all_freqs[:, 1])
    assert lo < f_defect < hi, f"defect {f_defect} outside gap ({lo}, {hi})"
    # and it is the same mode the full 50-band run puts in the gap
    in_gap = [f for f in full.get_freqs() if lo < f < hi]
    assert len(in_gap) == 1
    assert f_defect == pytest.approx(in_gap[0], abs=1e-10)

    targeted.get_dfield(1)
    targeted.compute_field_energy()
    energy = targeted.compute_energy_in_objects(
        [Cylinder(1.0, material=Medium(epsilon=1.0))])

    ref = json.load(open(DATA / "defect_mpb.json"))
    assert f_defect == pytest.approx(ref["defect_freq"], rel=0.01)
    assert energy == pytest.approx(ref["energy_in_cylinder_r1"], rel=0.01)
    assert ref["gap_lo"] < f_defect < ref["gap_hi"]


def test_output_files_match_meep_names(tmp_path, monkeypatch):
    import h5py
    monkeypatch.chdir(tmp_path)
    ms = _field_ms("tm", num_bands=1)
    ms.filename_prefix = "t"
    from meep_torch import mpb as _mpb
    _mpb.output_efield_z(ms, 1)
    f = tmp_path / "t-e.k01.b01.z.tm.h5"
    assert f.exists(), sorted(p.name for p in tmp_path.iterdir())
    with h5py.File(f) as h:
        assert h["z.r"].shape == (FIELD_RES, FIELD_RES)
        assert h["z.i"].shape == (FIELD_RES, FIELD_RES)
        assert h["lattice vectors"].shape == (3, 3)
        assert "Bloch wavevector" in h and "description" in h
    _mpb.output_dpwr(ms, 1)
    g = tmp_path / "t-dpwr.k01.b01.tm.h5"
    assert g.exists()
    with h5py.File(g) as h:
        assert h["data"].shape == (FIELD_RES, FIELD_RES)
        assert float(np.asarray(h["data"]).sum()) * _dA(ms) == pytest.approx(
            1.0, abs=1e-8)
    ms.output_epsilon()
    e = tmp_path / "t-epsilon.h5"          # no parity suffix on epsilon
    assert e.exists()
    with h5py.File(e) as h:
        assert h["epsilon.xx"].shape == (FIELD_RES, FIELD_RES)
        assert np.allclose(np.asarray(h["epsilon_inverse.zz"]),
                           1.0 / np.asarray(h["data"]))


def test_fix_field_phase_is_canonical():
    ms = _field_ms("tm")
    ms.get_efield(2, False)
    a = ms.fix_field_phase().copy()
    # re-fetching and re-fixing reproduces the same phase, and starting
    # from an arbitrarily rotated field lands on the same answer
    ms.get_efield(2, False)
    ms._curfield = ms._curfield * np.exp(1j * 0.7)
    b = ms.fix_field_phase()
    assert np.abs(a - b).max() < 1e-10
    # MPB picks the phase that maximises the sum of squares of the real
    # parts. That does NOT make the field real -- a Bloch envelope at
    # k != 0 cannot be -- but it does make the choice stationary, which is
    # sum(2 Re Im) == 0. That is the invariant worth asserting.
    f = a.reshape(-1)
    scale = float((np.abs(f) ** 2).sum())
    assert abs(float((2 * f.real * f.imag).sum())) < 1e-10 * scale
    # and applying it again is a no-op
    ms._curfield = a.copy()
    ms._curfield_type = "e"
    assert np.abs(ms.fix_field_phase() - a).max() < 1e-12
