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

import meep_torch.mpb as mp_mpb
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
    # 2D no-parity is backed now (the union of TM and TE); the y-parities
    # still are not, because they need a mirror-reduced cell.
    for p in (mp_mpb.EVEN_Y, mp_mpb.ODD_Y, mp_mpb.EVEN_Y + mp_mpb.EVEN_Z):
        with pytest.raises(NotImplementedError):
            ms.run_parity(p, True)
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
    assert ms.parity == "te"   # MPB nz==1 rule: 2D says tm/te
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


TRI = Lattice(size=Vector3(1, 1),
              basis1=Vector3(math.sqrt(3) / 2, 0.5),
              basis2=Vector3(math.sqrt(3) / 2, -0.5))


def test_oblique_triangular_bands_vs_mpb():
    """Triangular lattice through the oblique basis, against meep.mpb.

    This used to be test_oblique_lattice_raises: the cell was rasterized on
    a cartesian box, which solves a SHEARED structure (K band 1 came back
    27.75% off), so it refused rather than return plausible wrong numbers.
    The operator now works in lattice coordinates and carries the metric
    g_ij = b_i . b_j, whose off-diagonal term is exactly what a rectangular
    cell does not have.

    Gate is 2% like the sibling band tests; measured 0.54% worst case over
    Gamma/M/K on 4 bands at resolution 32.
    """
    ref = json.load(open(DATA / "tri_rods_mpb.json"))
    names = ["gamma", "M", "K"]
    kpts = [Vector3(), Vector3(y=0.5), Vector3(-1 / 3, 1 / 3)]
    ms = ModeSolver(num_bands=4, resolution=32, geometry_lattice=TRI,
                    k_points=kpts,
                    geometry=[Cylinder(ref["r"],
                                       material=Medium(epsilon=ref["eps"]))])
    ms.run_tm()
    worst = 0.0
    for i, name in enumerate(names):
        for got, want in zip(ms.all_freqs[i], ref[name]):
            if want == 0.0:
                assert got == 0.0          # the Gamma zero mode
                continue
            worst = max(worst, abs(got - want) / want)
    print(f"triangular TM vs mpb: worst {worst:.4%}")
    assert worst < 0.02, f"triangular bands off by {worst:.2%}"


def test_oblique_and_rectangular_supercell_agree():
    """Same physics, two grids: the oblique primitive cell against the
    sqrt(3) x 1 rectangular supercell that tests/examples/test_mpb_track.py
    has always used.

    The supercell holds two rods (corner and centre) and is a different
    discretization of the same crystal, so agreement here is independent
    evidence that the metric terms are right -- a wrong g12 would move the
    oblique run and leave the supercell untouched.

    The supercell contains TWO primitive cells, so its spectrum is a
    FOLDED SUPERSET: supercell band n is not primitive band n, and every
    primitive frequency must instead APPEAR somewhere in the supercell
    list. tests/examples/test_mpb_track.py matches the same way, and
    comparing index-wise here read 0.274 against 0.192 for band 1 purely
    from that.
    """
    r, eps_rod, res = 0.2, 12.0, 32
    obl = ModeSolver(num_bands=3, resolution=res, geometry_lattice=TRI,
                     k_points=[Vector3(-1 / 3, 1 / 3)],
                     geometry=[Cylinder(r, material=Medium(epsilon=eps_rod))])
    obl.run_tm()

    # K = (-1/3, 1/3) in the triangular reciprocal basis is ky = 2 pi/3
    # in the rectangular supercell's cartesian frame (test_mpb_track).
    from meep_torch.mpb.eigensolver import bloch_bands_2d
    nx, ny = int(round(math.sqrt(3) * res)), res
    Lx, dx = nx / res, 1.0 / res
    xs = (np.arange(nx * 4) + 0.5) / (4 * res)
    ys = (np.arange(ny * 4) + 0.5) / (4 * res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    fine = np.ones((nx * 4, ny * 4))
    for cx, cy in ((0.0, 0.0), (Lx / 2, 0.5)):
        ddx = (X - cx) - Lx * np.round((X - cx) / Lx)
        ddy = (Y - cy) - 1.0 * np.round((Y - cy) / 1.0)
        fine[ddx ** 2 + ddy ** 2 < r ** 2] = eps_rod
    eps = fine.reshape(nx, 4, ny, 4).mean(axis=(1, 3))
    rect = bloch_bands_2d(eps, dx, dx, 0.0, 2 * math.pi / 3, 8, "tm")

    for b, o in enumerate(obl.get_freqs()):
        near = min(rect, key=lambda q: abs(q - o))
        rel = abs(near - o) / o if o else abs(near - o)
        print(f"K band {b + 1}: oblique {o:.6f}  nearest supercell "
              f"{near:.6f}  rel {rel:.4%}")
        # 1% covers the two grids being genuinely different: the supercell
        # is sqrt(3)*res rounded to an integer, so its cell is 0.8% short
        # of sqrt(3) at res=32, which shifts its bands on its own.
        assert rel < 0.01, f"band {b + 1}: {o} vs nearest {near}"


def test_oblique_rasterizer_coverage_fraction():
    """A rod of radius r fills pi r^2 / |a1 x a2| of the triangular cell.

    The rasterizer samples on the rhombus in fractional coordinates and
    wraps min-image through the lattice; a cartesian box or a cartesian
    wrap both get this wrong.
    """
    r, eps_rod = 0.2, 12.0
    ms = ModeSolver(num_bands=1, resolution=32, geometry_lattice=TRI,
                    k_points=[Vector3(0.1)],
                    geometry=[Cylinder(r, material=Medium(epsilon=eps_rod))])
    ms.init_params(ODD_Z, True)
    eps = ms._eps[0].cpu().numpy()
    frac = (eps.mean() - 1.0) / (eps_rod - 1.0)
    want = math.pi * r ** 2 / (math.sqrt(3) / 2)
    print(f"rod coverage {frac:.6f} vs pi r^2 / area {want:.6f}")
    assert abs(frac - want) < 1e-3


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
    assert ms.parity == "tm"   # MPB nz==1 rule: 2D says tm/te
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
        assert E.shape == (FIELD_RES, FIELD_RES, 1, 3)   # meep pads dims to 3, then + 3
        assert E.dtype == np.complex128
        assert np.abs(E[..., 0, 0]).max() == 0.0    # TM carries Ez only
        assert np.abs(E[..., 0, 1]).max() == 0.0
        eps = ms.get_epsilon()
        e_energy = float((eps * np.abs(E[..., 0, 2]) ** 2).sum()) * _dA(ms)
        assert e_energy == pytest.approx(1.0, abs=1e-10)

        H = ms.get_hfield(2)
        assert np.abs(H[..., 0, 2]).max() == 0.0    # ... and Hx, Hy only
        h_energy = float((np.abs(H[..., 0, 0]) ** 2
                          + np.abs(H[..., 0, 1]) ** 2).sum()) * _dA(ms)
        assert h_energy == pytest.approx(e_energy, rel=1e-8), (
            f"Faraday residual: eps|E|^2 {e_energy} vs |H|^2 {h_energy} "
            f"(differentiable={differentiable})")


def test_te_hfield_is_the_mirror_of_tm():
    ms = _field_ms("te")
    H = ms.get_hfield(2)
    assert H.shape == (FIELD_RES, FIELD_RES, 1, 3)
    assert np.abs(H[..., 0, 0]).max() == 0.0 and np.abs(H[..., 0, 1]).max() == 0.0
    assert float((np.abs(H[..., 0, 2]) ** 2).sum()) * _dA(ms) == pytest.approx(
        1.0, abs=1e-10)
    E = ms.get_efield(2)
    assert np.abs(E[..., 0, 2]).max() == 0.0        # TE carries Ex, Ey only
    assert np.abs(E[..., 0, 0]).max() > 0.0


def test_bloch_phase_flag_is_exactly_the_phase_factor():
    ms = _field_ms("tm")
    full = np.asarray(ms.get_efield(2, True))
    env = np.asarray(ms.get_efield(2, False))
    _, d = ms._eps
    d = float(d)
    x = (np.arange(FIELD_RES) + 0.5) * d - FIELD_RES * d / 2.0
    ph = np.exp(1j * 2 * math.pi * FIELD_K.x * x)[:, None]
    assert np.abs(full[..., 0, 2] - env[..., 0, 2] * ph).max() < 1e-12


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


def _to_cell_centres(x):
    """MPB's node samples averaged onto this engine's cell centres.

    Our sample i sits half a cell above MPB's sample i on each axis, so the
    matching value is the mean of MPB's four surrounding samples. Periodic
    wrap, like the cell.
    """
    return 0.25 * (x + np.roll(x, -1, 0) + np.roll(x, -1, 1)
                   + np.roll(np.roll(x, -1, 0), -1, 1))


def test_h5_output_matches_meep_byte_for_byte_metadata(tmp_path, monkeypatch):
    """Diff our HDF5 output against meep's OWN files for the same mode.

    tests/examples/data/defect-{e.k01.b25.z.tm,dpwr.k01.b25}.h5 were
    written by real meep.mpb (generate_defect_mpb.py, 2026-09-07). The
    metadata is exact parity surface -- dataset names, the description
    strings, and "Bloch wavevector", which MPB documents as being in the
    RECIPROCAL LATTICE BASIS ("real output_k[3]; /* kvector in reciprocal
    lattice basis */", mpb/fields.c). Writing it in cartesian put 0.1 in
    the file where meep has 0.5 on this 5x5 cell, and no test saw it.

    Field VALUES only get a correlation gate: 2nd-order FDFD against a
    planewave basis, and each solver picks its own global phase.

    They are also compared on a COMMON grid. MPB samples epsilon and
    fields at ``i*d - L/2`` (mpb/fields.c, the energy loop); this engine
    samples cell centres ``(i+0.5)*d - L/2``. Index i is therefore not the
    same physical point in the two files, and the offset is systematic.
    Measured on this fixture, raw against half-cell aligned:

        Ez        0.995976 -> 0.999937
        |Ez|^2    0.995949 -> 0.999960
        eps|Ez|^2 0.950569 -> 0.992274

    The energy density is the one that exposes it, because it peaks at the
    rod boundary where a half-cell shift bites hardest, while Ez itself is
    smooth enough to survive one. Aligning first turns a vague 0.95 into a
    0.99 gate that would actually catch a wrong field.
    """
    import h5py
    from meep_torch.mp.functions import geometric_objects_lattice_duplicates
    from meep_torch import mpb as _mpb

    lat = Lattice(size=Vector3(5, 5))
    geom = geometric_objects_lattice_duplicates(
        lat, [Cylinder(0.2, material=Medium(epsilon=12))])
    geom.append(Cylinder(0.2, material=Medium(epsilon=1.0)))
    ms = ModeSolver(num_bands=50, resolution=16, geometry_lattice=lat,
                    geometry=geom, k_points=[Vector3(0.5, 0.5)])
    ms.filename_prefix = "defect"
    monkeypatch.chdir(tmp_path)
    ms.run_tm()
    _mpb.output_efield_z(ms, 25)
    _mpb.output_dpwr(ms, 25)

    for name in ("e.k01.b25.z.tm", "dpwr.k01.b25.tm"):
        ours = tmp_path / f"defect-{name}.h5"
        ref = DATA / f"defect-{name}.h5"
        assert ours.exists(), sorted(p.name for p in tmp_path.iterdir())
        with h5py.File(ours) as a, h5py.File(ref) as b:
            assert sorted(a.keys()) == sorted(b.keys()), name
            # the description embeds the mode frequency, and ours differs
            # from meep's by the 0.6% discretization gap -- compare
            # everything up to it, which is what carries the format
            da = a["description"][()].decode().split(", freq=")
            db = b["description"][()].decode().split(", freq=")
            assert da[0] == db[0], f"{name}: {da[0]!r} != {db[0]!r}"
            assert len(da) == len(db) == 2
            np.testing.assert_allclose(a["lattice vectors"][()],
                                       b["lattice vectors"][()], atol=1e-12)
            if "Bloch wavevector" in b:
                np.testing.assert_allclose(a["Bloch wavevector"][()],
                                           b["Bloch wavevector"][()],
                                           atol=1e-12)
            if "z.r" in b:
                fa = a["z.r"][()] + 1j * a["z.i"][()]
                fb = b["z.r"][()] + 1j * b["z.i"][()]
                assert fa.shape == fb.shape
                fb = _to_cell_centres(fb)
                corr = abs(np.vdot(fb, fa)) / (np.linalg.norm(fa)
                                               * np.linalg.norm(fb))
                assert corr > 0.999, f"Ez pattern correlation {corr:.5f}"
            else:
                da, db = a["data"][()], b["data"][()]
                assert da.shape == db.shape
                assert float(da.sum()) * (5.0 / 80) ** 2 == pytest.approx(
                    1.0, abs=1e-8)
                db = _to_cell_centres(db)
                corr = float((da * db).sum()
                             / (np.linalg.norm(da) * np.linalg.norm(db)))
                assert corr > 0.99, f"energy density correlation {corr:.5f}"


def test_fix_field_phase_rotates_the_eigenvector():
    """MPB rotates the eigenvector, so a later fetch of ANY field of that
    band carries the canonical phase. Rotating only the loaded curfield
    made combine_band_functions(fix_efield_phase, output_efield_z) write an
    unfixed field -- the fix landed on a curfield the next get_efield
    immediately overwrote."""
    ms = _field_ms("tm")
    before = np.asarray(ms.get_efield(2, False)).copy()
    ms.get_efield(2, False)
    fixed = np.asarray(ms.fix_field_phase()).copy()
    after = np.asarray(ms.get_efield(2, False))
    assert np.abs(after - fixed.reshape(after.shape)).max() < 1e-12, (
        "re-fetching after fix_field_phase did not carry the phase")
    # and it really did change something
    assert np.abs(after - before.reshape(after.shape)).max() > 1e-9


def test_gamma_zero_mode_is_exactly_zero():
    """The omega = 0 TM mode at Gamma must be EXACTLY 0.0, not 1e-8.

    The eigensolver puts it anywhere in +-1e-15 of the spectrum scale;
    clamping only negatives let +5.3e-15 through as f = 1.2e-8, which is
    nonzero, so every `omega == 0` guard downstream silently stopped
    firing and band 1's H field became curl(const)/roundoff.
    """
    ms = _field_ms("tm", k=Vector3())
    assert ms.get_freqs()[0] == 0.0
    H = ms.get_hfield(1)
    assert np.abs(H).max() == 0.0
    ms.get_dfield(1)
    assert ms.compute_field_energy()[0] == pytest.approx(1.0, abs=1e-8)


def test_allow_negative_epsilon_does_not_become_a_target():
    """allow_negative_epsilon() sets target_freq = 1/inf, and the solver
    must still return the lowest bands rather than shift-inverting at
    sigma ~ 0 on the singular Gamma operator.

    Reviewed as a live bug on the premise that 1/inf is a denormal 1e-20
    and therefore truthy; in Python it is 0.0 exactly, so the original
    truthiness test was already correct. The guard is now a threshold,
    which is what the code means, and this pins the behaviour either way.
    """
    ms = ModeSolver(num_bands=3, resolution=FIELD_RES, k_points=[Vector3()],
                    geometry_lattice=Lattice(size=Vector3(1, 1)),
                    geometry=[Cylinder(0.2, material=Medium(epsilon=12))])
    ms.allow_negative_epsilon()
    assert ms.target_freq == 0.0        # 1/inf, not a denormal
    ms.run_tm()
    assert ms.get_freqs()[0] == 0.0
    assert ms.get_freqs()[1] > 0.1


def test_freqs_keep_autograd_through_modesolver():
    """differentiable=True has to survive ModeSolver, not just the direct
    bloch_bands_2d call: ms.freqs is the only handle an optimiser has."""
    import torch
    eps = torch.tensor(12.0, dtype=torch.float64, requires_grad=True)
    ms = ModeSolver(num_bands=2, resolution=8, k_points=[Vector3(0.3)],
                    geometry_lattice=Lattice(size=Vector3(1, 1)),
                    geometry=[Cylinder(0.2, material=Medium(epsilon=12))])
    ms.differentiable = True
    ms.run_tm()
    f = ms.freqs[1]
    assert isinstance(f, torch.Tensor), type(f)
    # NOT asserting requires_grad here: _rasterize_epsilon builds the eps
    # tensor from geometry floats, so there is no upstream graph for a
    # plain ModeSolver run to carry. Returning the tensor is what lets a
    # caller attach one (assign a leaf tensor to ms._eps); listifying it
    # made that impossible, which is the regression this pins.
    assert torch.is_floating_point(f)


def test_solve_kpoint_resets_the_loaded_field():
    """MPB's solve_kpoint ends with curfield_reset. Without it a density
    computed at k=N stays loaded and gets written under k=N+1's name."""
    ms = _field_ms("tm")
    ms.get_dfield(2)
    ms.compute_field_energy()
    assert ms._curfield is not None
    ms.solve_kpoint(Vector3(0.4))
    assert ms._curfield is None and ms._curfield_type is None


def test_later_geometry_objects_win_in_1d_too():
    """The precedence fix (d7456a0) also moves the 1D/3D harminv
    rasterization; that path has its own fixtures and none of them
    overlapped either."""
    lat = Lattice(size=Vector3(1, 0, 0))
    over = ModeSolver(num_bands=1, resolution=32, k_points=[Vector3(0.2)],
                      geometry_lattice=lat,
                      geometry=[Cylinder(0.4, material=Medium(epsilon=12)),
                                Cylinder(0.4, material=Medium(epsilon=2))])
    plain = ModeSolver(num_bands=1, resolution=32, k_points=[Vector3(0.2)],
                       geometry_lattice=lat,
                       geometry=[Cylinder(0.4, material=Medium(epsilon=2))])
    over.init_params(ODD_Z, True)
    plain.init_params(ODD_Z, True)
    np.testing.assert_allclose(over._eps[0].cpu().numpy(),
                               plain._eps[0].cpu().numpy(), atol=1e-12)


def test_oblique_triangular_te_bands_vs_mpb():
    """TE on the triangular lattice -- the case that leans hardest on the
    metric CROSS term.

    TM only needs Gx^H Gx + Gy^H Gy, where the metric is already inside the
    gradients. TE carries a 1/eps weight per edge family, and a mixed u-v
    derivative whose weight lives on the cell CORNER; writing it as
    Gx^H Wx Gx + Gy^H Wy Gy instead would be silently wrong off-rectangle,
    because Gx mixes both edge families and one weight cannot sit on both.

    Gate is 4%, looser than TM's 2%, for the reason the square-lattice TE
    gate is already 2.6%: in-plane E is discontinuous across the rod
    boundary, so the interface treatment dominates. Oblique cells also
    fall back to SCALAR eps -- Kottke blends against the cartesian axes and
    is not meaningful when the cell axes are not those. Measured 2.62%.
    """
    ref = json.load(open(DATA / "tri_te_mpb.json"))
    kpts = [Vector3(), Vector3(y=0.5), Vector3(-1 / 3, 1 / 3)]
    ms = ModeSolver(num_bands=ref["num_bands"], resolution=ref["resolution"],
                    geometry_lattice=TRI, k_points=kpts,
                    geometry=[Cylinder(ref["r"],
                                       material=Medium(epsilon=ref["eps"]))])
    ms.run_te()
    worst = 0.0
    for i, name in enumerate(("gamma", "M", "K")):
        for got, want in zip(ms.all_freqs[i], ref[name]):
            if want == 0.0:
                continue
            worst = max(worst, abs(got - want) / want)
    print(f"triangular TE vs mpb: worst {worst:.4%}")
    assert worst < 0.04, f"triangular TE bands off by {worst:.2%}"


def test_oblique_fields_normalise_and_satisfy_faraday():
    """The field machinery on an oblique cell: same identity as the
    rectangular case, now with the cell-area measure and the metric
    gradients. If the measure were still dx*dy the energies would come out
    wrong by |a1 x a2| / (dx dy) = 0.866."""
    ms = ModeSolver(num_bands=3, resolution=24, geometry_lattice=TRI,
                    k_points=[Vector3(-1 / 3, 1 / 3)],
                    geometry=[Cylinder(0.2, material=Medium(epsilon=12))])
    ms.run_tm()
    E = ms.get_efield(2)
    assert E.shape == (24, 24, 1, 3)
    ms.get_dfield(2)
    d = ms.compute_field_energy()
    assert d[0] == pytest.approx(1.0, abs=1e-8)
    ms.get_bfield(2)
    b = ms.compute_field_energy()
    assert b[0] == pytest.approx(1.0, abs=1e-8), (
        "Faraday identity broken on the oblique cell: eps|E|^2 and |H|^2 "
        "must both integrate to 1 from the single normalisation")
    # a rod covers pi r^2 / |a1 x a2| of the cell; the dielectric band
    # concentrates in it
    ms.get_dfield(1)
    ms.compute_field_energy()
    inside = ms.compute_energy_in_objects(
        [Cylinder(0.2, material=Medium(epsilon=12))])
    assert 0.5 < inside < 1.0, inside


def test_object_centres_are_lattice_coordinates():
    """MPB reads a geometric object's `center` in LATTICE coordinates.

    Gated on the honeycomb, which is the cheapest structure that can tell
    the two readings apart: two rods at +-(1/6, 1/6) on the triangular
    basis. Under the cartesian reading they land at +-(0.1667, 0.1667)
    instead of +-(0.2887, 0) and the result is a different crystal that
    still solves cleanly -- which is exactly why nothing caught it. Every
    other fixture in the repo puts its objects at the origin, or on an
    identity basis where the two readings coincide.

    Measured against honey_rods_mpb.json: 0.74% worst at resolution 32,
    the same order as the other oblique TM gates.

    The mapping uses the UNIT basis vectors and does NOT scale with
    Lattice.size -- checked against meep.mpb on size=(2,2) with an
    identity basis, where center=(0.25, 0) stays at cartesian (0.25, 0).
    """
    ref = json.load(open(DATA / "honey_rods_mpb.json"))
    rod = lambda c: Cylinder(ref["r"], center=c,
                             material=Medium(epsilon=ref["eps"]))
    ms = ModeSolver(geometry_lattice=TRI, resolution=32, num_bands=4,
                    k_points=[Vector3()],
                    geometry=[rod(Vector3(1 / 6, 1 / 6)),
                              rod(Vector3(-1 / 6, -1 / 6))])
    ms.run_tm()
    got, want = list(ms.get_freqs()), ref["gamma"][:4]
    worst = max(abs(g - w) / w for g, w in zip(got, want) if w > 1e-9)
    print(f"honeycomb (off-origin rods, oblique basis): worst {worst:.4%}")
    assert worst < 0.02, f"honeycomb bands off by {worst:.2%}"

    # and the size scaling: an identity basis must be untouched by the map
    sq = ModeSolver(geometry_lattice=Lattice(size=Vector3(2, 2)),
                    resolution=16, num_bands=1, k_points=[Vector3(0.1)],
                    geometry=[Cylinder(0.2, center=Vector3(0.25, 0),
                                       material=Medium(epsilon=12))])
    sq.init_params(ODD_Z, True)
    eps = np.asarray(sq.get_epsilon())
    nu = eps.shape[0]
    # the rod centre sits a quarter of a UNIT basis vector from the origin,
    # i.e. at cell fraction 0.25/2 = 0.125, not 0.25
    col = int(round((0.125 + 0.5) * nu))
    assert eps[col, nu // 2] > 6.0, "size=2 cell: centre scaled with size"
