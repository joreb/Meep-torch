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
    with pytest.raises(NotImplementedError):
        ms.get_efield(1)                  # band fields not exported yet


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
    eps_np = 1.0 + rod.numpy() * 11.0
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
