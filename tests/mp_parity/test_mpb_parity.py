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
        ms.run_te()                       # TE Bloch stepper not available
    with pytest.raises(NotImplementedError):
        ms.run()                          # 2D no-parity needs TE + TM
    with pytest.raises(NotImplementedError):
        ms.get_efield(1)                  # band fields need an eigensolver


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
