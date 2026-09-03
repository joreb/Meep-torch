# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""meep tutorial compatibility: step-function combinator x output-wrapper
composition (meep's f(sim, todo) protocol), the FluxData normalization
workflow (get_flux_data / load_flux_data / load_minus_flux_data), and the two
upstream tutorials (bent-waveguide, bend-flux) run verbatim as subprocesses
with only the import swapped (tutorials/)."""
import glob
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import meep_torch.mp as mp

FCEN, DF = 0.5, 0.2
TUTORIALS = pathlib.Path(__file__).with_name("tutorials")


def _sim(cell=6):
    return mp.Simulation(cell_size=mp.Vector3(cell, cell, 0), resolution=10,
                         sources=[mp.Source(mp.GaussianSource(frequency=FCEN,
                                                              fwidth=DF),
                                            component=mp.Ez,
                                            center=mp.Vector3(-1, 0))],
                         boundary_layers=[mp.PML(1.0)])


def _probe(rec):
    """1-arg output-function stand-in that records the wrapper context it
    fires under: (round_time, to_appended active, in_volume active, prefix)."""
    def probe(sim):
        rec.append((sim.round_time(),
                    getattr(sim, "_append_target", None) is not None,
                    getattr(sim, "_output_volume", None) is not None,
                    getattr(sim, "filename_prefix", None) or ""))
    return probe


# wrapper name -> (wrap(fn), context check on one probe record)
WRAPPERS = {
    "to_appended": (lambda f: mp.to_appended("mat", f), lambda r: r[1]),
    "in_volume": (lambda f: mp.in_volume(
        mp.Volume(center=mp.Vector3(), size=mp.Vector3(2, 2)), f),
        lambda r: r[2]),
    "in_point": (lambda f: mp.in_point(mp.Vector3(0.5, 0.5), f),
                 lambda r: r[2]),
    "with_prefix": (lambda f: mp.with_prefix("pre-", f),
                    lambda r: r[3].startswith("pre-")),
}

# combinator name -> (make(fn), needs source cutoff in the run window,
#                     times check(times, dt, tend, cutoff))
COMBINATORS = {
    "at_beginning": (lambda f: mp.at_beginning(f), False,
                     lambda ts, dt, tend, cut:
                     len(ts) == 1 and ts[0] <= dt + 1e-9),
    "at_end": (lambda f: mp.at_end(f), False,
               lambda ts, dt, tend, cut:
               len(ts) == 1 and abs(ts[0] - tend) <= 1e-9),
    "at_time": (lambda f: mp.at_time(2.0, f), False,
                lambda ts, dt, tend, cut:
                len(ts) == 1 and 2.0 - 1e-9 <= ts[0] <= 2.0 + 2 * dt),
    "at_every": (lambda f: mp.at_every(1.0, f), False,
                 lambda ts, dt, tend, cut:
                 4 <= len(ts) <= 6 and all(
                     min(t % 1.0, 1.0 - t % 1.0) <= dt + 1e-9 for t in ts)),
    "after_time": (lambda f: mp.after_time(2.0, f), False,
                   lambda ts, dt, tend, cut:
                   bool(ts) and min(ts) >= 2.0 - dt - 1e-9
                   and max(ts) >= tend - dt - 1e-9),
    "before_time": (lambda f: mp.before_time(2.0, f), False,
                    lambda ts, dt, tend, cut:
                    bool(ts) and max(ts) <= 2.0 + dt + 1e-9),
    "after_sources": (lambda f: mp.after_sources(f), True,
                      lambda ts, dt, tend, cut:
                      bool(ts) and min(ts) >= cut - dt - 1e-9),
    "during_sources": (lambda f: mp.during_sources(f), True,
                       lambda ts, dt, tend, cut:
                       bool(ts) and max(ts) <= cut + dt + 1e-9),
}


@pytest.mark.parametrize("comb", sorted(COMBINATORS))
@pytest.mark.parametrize("wrap", sorted(WRAPPERS))
@pytest.mark.parametrize("order", ["wrapper_outside", "combinator_outside"])
def test_combinator_wrapper_matrix(comb, wrap, order):
    """Every combinator and output wrapper composes in either nesting order
    (meep's step-function protocol): the probe fires at the combinator's
    times, inside the wrapper's context."""
    make_wrap, in_context = WRAPPERS[wrap]
    make_comb, needs_cutoff, check_times = COMBINATORS[comb]
    rec = []
    if order == "wrapper_outside":
        step_func = make_wrap(make_comb(_probe(rec)))
    else:
        step_func = make_comb(make_wrap(_probe(rec)))
    sim = _sim()
    sim.init_sim()
    cut = sim._source_cutoff()
    sim.run(step_func, until=(cut + 2 if needs_cutoff else 4))
    tend, dt = sim.round_time(), sim.grid.dt
    times = [r[0] for r in rec]
    assert check_times(times, dt, tend, cut), (times, tend, cut)
    assert all(in_context(r) for r in rec), rec


def test_to_appended_at_every_stacks_output(tmp_path, monkeypatch):
    """The bent-waveguide tutorial's exact composition:
    to_appended("ez", at_every(dt, output_efield_z))."""
    monkeypatch.chdir(tmp_path)
    sim = _sim()
    sim.run(mp.at_beginning(mp.output_epsilon),
            mp.to_appended("ez", mp.at_every(1.0, mp.output_efield_z)),
            until=4)
    assert glob.glob("meep_torch-eps-*")
    files = glob.glob("meep_torch-ez.*")
    assert files
    if files[0].endswith(".npz"):
        arr = np.load(files[0])["ez"]
    else:
        import h5py
        with h5py.File(files[0]) as f:
            arr = f["ez"][...]
    assert arr.ndim == 3 and 4 <= arr.shape[0] <= 6   # frames stacked on time


# --- FluxData normalization workflow (bend-flux) ---
def _guide_sim():
    return mp.Simulation(
        cell_size=mp.Vector3(12, 6, 0), resolution=10,
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf),
                           center=mp.Vector3(),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                           component=mp.Ez, center=mp.Vector3(-4.5, 0),
                           size=mp.Vector3(0, 1))],
        boundary_layers=[mp.PML(1.0)])


def _refl_fr():
    return mp.FluxRegion(center=mp.Vector3(-3.5, 0), size=mp.Vector3(0, 2))


def test_get_flux_data_shape_and_minus_subtraction():
    sim = _guide_sim()
    refl = sim.add_flux(FCEN, DF, 5, _refl_fr())
    sim.run(until=40)
    fd = sim.get_flux_data(refl)
    assert isinstance(fd, mp.FluxData)
    assert np.iscomplexobj(fd.E) and fd.E.shape == fd.H.shape
    assert fd.E.shape[0] == 5                          # (nfreq, npts)
    base = np.array(mp.get_fluxes(refl))
    assert np.abs(base).max() > 0

    # meep normalization semantics: an identical run with the straight-run
    # data loaded negated accumulates (fields - fields) -> residual ~ 0
    sim2 = _guide_sim()
    r2 = sim2.add_flux(FCEN, DF, 5, _refl_fr())
    sim2.load_minus_flux_data(r2, fd)
    sim2.run(until=40)
    res = np.array(mp.get_fluxes(r2))
    assert np.all(np.abs(res) <= 1e-8 * np.abs(base).max()), res


def test_load_flux_data_positive_twin_quadruples_flux():
    sim = _guide_sim()
    refl = sim.add_flux(FCEN, DF, 5, _refl_fr())
    sim.run(until=40)
    fd = sim.get_flux_data(refl)
    base = np.array(mp.get_fluxes(refl))
    # loading +data doubles the accumulated DFT fields; flux is bilinear -> 4x
    sim2 = _guide_sim()
    r2 = sim2.add_flux(FCEN, DF, 5, _refl_fr())
    sim2.load_flux_data(r2, fd)
    sim2.run(until=40)
    quad = np.array(mp.get_fluxes(r2))
    np.testing.assert_allclose(quad, 4.0 * base, rtol=1e-9)


def test_y_normal_flux_monitor_matches_x_normal():
    """A centered Ez point source radiates symmetrically: the flux through a
    y-normal line at +y equals the flux through an x-normal line at +x (the
    bend-flux bent-run transmission monitor is y-normal)."""
    sim = mp.Simulation(cell_size=mp.Vector3(8, 8, 0), resolution=10,
                        sources=[mp.Source(mp.GaussianSource(FCEN, fwidth=DF),
                                           component=mp.Ez,
                                           center=mp.Vector3())],
                        boundary_layers=[mp.PML(1.0)])
    fx = sim.add_flux(FCEN, DF, 3, mp.FluxRegion(center=mp.Vector3(2, 0),
                                                 size=mp.Vector3(0, 2)))
    fy = sim.add_flux(FCEN, DF, 3, mp.FluxRegion(center=mp.Vector3(0, 2),
                                                 size=mp.Vector3(2, 0)))
    sim.run(until=30)
    a, b = np.array(mp.get_fluxes(fx)), np.array(mp.get_fluxes(fy))
    assert np.all(a > 0) and np.all(b > 0)             # both outgoing-positive
    np.testing.assert_allclose(b, a, rtol=0.05)


def test_add_near2far_3d_dipole_pattern():
    """3D near2far shim: a closed six-face box around a z dipole; the far
    pattern from get_farfield reproduces sin^2(theta), and get_farfields
    returns the meep dict of grid-shaped arrays."""
    import math
    sim = mp.Simulation(cell_size=mp.Vector3(3.2, 3.2, 3.2), resolution=10,
                        dimensions=3,
                        sources=[mp.Source(mp.GaussianSource(1.0, fwidth=0.5),
                                           component=mp.Ez,
                                           center=mp.Vector3())],
                        boundary_layers=[mp.PML(0.8)])
    faces = [mp.Near2FarRegion(center=mp.Vector3(*(0.6 * np.eye(3)[a] * s)),
                               size=mp.Vector3(*(1.2 * (1 - np.eye(3)[a]))),
                               weight=int(s))
             for a in range(3) for s in (+1, -1)]
    n2f = sim.add_near2far(1.0, 0, 1, *faces)
    sim.run(until=15)
    ff = sim.get_farfield(n2f, mp.Vector3(100, 0, 0))
    assert ff.shape == (6,) and np.all(np.isfinite(ff)) and np.abs(ff).max() > 0
    th = np.linspace(0.05, math.pi - 0.05, 18)
    p = np.array([np.sum(np.abs(sim.get_farfield(
        n2f, mp.Vector3(100 * math.sin(t), 0, 100 * math.cos(t)))[:3]) ** 2)
        for t in th])
    p, ref = p / p.max(), np.sin(th) ** 2
    rms = float(np.sqrt(np.mean((p - ref / ref.max()) ** 2)))
    assert rms < 0.05, f"far pattern rms vs sin^2: {rms}"
    d = sim.get_farfields(n2f, resolution=1.0, center=mp.Vector3(50, 0, 0),
                          size=mp.Vector3(0, 4, 4))
    assert sorted(d) == ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]
    assert d["Ex"].shape == (5, 5, 1)
    assert all(np.all(np.isfinite(v)) for v in d.values())


# --- upstream tutorials, import-swap only (subprocess smoke) ---
def test_tutorial_bent_waveguide(tmp_path):
    proc = subprocess.run([sys.executable, str(TUTORIALS / "bent_waveguide.py")],
                          cwd=tmp_path, capture_output=True, text=True,
                          timeout=1800)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert list(tmp_path.glob("meep_torch-eps-*"))
    assert list(tmp_path.glob("meep_torch-ez.*"))


def test_tutorial_bend_flux(tmp_path):
    proc = subprocess.run([sys.executable, str(TUTORIALS / "bend_flux.py")],
                          cwd=tmp_path, capture_output=True, text=True,
                          env=dict(os.environ, MPLBACKEND="Agg"), timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert (tmp_path / "out.png").exists()
