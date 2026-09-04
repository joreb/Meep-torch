# SPDX-License-Identifier: MIT
# Copyright (C) 2026 meep-torch contributors (original code)
"""W1 gate: the meep-shaped shim (meep_torch.mp) reproduces the raw engine.

For each gate example, a direct-engine reference (rasterize_scene + build_coeffs
+ LineSource/FluxMonitor + step/run, the calls the raw-port twins make) is
compared against the same setup built through `mp.Simulation`. The shim is
correct iff it wires the identical engine calls, so fields/fluxes match to
floating-point. (Hard-eps raw twins like straight-waveguide.py are a separate
artifact; this test isolates adapter correctness via the shared rasterize path.)

Gate examples (master spec W1): straight-waveguide, bent-waveguide, ring,
refl-quartz, gaussian-beam.
"""
import math

import numpy as np
import pytest
import torch

import meep_torch.mp as mp
from meep_torch.core.coeffs import build_tm_coeffs
from meep_torch.core.grid import Grid
from meep_torch.core.monitors import FluxMonitorTM
from meep_torch.core.run import run_tm
from meep_torch.core.sources import CWRamp, ContinuousSource, GaussianPulse, LineSource
from meep_torch.core.stepper_tm import init_tm_state, step_tm
from meep_torch.geometry.polygons import Polygon
from meep_torch.geometry.scene import rasterize_scene
from meep_torch.core.materials import SiO2

RTOL = 1e-5


def _rel_l2(a, b):
    a = torch.as_tensor(a, dtype=torch.float64); b = torch.as_tensor(b, dtype=torch.float64)
    return float((a - b).abs().max().item() / max(1e-30, (a.abs().max().item())))


def test_straight_waveguide():
    g = Grid(nx=160, ny=80, dx=0.1)
    poly = Polygon(np.array([[0, 3.5], [16, 3.5], [16, 4.5], [0, 4.5]]))
    eps = rasterize_scene([(poly, 12.0)], g, background=1.0).eps_tm
    c = build_tm_coeffs(g, eps_inf=eps, npml=10)
    src = LineSource(g, 10, 35, 45, profile=ContinuousSource(0.15))   # matches mp.ContinuousSource
    st = init_tm_state(g)
    for t in range(round(200 / g.dt)):
        st = step_tm(st, c, src.amplitude(t), src.mask)
    raw_ez = st.Ez

    sim = mp.Simulation(cell_size=mp.Vector3(16, 8, 0), resolution=10,
        geometry=[mp.Block(size=mp.Vector3(mp.inf, 1, mp.inf), center=mp.Vector3(),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.ContinuousSource(frequency=0.15), component=mp.Ez,
                           center=mp.Vector3(-7, 0), size=mp.Vector3(0, 1, 0))],
        boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
    sim.run(until=200)
    shim_ez = torch.as_tensor(sim.get_array(component=mp.Ez))
    assert _rel_l2(raw_ez, shim_ez) < RTOL


def test_bent_waveguide():
    g = Grid(nx=160, ny=160, dx=0.1)
    h = Polygon(np.array([[0, 4.0], [11.5, 4.0], [11.5, 5.0], [0, 5.0]]))      # horiz arm
    v = Polygon(np.array([[11.0, 4.0], [12.0, 4.0], [12.0, 16.0], [11.0, 16.0]]))  # vert arm
    eps = rasterize_scene([(h, 12.0), (v, 12.0)], g, background=1.0).eps_tm
    c = build_tm_coeffs(g, eps_inf=eps, npml=10)
    fcen = 1.0 / (2.0 * math.sqrt(11.0))
    src = LineSource(g, 10, 40, 50, profile=ContinuousSource(fcen, width=20))  # matches mp.ContinuousSource(width=20)
    st = init_tm_state(g)
    for t in range(3000):
        st = step_tm(st, c, src.amplitude(t), src.mask)
    raw_ez = st.Ez

    sim = mp.Simulation(cell_size=mp.Vector3(16, 16, 0), resolution=10,
        geometry=[mp.Block(size=mp.Vector3(12, 1, mp.inf), center=mp.Vector3(-2.5, -3.5, 0),
                           material=mp.Medium(epsilon=12)),
                  mp.Block(size=mp.Vector3(1, 12, mp.inf), center=mp.Vector3(3.5, 2, 0),
                           material=mp.Medium(epsilon=12))],
        sources=[mp.Source(mp.ContinuousSource(frequency=fcen, width=20), component=mp.Ez,
                           center=mp.Vector3(-7, -3.5), size=mp.Vector3(0, 1, 0))],
        boundary_layers=[mp.PML(1.0)])
    sim.run(n_steps=3000)
    shim_ez = torch.as_tensor(sim.get_array(component=mp.Ez))
    assert _rel_l2(raw_ez, shim_ez) < RTOL


def test_ring():
    n, w, r, dpml, sxy = 3.4, 1.0, 1.0, 2.0, 16.0
    g = Grid(nx=160, ny=160, dx=0.1)
    cx = cy = sxy / 2
    th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    outer = Polygon(np.stack([cx + (r + w) * np.cos(th), cy + (r + w) * np.sin(th)], 1))
    inner = Polygon(np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], 1))
    rm = rasterize_scene([(outer, n ** 2), (inner, 1.0)], g, background=1.0)
    c = build_tm_coeffs(g, eps_inf=rm.eps_tm, npml=20)
    src = LineSource(g, 91, 80, 81, profile=GaussianPulse(0.15, 0.1))
    st = init_tm_state(g)
    st, _ = run_tm(st, c, 8000, source=src, monitors=())
    raw_ez = st.Ez

    sim = mp.Simulation(cell_size=mp.Vector3(sxy, sxy, 0), resolution=10,
        geometry=[mp.Cylinder(radius=r + w, center=mp.Vector3(), material=mp.Medium(index=n), n_pts=64),
                  mp.Cylinder(radius=r, center=mp.Vector3(), material=mp.Medium(epsilon=1.0), n_pts=64)],
        sources=[mp.Source(mp.GaussianSource(frequency=0.15, fwidth=0.1), component=mp.Ez,
                           center=mp.Vector3(r + 0.1, 0))],
        boundary_layers=[mp.PML(dpml)], default_material=mp.Medium(epsilon=1.0))
    sim.run(n_steps=8000)
    shim_ez = torch.as_tensor(sim.get_array(component=mp.Ez))
    assert _rel_l2(raw_ez, shim_ez) < RTOL


def test_refl_quartz():
    sz, res = 12.0, 100
    g = Grid(nx=int(sz * res), ny=3, dx=1.0 / res)
    npml = res
    freqs = np.linspace(1.0 / 0.8, 1.0 / 0.4, 50)
    src = LineSource(g, npml, 0, g.ny, profile=GaussianPulse(1.875, 1.25))
    mon = FluxMonitorTM(g, freqs.tolist(), x_normal=True, idx=3 * res, lo=0, hi=g.ny)

    def raw_run(eps_inf, poles=(), masks=None):
        c = build_tm_coeffs(g, eps_inf=eps_inf, poles=poles, pole_masks=masks, npml=npml, y_pmc=True)
        st = init_tm_state(g, n_poles=len(poles))
        st, (acc,) = run_tm(st, c, int(round(80 / g.dt)), source=src, monitors=[mon])
        return mon.flux(acc, g.dx).numpy()
    empty = raw_run(1.0)
    slab = Polygon(np.array([[6.0, 0.0], [11.0, 0.0], [11.0, g.ny * g.dx], [6.0, g.ny * g.dx]]))
    rm = rasterize_scene([(slab, SiO2)], g, background=1.0)
    slab_f = raw_run(rm.eps_tm, rm.poles, rm.pole_masks)
    R_raw = -(slab_f - empty) / empty

    def shim_run(geom):
        sim = mp.Simulation(cell_size=mp.Vector3(0, 0, sz), resolution=res, dimensions=1,
            geometry=geom, sources=[mp.Source(mp.GaussianSource(frequency=1.875, fwidth=1.25),
               component=mp.Ex, center=mp.Vector3(z=-0.5 * sz + 1.0))],
            boundary_layers=[mp.PML(1.0)], default_material=mp.Medium(epsilon=1.0))
        fr = mp.FluxRegion(center=mp.Vector3(z=-0.25 * sz))
        name = sim.add_flux(1.875, 1.25, 50, fr)
        sim.run(n_steps=int(round(80 / sim.grid.dt)))
        return np.array(sim.get_fluxes(name))
    e_shim = shim_run([])
    s_shim = shim_run([mp.Block(size=mp.Vector3(mp.inf, mp.inf, 5.0), center=mp.Vector3(z=2.5),
                               material=mp.Medium(material=SiO2))])
    R_shim = -(s_shim - e_shim) / e_shim
    # fluxes match the raw engine bit-for-bit (same driver); R within 1e-4 abs.
    assert _rel_l2(empty, e_shim) < RTOL
    assert np.abs(R_shim - R_raw).max() < 1e-4


def test_gaussian_beam():
    s, res, w0 = 14, 20, 0.8
    g = Grid(nx=int(s * res), ny=int(s * res), dx=1.0 / res)
    npml = int(round(2.0 / g.dx))
    ix_src = int(round(((-0.5 * s + 2.0 + 1.0) + 0.5 * s) / g.dx))   # x=-4 -> ix=60
    src = LineSource(g, ix_src, 0, g.ny, profile=GaussianPulse(1.0, 0.3))
    y = torch.tensor([-0.5 * s + (j + 0.5) * g.dx for j in range(g.ny)])
    src.mask = g.zeros(g.nx, g.ny)
    src.mask[ix_src, :] = torch.exp(-(y ** 2) / (w0 ** 2))
    c = build_tm_coeffs(g, eps_inf=torch.ones(g.nx, g.ny, dtype=torch.float64), npml=npml)
    st = init_tm_state(g)
    for t in range(1000):
        st = step_tm(st, c, src.amplitude(t), src.mask)
    raw_ez = st.Ez

    sim = mp.Simulation(cell_size=mp.Vector3(s, s, 0), resolution=res,
        sources=[mp.Source(mp.GaussianSource(frequency=1.0, fwidth=0.3), component=mp.Ez,
                          center=mp.Vector3(-0.5 * s + 2.0 + 1.0, 0), size=mp.Vector3(0, s, 0),
                          amp_func=lambda p: math.exp(-(p.y ** 2) / (w0 ** 2)))],
        boundary_layers=[mp.PML(2.0)], default_material=mp.Medium(epsilon=1.0))
    sim.run(n_steps=1000)
    shim_ez = torch.as_tensor(sim.get_array(component=mp.Ez))
    assert _rel_l2(raw_ez, shim_ez) < RTOL